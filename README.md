# felbm_gpu

A **separate, GPU-only** CUDA port of the felbm multiphase (Lee–Liu free-energy)
D3Q19 solver. It reuses the `felbm_local` host code for everything that is not
performance-critical (config parsing, geometry/TIFF init, the multiphase initial
condition, the particle tracker, and the sparse stencil operators) and replaces the
compute engine with CUDA kernels. Single GPU, one subdomain, `felbm_gpu` binary only.

> **Status: validated and in production use.**
> The GPU engine reproduces the CPU `EngineMultiPhase` to **machine precision**
> (`max|Δ| ~ 1e-15`) in both the all-fluid and porous (obstacles + halfway
> bounce-back + wetting BC) regimes, for BGK and MRT, on every operator path —
> run `scripts/validate.sh`. Single precision (`-DFELBM_SINGLE`) reproduces it to
> the float floor (`~1e-7`).

## Quick start

```bash
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=86     # 80=A100, 86=RTX30, 89=RTX40, 90=H100
cmake --build . -j
cd .. && scripts/validate.sh build         # confirm the port on this machine

mkdir run && cd run && cp ../cfg/2d_cylinders/*.cfg . && mkdir out
../build/felbm_gpu settings.cfg
```

Two complete, tested example configurations are in `cfg/`:

| config | geometry | needs |
|---|---|---|
| `cfg/2d_cylinders/` | quasi-2D (`size_z = 1`) random cylinder pack, **auto-generated** | nothing |
| `cfg/3d_image/` | 3D porous medium from a **segmented TIFF stack**, periodic body force | your image; set `image_dir` in `domain.cfg` |
| `cfg/3d_image_openbnd/` | the same sample **downsampled to 150³**, pressure-driven with **open inlet/outlet** | your image |

Both are body-force-driven, fully periodic, MRT, with a slab initial condition and
passive tracers carrying a ladder of molecular diffusivities on both interfaces —
i.e. the production dispersion setup. Each folder holds `settings.cfg` (run + solver),
`params.cfg` (fluid physics), `domain.cfg` (geometry) and `fluid.cfg` (initial
condition). Every key is commented; `[gpu]` marks felbm_gpu-only keys.

## What is and isn't ported

| Ported | Not ported |
|---|---|
| BGK + MRT multiphase collision, Guo forcing | Multi-GPU / domain decomposition |
| Body-force / periodic **and** open inlet/outlet boundaries, with `copy_to_buffers` | |
| Streaming + halfway bounce-back | |
| Gradients / Laplacian / per-direction operators | |
| Order-parameter mass correction | |
| Particle tracking with per-particle diffusivities | |
| Checkpoint / restart | |
| `double` (default) and `float` (`-DFELBM_SINGLE`) | |
| HDF5 field + particle output, run log, timeseries | |

## Performance

Six cfg keys switch the solver between the correctness reference and the fast path.
All default **off** (= reference), and **all are validated exact** — the fast path is
not an approximation:

| key | effect |
|---|---|
| `stream_matrix_free` | streaming from a compact source-index table instead of stored CSR |
| `grad_matrix_free` | all six field operators (gradients, Laplacian, dir operators) matrix-free |
| `fused` | recompute directional derivatives inside the equilibria/force kernels |
| `fuse_collision` | equilibria + force + collision + apply in **one** per-site kernel |
| `mrt_fast_transform` | `real_t` MRT moment transform instead of double |
| `stream_inplace` | reversed-slot collision writes ⇒ in-place streaming, drops `h2`/`g2` |

Production setting is all six `true`. Three further keys tune the host side:
`particles_overlap`, `particles_velocity_skip`, and `checkpoint_skip` / `restart_file`.

**Measured, RTX 3090, single precision, fused MRT, 300³ percolating image
(9,949,099 fluid sites):**

| configuration | MLUPS |
|---|---|
| matrix-free + fused, no `mrt_fast_transform` | 65 |
| + `mrt_fast_transform` | **174** |
| ditto, minimal logging, no field dumps | 198 |
| with 12k tracers, `particles_overlap` | ~107 |

**Historical progression (A5000, 150³ porous, double)** — the optimisation ladder
that got here: CSR ~19.6 → matrix-free ~37 → fused **45.8 MLUPS (2.34×)**, with
per-site memory ~5.6 → ~3.4 → **~1.7 KB**, which is what made a 300³ porous run fit
a 24 GB card in double precision.

**Subsystem costs** (300³, RTX 3090), for choosing cfg values:

- Field dump (`file_skip`): ~208 ms each, gzip off — pinned batched D2H writing
  native `real_t` straight to HDF5.
- Log event (`log_skip`): ~ms. GPU-side reductions on a side stream, 48 bytes back
  per event, so dense monitoring is nearly free.
- Tracer step: ~1 ms with `particles_overlap = true`, as long as the GPU step is
  longer than the host scatter (~50–60 ms at 300³ with 12k tracers).

**This is not as fast as the code could reasonably be.** The remaining known levers,
roughly in order of expected value:

1. **GPU-side particle advection.** Tracers are still advected on the host on a
   downloaded velocity field. Overlap hides it today, but it caps how far the GPU
   step can shrink before the host becomes the critical path.
2. **AoSoA / coalescing** of the `k*n+i` layout for the memory-bound kernels.
3. **In-place-streaming op-table compression** — a per-site code table instead of
   three ~9n op arrays would roughly halve their ~740 MiB at 300³.
4. **Multi-GPU** for domains beyond one card.

After `mrt_fast_transform` there is no single hotspot left: the kernel profile is
flat and bandwidth-bound (`k_collide_fused_mrt` ~39%, then streaming, Laplacian and
gradients at ~8–12% each), so further single-GPU gains have to come from reducing
memory traffic rather than from optimising one kernel.

## Order-parameter mass correction

`correct_op_mass = true` removes the Lee–Liu scheme's intrinsic drift in total
order-parameter mass. This section documents it in full because it modifies the
distributions every step and is easy to misread as non-conservative.

### Why the drift exists

The order parameter `c` (0 = one phase, 1 = the other) is carried by the `h`
distribution, whose equilibrium correction terms are velocity-dependent. The scheme
therefore conserves `c` only to discretisation order, not exactly. Over 10^5–10^6
steps the accumulated error is enough to move the phase volumes measurably, which
matters for a long dispersion run where the slab is supposed to keep its size.

### What is conserved, and how

Let

- `c_i = sum_k h_k[i]` — the order parameter at site `i` (its zeroth moment),
- `M  = sum_i c_i` — the current total mass, over **streamed, non-ghost fluid sites**,
- `M0` — the target, recorded once immediately after the initial condition is
  uploaded (`record_target_mass`),
- `phi_i = c_i (1 - c_i)` — an **interface indicator**: ~0 in either bulk phase
  (`c ~ 0` or `c ~ 1`), maximal at `c = 1/2`, i.e. inside the diffuse interface,
- `W  = sum_i phi_i`.

Each step, after streaming, the drift `M - M0` is redistributed with weight `phi`:

```
lambda  = (M - M0) / W
dc_i    = -lambda * phi_i
h_k[i] += w_k * dc_i          for every direction k
```

Three properties make this a legitimate correction rather than a fudge:

1. **It restores the total exactly.**
   `sum_i dc_i = -lambda * sum_i phi_i = -lambda*W = -(M - M0)`, so the new total is
   exactly `M0` — algebraically exactly, up to floating-point summation.

2. **It adds no momentum.** The correction is injected along the lattice weights
   `w_k`. Since `sum_k w_k = 1`, the zeroth moment of `h` at site `i` changes by
   exactly `dc_i`; since `sum_k w_k e_k = 0`, the first moment does not change at
   all. The velocity field is untouched.

3. **It is interface-localised.** Because the weight is `phi_i = c_i(1-c_i)`, the
   correction is applied *inside the diffuse interface*, where the order parameter is
   genuinely ambiguous, and is ~0 in the bulk. It nudges the interface position by a
   fraction of a lattice unit rather than uniformly rescaling both phases — which
   would be the physically wrong way to absorb the same error.

### Implementation on the GPU

Two kernels per step (`device_engine.cuh`), both over fluid sites:

- **`k_mass_weight`** — one block-level shared-memory reduction producing `M` and `W`
  in a single pass, finished with a `double atomicAdd` into a 2-element device
  buffer. `c_i` is recomputed from `h` rather than read from the `c` field, so the
  reduction is consistent with the distributions it is about to modify.
- **`k_inject_mass`** — applies `h_k[i] += w_k * dc_i`, skipping sites with
  `phi_i <= 0` (bulk, and any site where round-off pushed `c` outside `[0,1]`).

Both accumulate in `double` regardless of `real_t`, and skip non-streamed sites so
the sum matches the CPU `MassConservationCorrector` domain exactly. Requires compute
capability >= 6.0 for double `atomicAdd` (any supported architecture).

### Two consequences worth knowing

- **`total_c` in the run log becomes a flat line.** That is the corrector working,
  not a sign that nothing is happening. A drift that grows steadily before correction
  means the interface is under-resolved (raise `interface_width`) or the time step is
  marginal.
- **It makes runs bitwise non-reproducible across rebuilds.** The reduction uses
  `atomicAdd`, so the summation order depends on block scheduling and can change
  between compilations. The injected correction then differs at the ~1e-12 level —
  immaterial for the physics, but chaotic advection amplifies it, so **individual
  tracer trajectories are not reproducible across builds with the corrector on.**
  For a bitwise A/B comparison of two builds, set `correct_op_mass = false` or
  compare within one binary. Bulk statistics are unaffected.

## Validation

Validation is built into the code, not a separate procedure. `compare_cpu_gpu` is
built alongside `felbm_gpu` and runs the CPU `EngineMultiPhase` and the GPU
`MultiPhaseGPU` from the *same* initial distributions, then reports max/mean absolute
difference on `h` and `g` — the fundamental state, so if these match, every derived
field matches. Both sides drive the *same* `MultiPhaseGPU` the app uses, so it tests
the real code path, not a replica.

```bash
scripts/validate.sh build      # the standard matrix (11 cases, a few minutes)
```

That covers the CSR reference path, matrix-free operators, both fusion stages,
`mrt_fast_transform` and `stream_inplace`, for BGK and MRT, on both geometries.
Pass criteria: `~1e-15` in a double build, `~1e-6` in a `-DFELBM_SINGLE` build (pure
float rounding).

Single cases directly:

```bash
# args: [steps] [N] [ratio] [geom=fluid|spheres] [coll=bgk|mrt] \
#       [mf] [mfg] [fused] [fusecoll] [mrtfast] [inplace]
./build/compare_cpu_gpu 20 48 5 spheres mrt 1 1 0 1 1 1     # full production path
./build/compare_cpu_gpu 20 48 5 fluid   bgk 0 0 0 0 0 0     # CSR reference
```

**Always validate a `spheres` case before trusting porous results.** It is the only
one that exercises halfway bounce-back streaming, the biased near-wall stencils and
the node-centred Laplacian with the wetting boundary condition; the all-fluid case
never touches those paths. `N ~ 48` needs ~0.8 GB of GPU memory, `N ~ 64` ~2 GB.

A large difference localises the bug rather than just failing: if `g` is off while
`h` is clean, look at the `g` terms in the equilibria/force kernels, and so on.

Two run-time self-checks also fire automatically:

- `stream_inplace` builds its op program at init and **verifies it on the host**
  against the gather semantics with a random vector, printing
  `stream_inplace program verified: ... ops`. If the geometry produces an index
  structure it cannot reduce, it falls back to the ping-pong path with a warning
  rather than running a wrong program.
- `restart_file` checks the checkpoint's state size against the current geometry and
  aborts with a clean message on mismatch.

## Open boundaries

`use_open_bnd = true` replaces the periodic body-force drive with an inlet/outlet
pair: fluid is injected on one face and drains through the opposite one. Ready-to-run
example in `cfg/3d_image_openbnd/` (a downsampled 150³ image; contrast with
`cfg/3d_image/`, the periodic version of the same sample).

### What is and isn't periodic

Two separate things decide this, and conflating them is easy:

- The **streaming operator** wraps arithmetically on every axis, unconditionally
  (`x_from = (x + N - c_m) % N`). There is no per-axis switch.
- The **geometry** decides whether that wrap is physically meaningful. A population
  wrapping into a solid node simply bounces back, so a wall of solid on a face closes
  that direction regardless of the streaming arithmetic.

For `domain_geometry = 3d_image`, `image_periodic` controls the geometry side:

| `image_periodic` | transverse to the flow | open boundaries |
|---|---|---|
| `true` | **periodic** — no padding layers, the lattice genuinely wraps | **fails**: grains land on the inlet/outlet faces and buffer-vertex identification asserts |
| `false` | **solid walls** — one solid layer stamped on each transverse face (unless `transverse_periodic = true`, below) | works |

Since open boundaries require `image_periodic = false`, the default combination gives
confining side walls. **`transverse_periodic = true` removes them**: the two axes normal
to the flow get no padding layer and no stamped wall, while the flow axis keeps its
`empty_layers` for the inlet/outlet. That is the usual choice for a representative
sample — laterally periodic, open along the flow.

```
use_open_bnd        = true
image_periodic      = false   # required (grains must not land on the inlet/outlet faces)
transverse_periodic = true    # no side walls; the non-flow axes stay periodic
```

Verified on the 150³ example: the domain loses its transverse padding (76x79x76 ->
75x79x75) and both transverse faces carry fluid, where with walls the low faces had
none. Default is `false`, so existing configurations are unchanged.

Open boundaries themselves do not change connectivity — they overwrite the inlet and
outlet planes every step, so whatever wraps from the outlet is discarded when the inlet
is rewritten. It is the stamped solid layers, not the boundary condition, that close
the transverse directions.

### Configuration

```
use_open_bnd    = true
in_out_dir      = 1          # flow axis: 0=x, 1=y, 2=z. The inlet/outlet are the
                             # two faces normal to it.
empty_layers    = 2          # clear fluid layers at those faces
extrude_buffers = true
buffer_layers   = 2
copy_to_buffers = true       # see "Direct vs. copy_to_buffers" below
image_periodic  = false      # REQUIRED (3d_image): see below

use_inlet_pressure  = true   ; pressure_inlet  = 1.005
use_outlet_pressure = true   ; pressure_outlet = 1.000
use_inlet_velocity  = false  # or true, with u_inlet_x/y/z

use_inlet_fluid     = true   ; inlet_fluid  = 0    # inject a defined phase
use_outlet_fluid    = false                        # open drain: both phases leave
correct_op_mass     = false  # REQUIRED
```

One setting **aborts** the run rather than silently doing something else:

- **`correct_op_mass = true`** — with fluid entering and leaving, the total order
  parameter is legitimately not conserved, and the corrector would fight the flow.

And one that asserts inside the geometry builder rather than failing cleanly:

- **`image_periodic = true` with `domain_geometry = 3d_image`.** Mirroring the image
  to make it periodic puts grains on the inlet/outlet faces, and buffer-vertex
  identification then fails with *"buffer vertex reference does not exist"*. Set it
  false; periodicity along the flow axis is meaningless when the BC overwrites those
  planes anyway.

### Direct vs. `copy_to_buffers`

The streaming operator wraps unconditionally on every axis, including the flow axis
(see above). `extrude_buffers` pads the domain with `buffer_layers` extra nodes beyond
the inlet/outlet plane specifically so this wrap links buffer-to-buffer across the two
ends of the domain, not real fluid node to real fluid node -- but that only holds if
the buffer nodes are prevented from colliding and streaming like ordinary fluid. Two
ways to do that:

- **`copy_to_buffers = false`** (direct): `k_open_bnd` overwrites the true inlet/outlet
  nodes' distributions every step. The buffer nodes beyond them are left to collide
  normally, which turns them into a short periodic loop connecting the outlet-side
  buffers to the inlet-side buffers (`buffer_layers` nodes each way) -- fluid injected
  at the inlet leaks into the outlet's buffer neighbour in a handful of steps,
  independent of the true domain length, instead of taking O(N) steps to physically
  advect there. With `buffer_layers = 2` this is a real, measurable contamination
  (concentration at the outlet-adjacent layer pulled toward the inlet composition),
  not just a rounding-level artifact.
- **`copy_to_buffers = true`** (ghost): the true inlet/outlet nodes collide normally
  under their imposed macroscopic fields; only their buffer neighbours are overwritten
  each step with the equilibrium of the vertex they reference (`k_open_bnd_buf_fields` /
  `k_open_bnd_buf`), breaking the periodic loop at the source. This is the mode the CPU
  reference actually validates the physics against and the recommended default.

`copy_to_buffers = true` needs the buffer node fields excluded from the velocity/
pressure correction the same way the inlet/outlet nodes are (they are "fully imposed"
nodes, not corrected); missing that lets the correction overwrite a buffer's pressure
between the concentration and pressure gradient passes, corrupting the pressure
gradient at the adjacent domain node. Validated against the CPU to machine precision
alongside the direct path (`compare_cpu_gpu geom=openbnd ... <mode> 1`, the trailing
`1` selects `copy_to_buffers`).

### Injection: single phase or co-injection

`inlet_mode` controls what is injected at the inlet. The outlet is unaffected — with
`use_outlet_fluid = false` it always takes the local composition and density, so both
phases drain freely whatever the inlet is doing.

| `inlet_mode` | behaviour | parameters |
|---|---|---|
| `single` *(default)* | one phase throughout — the original behaviour | `inlet_fluid` |
| `alternate` | slugs in time: `inlet_fluid` for a fraction of each cycle, the other phase for the rest | `inlet_period`, `inlet_duty` |
| `split` | side-by-side co-injection across the inlet face | `inlet_split_dir`, `inlet_split_pos` |

`inlet_ramp` tanh-smooths the switch — in **steps** for `alternate`, in **lattice units**
for `split`. Do not leave it at 0 at appreciable density contrast: a step change in the
composition imposes an interface with no diffuse profile, and the resulting
chemical-potential spike scales with the density difference. Set it comparable to
`interface_width` (space), or to the time an interface needs to advect its own width.

For `alternate` the injecting window is centred in the period, so the profile is
continuous across the cycle wrap; a window placed at the period boundary would
reintroduce exactly the discontinuity the smoothing exists to remove.

The imposed density follows the blended composition, `rho = C*rho0 + (1-C)*rho1`, so a
partially blended inlet injects a consistent (C, rho) pair rather than a mismatched one.

### Implementation note

The boundary condition has **two halves**, and both matter:

1. the **distributions** on the boundary nodes are overwritten with the equilibrium
   built from the prescribed values (`k_open_bnd`);
2. the **macroscopic fields** `c`, `rho`, `p`, `u` are imposed on those nodes
   independently of the distributions (`k_open_bnd_fields`), from inside
   `compute_fields`, before the gradients consume them.

Half (2) is easy to overlook: at an inlet node the code deliberately carries e.g.
`c = 1` while the distributions there sum to ~0. That is the intended meaning of a
Dirichlet boundary value, not an inconsistency — and the collision reads the *field*.
Nodes whose fields are imposed are also excluded from the velocity/pressure correction,
or it would overwrite the prescribed `u` and `p`.

Validated against the CPU to machine precision (double, 20 steps, max|Δh|): single
1.7e-16, alternate 2.2e-16, split 1.7e-16, local-values 1.7e-16, MRT 1.7e-16 — see
`compare_cpu_gpu geom=openbnd` (argument 13 selects the injection mode). Repeated with
`copy_to_buffers = true` (argument 14): single 1.7e-16, alternate 1.7e-16, split
1.7e-16, stripes 1.7e-16 — identical floor to the direct path.

## Checkpoint / restart

```
checkpoint_skip = 20000                        # write checkpoint_<step>.h5 every N
restart_file    = ./out/checkpoint_20000.h5    # resume from one
```

The checkpoint stores the raw `h`/`g` distributions — the true LBM state — plus the
step index and the mass-correction target `M0`. **The field HDF5 output is not a
restart source:** it holds only macroscopic moments, and the non-equilibrium part of
`h`/`g` cannot be recovered from them.

Distributions are stored as `double`, so a checkpoint written by a single-precision
build restarts a double build and vice versa. On restart the loop resumes at the
checkpoint step, the original `M0` is kept (recomputing it would lock the drift to
the wrong target), and tracers are restored exactly from the matching
`particles_<step>.h5`.

Exactness depends on precision: in a **double** build the continuation is bit-exact
(particle positions identical to the last bit). In a **single** build the fields match
to the float floor, but tracer *trajectories* diverge over a few hundred steps,
because velocity interpolation amplifies that ~1e-6 field difference chaotically. Use
double precision when exact trajectory continuation matters.

## Particle tracking

Tracers reuse the validated CPU `ParticleManager`: they live on the host and are
advected on the GPU velocity field, which is downloaded (compressed by the subdomain)
and scattered into the global grid. Enabled with the usual `particles_*` keys; output
is HDF5 by default (`particles_format = csv` for the plain-text fallback).

**Per-particle diffusivities.** `particles_dm_groups = "Dm:count, Dm:count, ..."`
gives tracers different molecular diffusivities in one run, so a single flow field
yields dispersion at several `Dm` values sharing identical pore realisations. The
per-particle values are written as a `Dm` dataset alongside
`position`/`velocity`/`id`. Zero-diffusivity tracers skip the RNG entirely, so they
cost nothing.

**Multi-plane seeding.** `particles_init_mode = plane_xz` with
`particles_plane_offsets = "512, 1536"` seeds several planes normal to the flow —
e.g. both faces of a slab. Each plane resamples until its full quota of accepted
(non-solid) points is reached, so planes cutting different amounts of grain still get
equal counts. A `plane_id` dataset records which plane each tracer started on. When
the group counts sum to `particles_number / n_planes`, the `Dm` ladder is applied
**within** each plane, giving a crossed design (every plane carries every
diffusivity) rather than the first plane taking the first groups.

**2D runs.** With `size_z = 1` the out-of-plane Brownian kick is suppressed
automatically — otherwise tracers diffuse in `z` and wrap periodically, which is
unphysical and contaminates any transverse-variance estimate.

**Throttling the velocity copy.** `particles_velocity_skip = N` refreshes the
velocity snapshot every `N` steps while still advecting every step, so tracer time
resolution is unchanged and only the copy frequency drops. It assumes the flow is
quasi-steady over `N` steps; keep `N` well below the timescale on which the velocity
field changes. With `particles_overlap = true` (default) the host tracer work is
hidden behind the next step's kernels, which is usually the better lever.

## Output and visualisation

Fields go to `<output_dir>/<output_name>_<iter>.h5` as fluid-only 1-D arrays
(`density, concentration, u_x, u_y, u_z, pressure`), so they are compact but carry no
geometry. Storage keys (all felbm_gpu-only):

- `output_float32` — write `float32` instead of `float64`. Halves the size, ample for
  visualisation, XDMF-safe.
- `output_deflate` — gzip 0–9, **default 0**. ParaView's Xdmf3 reader can crash on
  gzipped datasets read through XDMF, so compression is off by default. Use it for
  archival, and decompress before loading:
  `for f in out/*.h5; do h5repack -f NONE "$f" "unc/$(basename $f)"; done`

With `output_xdmf = true` the run also writes a one-time `geometry.h5` (grid
coordinate of each fluid site, same index order as the fields). Turn a run into a
ParaView time series with:

```bash
python scripts/make_xdmf.py <output_dir>              # fields (point cloud)
python scripts/make_xdmf.py <output_dir> --particles  # tracers
python scripts/make_xdmf.py <output_dir> --volume     # fields as a dense 3-D volume
```

The default emits a Polyvertex XDMF temporal collection: a point cloud, which suits the
compressed fluid-only storage and loads quickly, but which ParaView cannot volume-render
or slice directly.

`--volume` scatters the fluid-only arrays back onto the full lattice and writes
`vol_<prefix>_<iter>.h5` alongside the originals, with an ImageData (`3DCoRectMesh`)
XDMF. Volume rendering, slices, contours and stream tracing then work directly, with no
resampling: each value is placed at its own grid coordinate, so nothing is interpolated
across the grain space. Solid nodes get a sentinel — NaN by default, which ParaView
renders as blank and `Threshold` removes; pass `--solid-value 0` (or the
`density_solid` value) if you prefer a number. Note the cost: the dense grid carries the
solid too, so a 300³ run is 108 MB per field per snapshot against ~17 MB for the point
cloud. Use `--fields concentration,u_y` to densify only what you need, and `--overwrite`
to rebuild. Velocity is written as three scalar components because ParaView's Xdmf3
reader can crash on time-step when velocity is an XDMF `Function` vector; rebuild it
with the **Merge Vector Components** filter, or pass `--vector-velocity` if your
reader handles Function items.

## Analysis scripts

- `scripts/make_xdmf.py` — build ParaView XDMF for fields (point cloud or dense
  volume via `--volume`) and tracers.
- `felbm_local/scripts/dispersion_by_dm.py` — displacement-covariance tensor per
  diffusivity group and per seed plane, computed by FFT in O(T log T). Reports the
  **central** second moment (mean drift subtracted); `--dims 2` restricts the
  transverse variance to the in-plane axis for `size_z = 1` runs.

## Layout

```
cfg/2d_cylinders/      ready-to-run 2D example (auto-generated geometry)
cfg/3d_image/          ready-to-run 3D example (TIFF stack)
scripts/validate.sh    standard CPU-vs-GPU validation matrix
scripts/make_xdmf.py   ParaView XDMF generator
include/felbm_gpu/
  precision.h          real_t compile-time flag (double default / float)
  gpu_common.cuh       CUDA error checks, launch config, typed alloc/copy
  d3q19.cuh            D3Q19 constants in constant memory, matches the CPU set
  device_csr.cuh       device CSR + SpMV (the correctness reference path)
  operator_access.h    subclasses exposing the CPU operators' CSR (no felbm_local edit)
  device_engine.cuh    all CUDA kernels (moments, equilibria, force, collision, ...)
  multiphase_gpu.cuh   MultiPhaseGPU: allocation, init, step sequence, reductions
src/felbm_gpu_main.cu  driver: host setup, upload, run loop, output, checkpointing
src/compare_cpu_gpu.cu the validation harness
docs/PORTING_SPEC.md   CPU->GPU mapping, exact formulas, measurement tables
```

`FELBM_LOCAL` defaults to `../felbm_local`; override with `-DFELBM_LOCAL=/path`.
Requires the CUDA toolkit, HDF5 (C++) and libtiff — the same libraries as
`felbm_local`.
