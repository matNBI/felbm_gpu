# felbm_gpu

A **separate, GPU-only** CUDA port of the felbm multiphase (Lee–Liu free-energy)
solver. It reuses the `felbm_local` host code for everything that is not
performance-critical (config parsing, geometry/TIFF init, the multiphase initial
condition, and the sparse stencil operators) and replaces the compute engine with
CUDA kernels. Single GPU, one subdomain, `felbm_gpu` binary only.

> **Status: validated (BGK, body-force/periodic, double precision).**
> The GPU engine reproduces the CPU `EngineMultiPhase` to **machine precision**
> (`max|Δ| ~ 2e-16` on the `h`/`g` distributions after 20 steps) in both the
> all-fluid and porous (obstacles + bounce-back + wetting BC) regimes — see
> `compare_cpu_gpu` under "Validation plan". Not yet ported: MRT collision, open
> boundaries, order-parameter mass correction, particle tracking (see the table
> below). Single precision (`-DFELBM_SINGLE`) builds but its drift is uncharacterised.

## Design: correct first, fast later

The CPU code precomputes every **stencil** — streaming, gradients, Laplacian, the
per-direction operators — as a sparse CSR matrix and applies it as a matrix–vector
product each step. felbm_gpu **uploads those exact CSR matrices** and applies them
with a device SpMV, so the stencils — including all the halfway bounce-back
near-wall terms, which are the easy thing to get subtly wrong — are **bit-for-bit
the CPU operators**. The CSR arrays are reached via thin `felbm_gpu` subclasses
(`operator_access.h`) that expose the operators' `protected` members, so **no edit
to `felbm_local` is required** — it builds against an unmodified checkout.

The genuinely new GPU code is therefore only the **pointwise, embarrassingly
parallel physics** (moments, chemical potential, equilibria, force term, collision
combine), ported kernel-by-kernel in `include/felbm_gpu/device_engine.cuh` with the
CPU source function named above each kernel.

This is the "naive but provably correct" port. It uses more memory than necessary
(stores the operators + many field temporaries) and SpMV is not the fastest GPU
pattern. **Once validated**, the roadmap is to replace the SpMV operators with
matrix-free stencil kernels and fuse passes — see `docs/PORTING_SPEC.md`.

## What is and isn't ported

| Ported | Not yet ported |
|---|---|
| BGK + MRT multiphase collision, Guo forcing | Open inlet/outlet boundaries (`use_open_bnd`) |
| Body-force / fully periodic runs | Multi-GPU / domain decomposition |
| Streaming + halfway bounce-back (CPU CSR) | XDMF metadata (compressed-index → xyz) |
| Gradients / Laplacian / dir operators (CPU CSR) | |
| Order-parameter mass correction (`correct_op_mass`) | |
| Particle tracking — D⊥ (host-side, on GPU velocity) | |
| Compile-time `double` (default) / `float` (`-DFELBM_SINGLE`) | |
| Minimal HDF5 field + particle dump; run log + timeseries | |

The GRL dispersion runs are body-force-driven and periodic, so they fall entirely in
the **ported** regime, including the D⊥ particle tracking (see below).

Particle tracking reuses the validated CPU `ParticleManager`: tracers live on the
host and are advected on the GPU velocity field, which is downloaded (compressed) and
scattered into the global grid. Enabled by the usual `particles_*` cfg keys.

Output defaults to **HDF5**: `<output_dir>/particles_<iter>.h5` with datasets
`position`, `velocity`, `id` (the velocity is interpolated from the GPU velocity field
the manager holds, so it needs no extra work). This uses `ParticleManager::
output_state`, so it requires a `felbm_local` with the HDF5 particle writer. Set
`particles_format = csv` in the cfg for the plain-text `id,x,y,z` fallback (works on
any `felbm_local`). The format key is read directly from the cfg, so it does not
depend on `felbm_local`'s `Settings`.

**Throttling the velocity copy.** The device→host velocity copy is the only
per-step GPU cost of tracking; advection itself is cheap host work. The
felbm_gpu-only cfg key `particles_velocity_skip = N` (default 1) refreshes the
velocity snapshot every `N` LBM steps and holds it constant in between, while still
advecting every step — so tracer time resolution (and hence D⊥) is unchanged, only
the copy frequency drops `N`-fold. It assumes the flow is quasi-steady over `N`
steps, which is exactly the regime in which D⊥ is measured; keep `N` well below the
timescale on which the velocity field changes.

## Build

Requires the CUDA toolkit (`nvcc`), plus HDF5 (C++) and libtiff — the same
libraries as `felbm_local`.

```bash
cd felbm_gpu
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80     # 80=A100, 86=RTX30, 89=RTX40, 90=H100
cmake --build . -j
```

Single precision (≈2× memory/throughput, larger domains, but changes numerics):

```bash
cmake .. -DFELBM_SINGLE=ON
```

`FELBM_LOCAL` defaults to `../felbm_local`; override with `-DFELBM_LOCAL=/path`.

## Run

```bash
cd bin && mkdir -p out
./felbm_gpu ../felbm_local/bin/settings.cfg      # model = multi_phase, use_open_bnd = false
```

Output: `<output_dir>/<output_name>_<iter>.h5` with compressed-site datasets
`density, concentration, u_x, u_y, u_z, pressure` (length = number of fluid sites).
These are 1-D fluid-only arrays with no built-in geometry, so they are compact but
not directly loadable in ParaView — see below.

## Visualization (ParaView / XDMF)

The solver does not write XDMF inline. With `output_xdmf = true` it writes a one-time
`<output_dir>/geometry.h5` (the grid coordinate of each fluid site, same index order
as the fields, plus the bounding size). Turn the run into a ParaView time series with:

```bash
python scripts/make_xdmf.py <output_dir>            # needs h5py
# then open <output_dir>/output.xdmf in ParaView
```

`make_xdmf.py` emits a Polyvertex (point-cloud) XDMF temporal collection: each fluid
site is a point at its grid coordinate, colored by `density`/`concentration`/
`pressure`, with `velocity` assembled as a vector from `u_x/u_y/u_z`. Options:
`--prefix` (field-file prefix, default `output`), `--dt` (LBM steps per snapshot, for
the Time axis), `--geom`. It rebuilds purely from the `.h5` files, so re-run it any
time more snapshots land. (A point cloud, not a filled volume — that suits the
compressed porous data and needs no solid cells; use ParaView glyphs / a Delaunay or
resample filter if you want a solid rendering.)

For the tracers (D⊥), add `--particles`:

```bash
python scripts/make_xdmf.py <output_dir> --particles   # writes particles.xdmf
```

This builds a *moving* point cloud — each snapshot uses its own `position` dataset as
geometry — with `velocity` as a vector and `id` as a scalar, and it handles a
changing particle count across snapshots. It needs HDF5 particle output
(`particles_format = h5`); it has no field/geometry dependency.

## Validation plan (do this before trusting results)

**Step 1 is automated by the `compare_cpu_gpu` harness** (built alongside
`felbm_gpu`). It runs the CPU `EngineMultiPhase` and the GPU `MultiPhaseGPU` from
the *same* initial distributions on a small fully-periodic droplet and reports the
max/mean absolute difference of the `h` and `g` distributions — the fundamental
state, so if these match everything derived matches. Both drive the *same*
`MultiPhaseGPU` used by the app, so the test validates the real code path.

```bash
./compare_cpu_gpu 1                        # 1 step, 48^3 (~0.8 GB GPU), ratio 5, all-fluid
./compare_cpu_gpu 20 48 5                  # 20 steps to watch drift accumulate
./compare_cpu_gpu 20 48 5 spheres         # WITH solid obstacles -> tests bounce-back
./compare_cpu_gpu 20 48 5 spheres mrt     # MRT collision + obstacles
# args: [steps=1] [N=48] [ratio=5] [geom=fluid|spheres] [coll=bgk|mrt];  N≈64 ~ 2 GB
```

`geom=spheres` inserts solid sphere obstacles (with `use_halfway_bb=true`), so the
run exercises the halfway bounce-back streaming and the biased-difference near-wall
stencils — the porous regime your GRL runs use. The all-fluid case never touches
those paths, so **run the `spheres` case before trusting porous-media results**.

In double precision expect `max|Δ| ~ 1e-11` or smaller after one step (the
operators are identical, so only floating-point ordering in the pointwise kernels
differs). A large diff localises the bug: if `g` is off but `h` is clean, look at
`k_equilibria`/`k_force_term`'s `g` terms, etc.

Then, by hand:

2. **Static droplet / Laplace law.** Spurious currents bounded, mass conserved,
   Δp ≈ 2σ/R — the same checks as `felbm_local` `test_mp_droplet`.
3. **Density-ratio sweep** at ratios 20 and 100 for stability.
4. **A GRL case** vs the CPU for a few thousand steps; compare bulk statistics.

## Layout

```
include/felbm_gpu/
  precision.h        real_t compile-time flag (double default / float)
  gpu_common.cuh     CUDA error checks, launch config, typed alloc/copy
  d3q19.cuh          D3Q19 constants (constant memory), matches the CPU set
  device_csr.cuh     device CSR + SpMV (single- and triple-value)
  operator_access.h  subclasses exposing the CPU operators' CSR (no felbm_local edit)
  device_engine.cuh  the pointwise CUDA kernels (moments, equilibria, force, ...)
src/felbm_gpu_main.cu  driver: reuse host setup, upload operators, run loop, output
docs/PORTING_SPEC.md   CPU→GPU mapping, formulas, and the optimisation roadmap
```
