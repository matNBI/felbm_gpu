# felbm_gpu — session handoff / continuation brief

Portable context for picking this work up in a fresh session. The repo is the source of
truth for *what the code does*; this document captures the state of play, the open
questions, and the things that are only obvious after having been bitten by them.

Both `felbm_gpu` and `felbm_local` are committed and pushed as of this writing.
Everything below is on `main`.

---

## 1. Current state

felbm_gpu is a validated single-GPU CUDA port of the multiphase (Lee–Liu free-energy)
D3Q19 solver, in production use for two-phase dispersion studies. It reuses `felbm_local`
for config parsing, geometry/TIFF init, the initial condition and the particle tracker,
and replaces the compute engine with CUDA kernels.

**Correctness** is established by `compare_cpu_gpu`, which drives the CPU
`EngineMultiPhase` and the GPU `MultiPhaseGPU` from the same initial distributions and
compares `h`/`g` node by node. `scripts/validate.sh` runs the standard 11-case matrix
(CSR reference, matrix-free, both fusion stages, `mrt_fast_transform`, `stream_inplace`;
BGK and MRT; fluid and spheres). Pass: ~1e-15 double, ~1e-6 single. **Run it after any
kernel change** — every optimisation below is exact, and that is the property worth
protecting.

**Optimisation flags** (all default off = reference path; all validated exact):
`stream_matrix_free`, `grad_matrix_free`, `fused`, `fuse_collision`,
`mrt_fast_transform`, `stream_inplace`. Production sets all six true.
Host-side: `particles_overlap` (default true), `particles_velocity_skip`,
`particles_threads`. Restart: `checkpoint_skip`, `restart_file`.

**Performance** (RTX 3090, 300³ percolating image, 9,949,099 fluid sites, single, MRT):

| configuration | MLUPS |
|---|---|
| matrix-free + fused, no `mrt_fast_transform` | 65 |
| + `mrt_fast_transform` | 174 |
| ditto, minimal logging, no field dumps | 198 |
| with 12k tracers, `particles_overlap` | ~107 |

Subsystem costs: field dump ~208 ms (gzip off); log event ~ms (GPU-side reductions,
48 B back per event — dense monitoring is nearly free); tracer step ~1 ms with overlap.

**Post-optimisation profile** (nsys, 300³, after `mrt_fast_transform`):
`k_collide_fused_mrt` 38.9% (down from 60%), then streaming, Laplacian and gradients at
~8–12% each, ~53 ms/step GPU total. **The profile is now flat and bandwidth-bound.**
There is no single hotspot left; further single-GPU gains must come from reducing memory
traffic, not from optimising one kernel.

## 2. What landed, and where

| commit | change |
|---|---|
| `1fc2b41` | tracking: pinned D2H + fused parallel scatter + overlap worker (~494 → ~1 ms/step) |
| `072b9d3` | persistent overlap worker (removes per-step OMP team spin-up) |
| `0a65aca` | `mrt_fast_transform`: real_t MRT moment transform (65 → 174 MLUPS) |
| `963bacd` | `d_relax` Vn→n shrink (exact; ~0.7 GB single / ~1.4 GB double at 300³) |
| `e97a75f` | `stream_inplace`: reversed-slot writes + host-verified disjoint op program; drops `d_h2`/`d_g2` and `d_src` |
| `0313e50` | monitoring: async GPU-side flow stats + buffered timeseries flush |
| `ffb0f56` | h5 output: pinned batched async D2H + native real_t write (1249 → 208 ms/dump) |
| `8ab369b`, `d313430` (local) | `slab_interface` initial condition |
| `27bbe51`, `0661dfa` (local) | checkpoint / restart |
| `6821a6f` | README rewrite, `cfg/` examples, `scripts/validate.sh` |

Per-site memory across the ladder: ~5.6 → ~3.4 (matrix-free) → ~1.7 kB (fused), which is
what makes a 300³ porous run fit a 24 GB card in double precision.

---

## 3. Remaining optimisations

The 2.3×-then-2.7× era is over: the memory-bandwidth wins have been taken and the
kernel profile is flat. What follows is honest about expected value, and **none of it is
urgent** — at 174–198 MLUPS the solver is no longer the constraint on the science.

**Worth doing if the host becomes the critical path:**

1. **GPU-side particle advection.** Tracers are still advected on the host on a
   downloaded velocity field. `particles_overlap` hides this today (~1 ms/step effective)
   *provided the GPU step is longer than the host scatter*, which is ~50–60 ms at 300³
   with 12k tracers. The margin is real but not large: if the GPU step ever drops below
   that — a faster card, a smaller domain, or more tracers — the host becomes the
   critical path and this becomes the top item. It is also the only way to remove the
   velocity D2H entirely. Biggest change of anything here.

**Memory, if a bigger domain is wanted:**

2. **In-place-streaming op-table compression.** The `stream_inplace` program stores three
   arrays of ~9n ops (9 B/op, ~740 MiB at 300³). A per-site code table would roughly
   halve that. Only worth it if memory, not speed, is the binding constraint.
3. **AoSoA / coalescing** of the `k*n+i` layout for the memory-bound kernels. This is the
   one remaining *speed* lever consistent with the flat bandwidth-bound profile, and also
   the most speculative: it is a pervasive layout change, it would touch every kernel,
   and the payoff is unknown without a prototype. If someone wants a performance project,
   this is it — but measure a single kernel first before committing.
4. **Multi-GPU / domain decomposition** for domains beyond a single card. Large, and
   unnecessary while 300³ fits comfortably.

**Explicitly NOT worth doing (measured, don't repeat these):**

- *Sparse MRT moment transform.* The original plan was a hand-coded sparse walk over the
  mostly-zero M / M⁻¹. Unnecessary: dense FP32 is throughput-free on Ampere, and fixed
  loop indices keep the 19-element arrays in registers, whereas runtime column indices
  would force spills. `mrt_fast_transform` (same algebra, same summation order, in
  `real_t`) got the full win with none of the risk.
- *Reducing logging frequency.* Measured at ~18 ms/log event before the async rework and
  effectively free after it. `log_skip = 10` costs nothing; earlier claims that logging
  was ~40% of wallclock were a misattribution of the **field download** cost.
- *Capping the tracer OMP team.* `particles_threads` helped only while the worker was
  re-created per step. With the persistent worker the full-width default wins.

## 4. Open issues

**Order-parameter mass correction is non-local.** See
`~/code/docs/MEMO_mass_correction_nonlocality.md` for the full diagnosis. Summary: the
corrector uses a single global λ, so it conserves globally but not locally and can move
phase between regions with no transport connecting them. Measured on the 2D slab case:
the drainage front generates 1.7× more error per unit interface than the imbibition
front, so 13% of the applied correction is spurious transport (0.17% of phase volume per
1e6 steps, against the 1.4% drift it removes). We already implement the best scheme in
that literature line (Kim, Lee & Choi 2014 — verified algebraically identical to ours),
so fixing it means leaving the family: flux-form correction, or conservative Allen–Cahn.
Not urgent, but it should be bounded before any claim that rests on ganglion
connectivity. The cheap first step is comparing ganglion size distributions with the
corrector on vs off.

**High density ratios.** Ratio ~20 runs; beyond that expect to lower the forcing
considerably. This is *not* a hard model limit — the earlier "ceiling at ~20" was an
artifact of the `single_interface` periodic-wrap bug, now fixed by `slab_interface`, and
ratio 100 has been run stably in 2D at low enough velocity. The remaining constraint
appears to be a velocity/Ca ceiling, not the density ratio itself, but this was never
diagnosed properly — if ratio ≥ 32 matters, dump fields just before divergence and find
where the instability nucleates rather than sweeping parameters.

**R16 of the `~/runs/dispersion` campaign** diverged at step 157832 of 218000 (max|u|
only 0.0166, so not obviously velocity-driven). Truncated to the last clean dump; mean
displacement 327 lu of the 512 lu target, so it is usable but short. The 69 corrupt dumps
are preserved in `~/runs/dispersion/R16/corrupt_dumps/` if anyone wants to diagnose it.
The lower-Bo `c*` series in the same folder is generated but was never launched.

## 5. Hard-won gotchas

Every one of these cost real time. They are not obvious from the code.

- **Clear `out/` before relaunching a run.** Two runs with different
  `particles_file_skip` interleave rather than overwrite, and the analysis then silently
  mixes two physically different simulations. This produced a spurious "oscillating
  σ²(τ)" that got a detailed and completely wrong physical explanation before the cause
  was found.
- **Screen particle data on magnitude, not NaN.** A diverging run wrote finite ~1e24
  positions one dump *before* the field log showed NaN. `isfinite()` passes them. Use
  `max|pos| > 1e4`.
- **Trajectories are not reproducible across rebuilds with `correct_op_mass = true`.**
  The corrector's double `atomicAdd` summation order changes between compilations
  (~1e-12), and chaotic advection amplifies it. For bitwise A/B, use one binary and/or
  set `correct_op_mass = false`. Bulk statistics are unaffected.
- **Checkpoint continuation is bit-exact in double, not in single.** Fields match to the
  float floor either way, but in a single build tracer trajectories diverge over a few
  hundred steps for the same chaotic-amplification reason. Use double if exact
  trajectory continuation matters.
- **`stream_inplace` requires `fuse_collision` + `stream_matrix_free`** (enforced). It
  host-verifies its op program at init and falls back to ping-pong with a warning rather
  than running a wrong program — check for `stream_inplace program verified` in the log.
- **`rho0 == rho1` makes `c = (ρ−ρ1)/Δρ` degenerate.** The stock `multi_phase.cfg` ships
  with both equal. Density-matched is a legitimate case, but set the densities explicitly
  if you want contrast.
- **`fluid_initializer` names are matched as substrings and fall back to `uniform` with a
  warning if unrecognised.** A typo (`slap_interface`) silently gives no interface at
  all. Check the `fluid_initializer = ... -> ...` line in the log.
- **`output_deflate` defaults to 0** because ParaView's Xdmf3 reader can crash on gzipped
  datasets read through XDMF. Compress for archival, `h5repack -f NONE` before loading.
- **Don't test in a live run directory.** A test whose `output_dir` silently failed to be
  redirected wrote into a production `out/`, and the subsequent cleanup deleted 486 of
  497 particle dumps. Recovered only from a zip backup. Test under `/tmp`, and assert the
  config points where you think it does.

## 6. Workflows

```bash
# Build (double default; -DFELBM_SINGLE=ON for float)
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86 && cmake --build build -j

# Validate — do this after any kernel change
scripts/validate.sh build

# Run one of the shipped examples
mkdir run && cd run && cp ../cfg/2d_cylinders/*.cfg . && mkdir out
../build/felbm_gpu settings.cfg

# Profile (nsys works without ncu's elevated counter permissions)
nsys profile --stats=true -o report --force-overwrite=true ./felbm_gpu settings.cfg
```

Ready-to-run configs: `cfg/2d_cylinders/` (auto-generated geometry, needs nothing) and
`cfg/3d_image/` (TIFF stack). CPU equivalents in `felbm_local/bin/settings_2d_cylinders.cfg`
and `settings_3d_image.cfg`.

Key sources: `include/felbm_gpu/device_engine.cuh` (all kernels),
`multiphase_gpu.cuh` (`MultiPhaseGPU`: init, step sequence, reductions),
`src/felbm_gpu_main.cu` (driver, output, checkpointing),
`src/compare_cpu_gpu.cu` (validation harness),
`../felbm_local/LBM/include/lbm_particle_manager.h` (host tracker).

## 7. Dispersion analysis context

- `felbm_local/scripts/dispersion_by_dm.py` — displacement-covariance tensor per Dm group
  and per seed plane, by FFT in O(T log T) (full 3001-dump × 10k-tracer dataset in ~17 s;
  the old O(T²) loop was unusable). Reports the **central** second moment.
  `--dims 2` is **required** for `size_z = 1` runs, and for any run predating the 2D
  z-kick fix, where tracers wandered thousands of cells out of plane.
- **Two estimator subtleties that silently changed answers.** (i) The mean displacement
  must be subtracted: in the 3D porous runs the forcing is not along a principal axis of
  the permeability tensor, giving a mean *transverse* velocity whose (v·τ)² contribution
  reached 59% of the reported σ²⊥ and inflated the exponent from 1.33 to 1.56.
  (ii) `D = σ²/(2nτ)` averaged over late lags is biased by any non-zero intercept of
  σ²(τ); prefer a slope fit, as `runs/analyze_dispersion.R` does. Note that R script
  reports a *dispersivity* dσ²/d(uτ) = 2D/u, not a diffusivity — the two trend in
  opposite directions with Bo because ⟨u⟩ varies by 59× across that series.
- Pore scale ℓ (distance transform, medial axis): **13.3** lu at 300³, **8.4** lu at 150³.
  These are measured. The 2D cylinder values used in places (12.9 / 6.45 / 3.225) are
  *estimates* scaled from the cylinder radius and have never been distance-transformed.
- Bo definition is ambiguous in the older configs: `image.cfg` uses Δρ, SESSION notes use
  ρ̄. At ratio 1.25 they are nearly proportional; at ratio 100 they differ by 2×. Pin it
  down before comparing new runs to the Bo^1.37 result.
- Transport is **non-Fickian**: σ²⊥ ∝ τ^1.33 after correcting the mean-subtraction bug, so
  a straight-line fit of σ² vs τ is ill-posed. Report the exponent, or a local slope at
  matched dimensionless lag.

## 8. Suggested first move

State the goal. If it is **science**, the code is not the constraint — go straight to the
campaign and read §7 first, especially the two estimator subtleties. If it is **code**,
there is no urgent optimisation; the highest-value work is now correctness-adjacent
(§4: bounding the mass-correction non-locality, or diagnosing the high-density-ratio
instability properly). If it is **performance** anyway, §3.3 (AoSoA) is the only lever
consistent with the current profile — prototype one kernel and measure before committing.

Whatever the change: put it behind a flag, validate with `scripts/validate.sh`, benchmark
MLUPS A/B, and re-profile with nsys to confirm the bottleneck actually moved.
