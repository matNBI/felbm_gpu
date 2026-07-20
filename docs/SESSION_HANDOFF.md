# felbm_gpu — session handoff / continuation brief

Portable context for continuing this work in a fresh Claude (or Claude Code) session,
ideally on the machine with the dedicated GPU where the code can actually be built,
validated, and profiled. The repo itself is the source of truth; this doc captures the
*decisions and plan* that aren't obvious from the code alone.

Repo on the GPU box: `/lscr/nbicmplx/mathies/felbm_gpu` (this checkout mirrors it).

---

## 0. Update (RTX 3090 box, this session)

Both priority items from section 3 are DONE, committed on `main`, validated:

- **Tracking overhead (3.1-3.3)** -> `1fc2b41`: pinned D2H + fused parallel
  scatter (kills the `ucx=unx` copy) + overlap worker (`particles_overlap`,
  default true). ~494 -> ~1 ms/step at 300^3; trajectories bit-identical
  (h5diff, overlap on vs off).
- **MRT fast transform (3.4)** -> `0a65aca`: `mrt_fast_transform = true` runs
  the moment relaxation in real_t, identical summation order. Bit-identical in
  double builds; float drift unchanged (~5e-7). No sparse walk needed: dense
  FP32 is throughput-free on Ampere, and fixed indices keep arrays in registers.

Numbers (RTX 3090, 300^3 porous = 9,949,099 sites, single, MRT fused, 100 steps):
original tracking + dense MRT 16.7 MLUPS -> new tracking 96.3 -> + mrt_fast
**98.5 MLUPS with tracking / 126.0 without** (OMP_NUM_THREADS=24; the worker's
per-step OMP team spin-up is measurable at the 128-thread default).
Follow-ups also landed: persistent overlap worker (`072b9d3`, thread-per-step
team spin-up removed -> **107.2 MLUPS with tracking**; `particles_threads` cfg
key exists but the full-width default now wins) and the exact `d_relax` Vn->n
shrink (`963bacd`, ~0.7 GB single / ~1.4 GB double at 300^3). Post-mrt_fast
nsys: k_collide_fused_mrt 60% -> 38.9% (20.7 ms/call), remaining kernels flat
and bandwidth-bound (~53 ms/step GPU total). Remaining tracking lever if ever
needed: GPU-side advection (3.3).

In-place streaming also landed (`e97a75f`, `stream_inplace` cfg key): drops
d_h2/d_g2 AND d_src via reversed-slot collision writes + a host-verified
disjoint swap/avg/zero program; bit-equivalent to ping-pong, ~1.15 GiB net at
300^3 single (~2.9 GiB double), speed neutral (128-130 MLUPS no-tracking,
106.6 with tracking). NOTE: commit `f026430` ("init cylinder optimization")
was a checkpoint that swept the in-place sources plus build/ dirs into git
(now untracked + .gitignored; history not rewritten since it was pushed).

VALIDATION GOTCHA for trajectory A/Bs: with `correct_op_mass = true` particle
trajectories are only reproducible for the *same binary* -- the corrector's
double-atomicAdd reduction order changes across recompiles (~1e-12), which
chaos amplifies over ~100 steps. Bitwise particle comparisons must use one
binary and/or `correct_op_mass = false`.
Bench configs live in `~/code/felbm_local/bin/settings_gpu_bench*.cfg`
(mp_gpu_bench.cfg fixes rho0==rho1 in the stock multi_phase.cfg, which makes
c=(rho-rho0)/drho blow up).

## 1. Where the code stands

felbm_gpu is a validated single-GPU port of the multiphase (Lee–Liu free-energy) D3Q19
LBM solver. It reuses the CPU host code in `felbm_local` unmodified. Correctness is
established by an exact CPU-vs-GPU harness (`compare_cpu_gpu`), matching to machine
precision (~1e-16 in double) for BGK + MRT, all-fluid + porous, on every operator path.

Optimization work completed (all behind cfg flags, default off = original CSR path;
all validated exact against the CPU):

- **Matrix-free operators** (`stream_matrix_free`, `grad_matrix_free`): the 7 stencil
  operators (streaming, grad_cd, grad_bd, lap, cd_dir, bd_dir, avg_dir) replaced by
  compact index tables + on-the-fly weight computation. Removes stored CSR + dir-buffer
  memsets. `k_*_mf` kernels in `device_engine.cuh`.
- **Fusion** (`fused`, `fuse_collision`): `fused` recomputes the directional derivatives
  inside the equilibria/force kernels; `fuse_collision` (implies `fused`) does
  equilibria + force + collision + apply in ONE per-site kernel
  (`k_collide_fused_bgk` / `k_collide_fused_mrt`), so the per-direction temporaries are
  never materialised. MRT holds the 19 `coll_g` in a register array for the moment
  transform.
- **Host-memory fix**: the CSR `FieldOperatorGPU` is only constructed in the `!mf_grad`
  branch (it was ~100 GB of host sparse matrices at 300³, built then wasted); the driver
  calls `malloc_trim(0)` after `init()`. Host RES dropped ~130 GB → a few GB at 300³.

Net vs the all-CSR baseline: ~**2.34× faster**, per-site memory ~5.6 → ~1.7 KB (double).
Single precision (`-DFELBM_SINGLE`) validated (~1e-7 drift, pure float rounding); needed
to fit 300³ on the 24 GB A5000.

Git: this work is on a **`fusion` branch**. Merging to `main` is safe (all flags default
off). See `docs/PORTING_SPEC.md` for the full measurement tables and roadmap.

---

## 2. Latest profiling (nsys, 300³ porous, single, MRT, WITH particle tracking)

Command used (works without elevated GPU-counter perms, unlike `ncu`):

```bash
nsys profile --stats=true -o report_300 --force-overwrite=true \
     ./felbm_gpu settings.cfg > prof_300.txt 2>&1
```

Run: 9,949,099 fluid sites, 1001 steps, 49.3 MLUPS, wallclock 201.8 s.

GPU kernel time ≈ **94 s total**. Breakdown (per-site, so ~identical at any N):

| kernel | % of GPU | note |
|---|---|---|
| `k_collide_fused_mrt` | **60%** (56.5 ms/call) | the fused collision — the MRT moment transform |
| `k_stream_gather` | 7.5% | streaming |
| `k_grad_bd_mf` | 7.4% | |
| `k_lap_mf` | 7.4% | |
| `k_grad_cd_mf` | 5.0% | |
| `k_moments` | 4.2% | |
| `k_inject_mass` + `k_mass_weight` | 6.1% | order-parameter mass corrector |
| grad_md / vel_press / packs / mu_axpy | ~2% | |

**Two bottlenecks, and which matters depends on the run:**

1. **`k_collide_fused_mrt` = 60% of GPU compute.** The hotspot inside it is the MRT
   moment transform: two dense 19×19 matrix–vector products (M and M⁻¹) done **in
   double precision, per site**, plus three 19-element local arrays (`cg`, `fgs`, `go`)
   that likely spill (register pressure). On the A5000, FP64 is ~1/32 of FP32, so this
   is disproportionately expensive in an otherwise-float32 run. This dominates
   non-tracking runs and BGK is unaffected (no transform).

2. **Particle tracking ≈ HALF the wallclock at 300³** (was ~10% at 150³ — it scales
   with domain size). The host-side tracker (option-b, `lbm_particle_manager.h` in
   `felbm_local`) copies the full 3-component velocity field (~119 MB) Device→Host
   **every step** — 122 GB total over the run — then integrates 10k tracers on the CPU.
   Neither overlaps with GPU compute. `cudaMemcpy` = 101 s of blocking host time; the
   remaining ~100 s is CPU particle integration. GPU physics is only ~47% of wallclock.

---

## 3. Optimization plan (prioritized)

**For production runs WITH particle tracking (dispersion campaigns), do the tracking
overhead first — it's the bigger wallclock lever:**

1. *Cheaper integrator / fewer substeps* (quick, cfg/param-level test first). The 2nd-order
   scheme builds the velocity-gradient tensor J per tracer per substep; dropping to
   1st-order Euler or fewer substeps cuts the ~100 s host cost. Verify D⊥ is unchanged.
2. *Overlap the velocity D2H + host integration with GPU compute* — `cudaMemcpyAsync` on
   a separate stream + double-buffer, pipeline step N+1's kernels behind step N's host
   work. Hides most of the ~100 s. Medium effort.
3. *Move particle advection on-GPU* — eliminates the D2H entirely and parallelises the
   integration. Biggest win, biggest change.

**For all runs (and non-tracking / BGK), the GPU compute lever:**

4. *Sparse/float MRT moment transform in `k_collide_fused_mrt`.* The D3Q19 M / M⁻¹ are
   mostly zeros with small integer entries — replace the ~720 dense double FMAs/site
   with a hand-coded float transform. Likely 2–3× on the collision kernel. **Must be
   validated** against `compare_cpu_gpu` (pure-float MRT may introduce small error);
   implement behind a flag so it's A/B-able. Also build with
   `nvcc --ptxas-options=-v` to check register/spill on this kernel; if it spills badly,
   consider splitting just the g-transform back into its own kernel.

**Minor / later:** shrink `d_relax` from `Vn` to `n` (uniform per site); in-place
streaming to drop the `d_h2/d_g2` ping-pong (→ ~1 KB/site, 300³ in double); reduce mass
corrector frequency if drift allows.

---

## 4. Workflows (run these on the GPU box)

```bash
# Build
cd felbm_gpu/build && cmake --build . -j          # add -DFELBM_SINGLE for float

# Validate (exact CPU vs GPU). Args:
#   [steps] [N] [ratio] [geom=fluid|spheres] [coll=bgk|mrt] [mf] [mfg] [fused] [fusecoll]
./compare_cpu_gpu 20 48 5 spheres mrt 0 0 0 1      # expect max|Δ| ~1e-16
./compare_cpu_gpu 20 48 5 fluid   bgk 0 0 0 1

# Benchmark (prints MLUPS + active mode). Flip cfg keys and A/B:
#   stream_matrix_free, grad_matrix_free, fused, fuse_collision  (all default false)
./felbm_gpu settings.cfg

# Profile (short max_iterations in cfg; nsys works without ncu's counter perms)
nsys profile --stats=true -o report --force-overwrite=true ./felbm_gpu settings.cfg
```

Key source files:
- `include/felbm_gpu/device_engine.cuh` — all CUDA kernels (namespace `felbm_gpu`,
  guarded by `__CUDACC__`). The fused kernels + `k_*_mf` matrix-free kernels are here.
- `include/felbm_gpu/multiphase_gpu.cuh` — `MultiPhaseGPU`: members, `init()` (builds mf
  tables, conditional allocs), `step()` (kernel launch sequence with the flag branches),
  `free()`.
- `src/felbm_gpu_main.cu` — driver: reads cfg keys, GPU select, MLUPS timing, HDF5 out.
- `src/compare_cpu_gpu.cu` — the validation harness.
- `../felbm_local/LBM/include/lbm_particle_manager.h` — host-side tracker (the D2H +
  integration cost); `update()` is the advection loop, `write_hdf5()` the output.

Config gotcha that bit us: domain size is set by `coarsening_levels` in
`domain_percolating_600.cfg` (0 → full, 1 → half). Confirm `New values: 300 300 300` and
`n_sites=9949099` in the run header to be sure you're at full 300³.

---

## 5. Dispersion analysis context (if continuing the science, not the code)

- `scripts/position_variance.py` computes the Mathiesen et al. (GRL 2023) displacement-
  covariance tensor σ²(Δτ) = ⟨[x(t+Δτ)−x(t)][…]ᵀ⟩_{k,t} (MTO estimator), matched by
  tracer `id`, positions are stored unwrapped so periodic wrap needs no handling.
  Args include `--flow-axis`, `--min-iter` (trim transient), `--stride`, `--intersection`.
- Characteristic pore diameter ℓ (distance-transform, medial-axis): **13.3** lu at 300³,
  **8.4** lu at 150³. Bo = ℓ²·ρ̄·g/γ with ρ̄=1.125, γ=0.0083.
- Result: transverse **D⊥ ∝ Bo^1.37** over the reliable range Bo ≈ 0.07–7.2. Below that
  the flow **capillary-arrests** (⟨v²⟩ decays to ~0); a7 (Bo 0.024) froze, half-size runs
  are pre-asymptotic. Longitudinal is non-Fickian.
- Convergence caveat: the running D⊥ (local slope) is still rising in ALL runs (a5 is the
  least converged at ~19 advection times), so absolute D⊥ are ~×1.5 lower bounds, but the
  *exponent* is robust to the fit window (1.33 early-lag vs 1.37 late-lag).
- Outputs live in `data/variance_results/` (`dispersion_vs_bond_all.csv`,
  `Dperp_vs_bond_all.png`, per-run curves).
- Run-length rule for new weak-forcing runs: dump interval ≈ τ_a/12 where
  τ_a = ℓ/⟨v⟩, and run ~18–20 τ_a to reach Fickian.

---

## 6. Suggested first move in the new session

State the goal (GPU optimization vs dispersion science). For GPU work, the highest-value
first step for tracking runs is the particle-overhead path (§3.1–3.3); for a well-scoped
compute win that helps every run, the sparse/float MRT transform (§3.4). Either way:
change behind a flag, validate with `compare_cpu_gpu` (~1e-16), benchmark MLUPS A/B, and
re-profile with nsys to confirm the bottleneck moved.
