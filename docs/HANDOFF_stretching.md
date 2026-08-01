# Handoff — Lagrangian stretching + 3D geometry pipeline

For a fresh session on the Linux/CUDA machine. Everything below was written and
syntax-checked on a host with **no GPU and no CUDA toolkit**, so the CUDA
translation unit has *not* been compiled and nothing has been benchmarked. That is
the work to do here.

Companion documents: `SESSION_HANDOFF.md` (state of the solver as a whole),
`felbm_local/docs/particle_tracking.md` (what the stretching feature is and why),
`scripts/geometry/README.md` (the packing pipeline).

---

## 1. Why this exists

The paper's 3D evidence is one transient drainage front, one geometry, one
capillary number. All the quantitative physics — the Ca–Bo relation, `s ~ Ca^1/4`,
and above all the **optimum in `lambda(Ca)`** — is 2D finite-element. The
Discussion says outright that a qualitatively similar `lambda(Ca)` is *expected*
in 3D. Turning that expectation into a measurement is the target, and it needs two
things the code did not have: the 3D granular geometry, and an affordable
stretching estimator.

Two independent commits:

| commit | what |
|---|---|
| `scripts/geometry/` | Yade RCP generator + periodic voxelizer → the existing `3d_image` path. No C++ change. Already tested (`voxelize_spheres.py --self-test`). |
| stretching patch | `rho` evolution in `ParticleManager` + Eq. (12), across `felbm_local` and `felbm_gpu`. **Not yet built with nvcc.** |

---

## 2. What the stretching patch does

Each tracer carries a material line element. Rather than integrating
`d(rho)/dt = (rho.grad)u` — `rho` overflows a double after ~`700/lambda` advective
times — the orientation is kept as a unit vector and the log-length accumulated
separately (Benettin renormalisation):

```
rho* = rho_hat + dt (J rho_hat) ;  g = |rho*| ;  log_rho += log g ;  rho_hat = rho*/g
```

`J` is the velocity-gradient tensor `probe()` already fills unconditionally, so
nothing existing is recomputed and the position update is arithmetically
untouched. Two estimators come out:

- `lambda = < rho_hat^T J rho_hat >` — **Eq. (12)**. Instantaneous, so it converges
  with ensemble size rather than integration time. This is what makes a Ca sweep
  affordable: a few advective times per point instead of tens.
- `<log rho>` and `Var(log rho)` — the latter is the `sigma^2_{log rho}` in
  `c_max ~ exp(-(lambda + sigma^2/2) t/t_a)`, which the paper's Discussion uses
  and currently has no measured value for.

### Files touched

| file | change |
|---|---|
| `felbm_local/LBM/include/lbm_particle_manager.h` | `m_rho_hat`/`m_log_rho`, `seed_orientations()`, the update-loop block, the reduction, HDF5 read/write, CSV columns, `nodes()`/`rho_hat()`/`log_rho()` accessors, `take_stretch_rows()` |
| `felbm_local/LBM/include/lbm_settings.h` | four optional keys |
| `felbm_local/LBM/include/lbm_scheduler.h` | `write_stretching()` for CPU runs |
| `felbm_local/tests/test_particle_stretching.cpp` | new analytic test (+ registered in `run_tests.sh`) |
| `felbm_gpu/src/felbm_gpu_main.cu` | open `stretching.txt`, `flush_stretch()`, drained at `sskip` behind the existing `p_join()`, plus a final drain |

### Config

```
particles_enable          = true
particles_stretching      = true
particles_stretching_file = stretching.txt
particles_stretching_skip = 0          # 0 -> follow log_skip
particles_stretching_seed = 20260801
```

Output columns: `step  lambda  mean_log_rho  var_log_rho  n_active`.

---

## 3. Do this first, in order

### 3.1 Build both trees

`felbm_local` is header-only for the parts that changed, so **both must be
rebuilt**. No CMake or dependency changes.

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86 && cmake --build build -j
```

The CUDA TU has never been through nvcc. The edits in `felbm_gpu_main.cu` are
plain host C++ in the same style as the surrounding code, but expect to fix
something trivial. Points to check if it complains: `pm.stretching()`,
`pm.take_stretch_rows()`, and the `StretchRow` member names (`step`, `lambda`,
`mean_log`, `var_log`, `n_active`).

### 3.2 Confirm nothing regressed

```bash
scripts/validate.sh build
```

This must still pass at the usual floor (~1e-15 double, ~1e-6 single). It compares
`h`/`g` and does not involve particles at all, so a failure here means something
unrelated broke — treat it as a red flag, not as noise.

Then the stronger claim, which is the one worth actually testing: **a run with
`particles_stretching = false` should be bit-identical to before the patch.** The
feature is guarded inside the loop body and does not touch the RNG draw order, so
this should hold exactly. Check it in **double** precision with
`correct_op_mass = false` (see the reproducibility gotcha in `SESSION_HANDOFF.md`),
by diffing `particles_*.h5` positions against a run from the previous commit.

### 3.3 Run the analytic test

```bash
felbm_local/tests/run_tests.sh test_particle_stretching        # substeps = 1
felbm_local/tests/run_tests.sh test_particle_stretching 4
```

It does not run the LBM — it fills the velocity arrays analytically — so it
already passes on a CPU-only host. Reference output there:

```
uniform shear     worst |err(log rho)| = 4.0e-15      (exact case)
Poiseuille (a)    mean |err(log rho)|  = 4.7e-04      (interpolation-limited)
Poiseuille (b)    <rho^2>-1 = 0.02291 vs 0.02344      (3.3% sampling error)
rotation          max |lambda| = 2.2e-19              (identically zero)
rotation          spurious <log rho> = 1.0e-4 vs 1.0e-4 analytic
```

Two of these are worth understanding rather than just watching go green:

- The Poiseuille per-particle error **does not improve with `substeps`** (identical
  at 1, 2 and 4). That is the signature of interpolation error on a quadratic
  profile, not time-integration error, and it is the expected behaviour.
- The rotation case shows explicit Euler inflating `|rho|` by
  `sqrt(1+(Omega dt)^2)` per substep while Eq. (12) stays at machine zero. The
  bias halves when substeps double. **This is the concrete argument for measuring
  `lambda` by Eq. (12) rather than by differentiating `<log rho>`** — worth keeping
  in mind when interpreting production output.

### 3.4 Benchmark — the open question

The claim made when scoping this was that the stretching measurement does **not**
need a GPU port, because the ~50 ms/step host cost at 300³ is the velocity
**scatter** (O(fluid sites), fixed) and not the tracer advection (O(N_p), ~1 ms at
12k). If that is right, going to 10^5 `rho`-carriers is close to free.

Measure it rather than assume it:

```bash
# 300^3 percolating image, all six fast-path flags on, minimal logging, no dumps
#   a) particles_enable = false                       -> baseline MLUPS
#   b) 12k  tracers, particles_stretching = false     -> current cost
#   c) 12k  tracers, particles_stretching = true      -> marginal cost of rho
#   d) 200k tracers, particles_stretching = true      -> does N_p matter yet?
```

Reference points from `SESSION_HANDOFF.md`: 198 MLUPS with no tracers, ~107 with
12k tracers and `particles_overlap`. Prediction: (c) ≈ (b), and (d) still close.
**If (d) drops materially, the host has become the critical path and GPU-side
particle advection moves to the top of the optimisation list** — which is item 1
in `SESSION_HANDOFF.md §3` anyway. Record the numbers in the README table either
way; a negative result is just as useful.

Also worth timing separately: `pt_update` vs `pt_scatter` in the tracking-phase
totals the driver already prints at the end of a run. Those two numbers answer the
question directly.

### 3.5 First physics check

Before any Ca sweep, reproduce a known number. Steady single-phase flow through a
random bead pack has a measured Lyapunov exponent `lambda ~ 0.21` (Heyman et al.,
ref. 10 of the paper). Run single-phase in an RCP pack from
`scripts/geometry/`, and see whether the Eq. (12) estimator lands there. If it
does, the tracking is validated against the literature and the multiphase numbers
are defensible. If it does not, fix that before anything else.

Note `lambda` in `stretching.txt` is per **lattice step**; multiply by
`t_a = d/U` in lattice units to compare with published `lambda t_a`.

---

## 4. Known limitations, deliberately

- **Incompatible with `particles_refine`**, refused with a warning. Refinement
  changes the node set and would desync the per-particle arrays, the same reason
  `particles_dm_groups` refuses it. Sheets measure area and grow their node count
  like `e^{2 lambda t/t_a}`, so they are a few-advective-time figure-making tool;
  the `rho` ensemble costs the same at `t = 100 t_a` as at `t = 0`.
- **Restart continuity is exact only in double.** `rho_hat`/`log_rho` are
  checkpointed and restored, but in a single build tracer trajectories already
  diverge over a few hundred steps from the float floor, so the stretching history
  inherits that.
- **No phase split.** `lambda_w` vs `lambda_nw` needs the concentration sampled at
  tracer positions, i.e. a 4th global scattered array — roughly +17 ms on the
  ~50 ms host scatter. Left out on purpose; it is the one part of this that is not
  free. If it is wanted, `gpu.d_c` exists and the plumbing mirrors `d_ux`.
- **Isotropic initial orientations** need a few advective times to align with the
  Lyapunov direction. Discard that transient before averaging; `n_active` in the
  timeseries will tell you if wall-blocking is also biasing the ensemble.

---

## 5. Geometry pipeline — what to run once there is a GPU

`scripts/geometry/README.md` has the detail. The one number that decides the
campaign:

| d (lu) | fluid sites | GPU mem | pore volume narrower than 3W |
|---|---|---|---|
| 12 | 0.67M | 1.1 GB | 98.5% |
| 16 | 1.6M | 2.5 GB | 91.1% |
| 20 | 3.1M | 4.9 GB | 77.9% |
| 24 | 5.3M | 8.4 GB | 62.8% |

(measured on a test pack at porosity **0.50**; RCP at 0.36 will be tighter.)

Resolving the diffuse interface pushes `d` up; memory pushes it down. Where those
cross sets the honest low-Ca limit for the sweep. Run the ladder on a real Yade
packing before committing to a production box, and settle the grain-shrink
question with a single-phase permeability check against the FEM value.

Worth chasing separately: ref. 82 of the paper deposits *"Meshes and analysis
scripts"*. If the original Yade sphere list is in that dataset, voxelizing **it**
makes the LBM-vs-FEM comparison a same-geometry comparison instead of a
statistically-equivalent one.
