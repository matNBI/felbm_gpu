# Handoff — Lagrangian stretching + 3D geometry pipeline

Originally written and syntax-checked on a host with no GPU and no CUDA toolkit.
**Worked through on the Linux/CUDA machine on 2026-08-01** (RTX 3090 24 GB, CUDA
13.3, Ubuntu 26.04, Yade 2026.1.0); the sections below now record what was
verified rather than what was planned. Status in one table:

| item | state |
|---|---|
| §3.1 build both trees | **done** — nvcc 13.3, `-DCMAKE_CUDA_ARCHITECTURES=86`, no fixes needed |
| §3.2 no regression / inert when off | **done** — bit-identical in double |
| §3.3 analytic test | **done** — reproduces the reference numbers |
| §3.4 benchmark | **open** — needs a 300³ image |
| §3.5 `lambda ~ 0.21` against Heyman | **open** |
| §5 geometry pipeline | **runs**, and the resolution/memory ladder is measured — see §5, the answer constrains the campaign |
| single-phase permeability vs FEM | **open** |

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
| `scripts/geometry/` | Yade RCP generator + periodic voxelizer → the existing `3d_image` path. No C++ change. felbm_gpu `4850f4e`. |
| stretching patch | `rho` evolution in `ParticleManager` + Eq. (12), across `felbm_local` and `felbm_gpu`. felbm_local `38e420d`, felbm_gpu driver half `e7281a4`. |

The two arrived on this machine separately from the code they describe. The
felbm_gpu driver half of the stretching patch and the whole of
`scripts/geometry/` had to be written or repaired here; `voxelize_spheres.py`
was used as received, but `yade_rcp.py` had never successfully run anywhere and
needed four fixes (see §6).

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

## 3. The checklist, and what came of it

### 3.1 Build both trees — done

`felbm_local` is header-only for the parts that changed, so **both must be
rebuilt**. No CMake or dependency changes.

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86 && cmake --build build -j
```

Clean under nvcc 13.3, warnings only, and all of them pre-existing. Note this
does exercise the changed header: `felbm_gpu_main.cu` includes
`lbm_particle_manager.h`, so the new `ParticleManager` code is in the CUDA TU.
The three suspects flagged when this was written — `pm.stretching()`,
`pm.take_stretch_rows()`, and the `StretchRow` member names — all compiled as
written. (`StretchRow` is declared in the manager's private section, but `auto`
deduction never names it, which is how the CPU scheduler gets away with it too.)

### 3.2 Confirm nothing regressed — done

```bash
scripts/validate.sh build
```

Passes at 1.4e-7 … 2.9e-7 across all twelve cases, i.e. the single floor (that
build is `FELBM_SINGLE=ON`), and unchanged by the patch.

The stronger claim also holds. A 256×512 cylinder pack, 6000 tracers, 3000 steps,
**double** with `correct_op_mass = false` and `output_float32 = false`, stretching
off vs on: `position`, `velocity`, `Dm` and `id` **bit-identical at float64**.
`|rho_hat|` stays a unit vector to 2.2e-16, and no `stretching.txt` is created at
all when the feature is off. The `.h5` gains `log_rho`/`rho_hat` only when it is on.

Restart continuity works through the GPU driver: restoring from a checkpoint at
step 1500 restores `rho_hat`/`log_rho` and reproduces the uninterrupted run's next
row to every digit.

### 3.3 Run the analytic test — done

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

Reproduced here at both substeps 1 and 4: shear worst 4.19e-15, Poiseuille 4.677e-04
(**identical at 1 and 4**, the interpolation-limited signature predicted below),
`<rho^2>-1` = 0.02248 vs 0.02344, rotation `max|lambda|` 7.0e-20 / 5.1e-20, and the
spurious `<log rho>` 1.0e-4 → 2.5e-5 for two doublings of `substeps`, matching its
closed form to 1e-10.

The GPU path agrees with itself too: over the last rows of a production-style run
`mean_log_rho` climbs 5.5e-5 per step against a reported `lambda` of 5.52e-5 —
two independent code paths, same number.

Two of these are worth understanding rather than just watching go green:

- The Poiseuille per-particle error **does not improve with `substeps`** (identical
  at 1, 2 and 4). That is the signature of interpolation error on a quadratic
  profile, not time-integration error, and it is the expected behaviour.
- The rotation case shows explicit Euler inflating `|rho|` by
  `sqrt(1+(Omega dt)^2)` per substep while Eq. (12) stays at machine zero. The
  bias halves when substeps double. **This is the concrete argument for measuring
  `lambda` by Eq. (12) rather than by differentiating `<log rho>`** — worth keeping
  in mind when interpreting production output.

### 3.4 Benchmark — still the open question

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

**Not done — it needs a 300³ percolating image, which §5 has not yet produced.**
The only datum so far is an indication from the 256×512 smoke test, one run each
and with the tracer work overlapped anyway: `pm_update` 0.92 s → 0.99 s over 3000
steps at 6000 tracers, about +7% for carrying `rho`. Consistent with (c) ≈ (b),
but it says nothing about (d), which is the row that actually matters.

### 3.5 First physics check — not done

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
- **Restarts are not reproducible whenever `particles_dm_groups` is nonzero**, and
  this is *not* the float issue above — it shows up in double. The Brownian RNG
  state is not checkpointed, so the `Dm > 0` groups resume on a fresh stream and
  the trajectories separate immediately. Measured over 1500 steps after a restart,
  `<log rho>` reached 0.1167 against 0.1123 for the uninterrupted run. Chaotic
  divergence amplifies it, which is precisely what this feature measures. Restarts
  are fine for continuing a measurement, not for reproducing one. Pre-existing and
  unrelated to stretching.
- **The `step` column is `m_n_updates`, not the LBM step, and it resets on
  restart.** A run resumed at 1500 writes rows numbered 1…1500 while the LBM is at
  1501…3000, and `stretching.txt` is truncated on restart besides. Concatenating a
  restarted sweep without accounting for this silently gives wrong abscissae. The
  driver knows `restart_step`, so emitting `restart_step + r.step` is a one-line
  fix — left alone only because it would put felbm_gpu's format out of step with
  `Scheduler::write_stretching()`, which has the same property. Fix both or
  neither.
- **No phase split.** `lambda_w` vs `lambda_nw` needs the concentration sampled at
  tracer positions, i.e. a 4th global scattered array — roughly +17 ms on the
  ~50 ms host scatter. Left out on purpose; it is the one part of this that is not
  free. If it is wanted, `gpu.d_c` exists and the plumbing mirrors `d_ux`.
- **Isotropic initial orientations** need a few advective times to align with the
  Lyapunov direction. Discard that transient before averaging; `n_active` in the
  timeseries will tell you if wall-blocking is also biasing the ensemble.

---

## 5. Geometry pipeline — the ladder, now measured

`scripts/geometry/README.md` has the detail. The pipeline runs: `--self-test`
passes, and voxelising a real pack gives a porosity discretisation error of
3.7e-5 at `d = 20`, pore space percolating on all three axes, 0.003% dead volume,
and a `domain.cfg` whose key set matches `DomainInitializer_3DImage` exactly.

The ladder was run on a **real Yade RCP pack at porosity 0.385**, replacing the
old estimate from a test pack at porosity 0.50. The prediction that RCP would be
tighter was right, and by a lot:

| d (lu) | fluid sites | GPU mem | <1W | <2W | **<3W** |
|---|---|---|---|---|---|
| 12 | 64k | 0.10 GB | 0.664 | 1.000 | **1.000** |
| 16 | 151k | 0.24 GB | 0.411 | 0.914 | **1.000** |
| 20 | 295k | 0.47 GB | 0.277 | 0.784 | **1.000** |
| 24 | 511k | 0.81 GB | 0.198 | 0.603 | **0.929** |
| 32 | 1.21M | 1.92 GB | 0.109 | 0.358 | **0.773** |
| 40 | 2.36M | 3.74 GB | 0.072 | 0.238 | **0.543** |
| 48 | 4.09M | 6.47 GB | 0.052 | 0.174 | **0.389** |

(`W = interface_width = 5` lu; site counts are for the 4×4×6 d test box.)

The old table put 77.9% at `d = 20`. At true RCP that number needs **`d ≈ 32`** —
1.6× the resolution, 4× the memory.

### Where it crosses, and the box that does not fit

At porosity 0.385 and 1700 B/site, against ~22 GB usable of the 3090's 24:

| d | MB per d³ | max box | as a cube |
|---|---|---|---|
| 20 | 5.2 | 4200 d³ | 16.1 d |
| 24 | 9.0 | 2430 d³ | 13.4 d |
| 32 | 21.4 | 1030 d³ | 10.1 d |
| 40 | 41.9 | 530 d³ | 8.1 d |

**The 20×20×30 box `scripts/geometry/README.md` suggests needs 63 GB at `d = 20`
and 257 GB at `d = 32`. It does not fit on this card at any resolution that
resolves the interface.** What fits is a 10–13 d cube — against the 60d × 90d the
paper needed in 2D for Ca ≳ 1e-3. That gap is the honest low-Ca limit this section
set out to find, and it is the single most important result on this page.

### The two levers

**Narrowing the interface is the stronger one.** At `interface_width = 3`:

| d | <3W (W=5) | <3W (W=3) |
|---|---|---|
| 20 | 1.000 | 0.742 |
| 24 | 0.929 | 0.550 |
| 32 | 0.773 | 0.331 |

W=3 at d=24 (a 13 d box) beats W=5 at d=32 (a 10 d box) on constrictions *and*
memory at once. This is the first thing to try.

**Shrinking the grains**, the paper's 2D analogue, at d=32 and W=5:

| shrink | porosity | fluid sites | <3W |
|---|---|---|---|
| 0.00 | 0.385 | 1.21M | 0.773 |
| 0.05 | 0.473 | 1.49M | 0.655 |
| 0.10 | 0.552 | 1.73M | 0.524 |

It costs sites and moves the geometry off RCP, so it needs the single-phase
permeability check against the FEM value before it is defensible.

> **Caveat.** These came from a 4×4×6 d pack of 114 spheres, generated to test the
> plumbing. The constriction distribution is intensive so it converges faster than
> `phi_J` does, but do not commit a production box to these numbers — re-run the
> ladder on an 8×8×12 pack, which is `README.md`'s own "suggested first use" and is
> now just compute.

Worth chasing separately: ref. 82 of the paper deposits *"Meshes and analysis
scripts"*. If the original Yade sphere list is in that dataset, voxelizing **it**
makes the LBM-vs-FEM comparison a same-geometry comparison instead of a
statistically-equivalent one. Given how tight §5 turned out, a sphere list that
fixes the geometry is worth more than it was when this was first written.

---

## 6. Environment, and the state of `yade_rcp.py`

Set up on this machine: `sudo apt install yade python3-scipy python3-tifffile`.
Yade is 2026.1.0 from Ubuntu 26.04 universe. The system Python 3.14 is
`EXTERNALLY-MANAGED` and `python3.14-venv` is not installed, so the apt packages
are the path of least resistance — and they keep scipy/tifffile on the same
interpreter Yade embeds.

`voxelize_spheres.py` needed nothing. `yade_rcp.py` had never successfully run,
and needed four fixes (felbm_gpu `4850f4e`), two of which are algorithmic rather
than environmental and would have bitten on any machine:

| | |
|---|---|
| `minieigen` | folded into `yade.minieigenHP`; packaged builds no longer ship the old module |
| arg parsing | Yade 2026.1.0 consumes the `--` separator, and the parser's "no `--` ⇒ no args" fallback then **silently ignored every flag** and produced a default pack |
| `nan` deadlock | `unbalancedForce()` is `0/0` with no contacts, and `nan` fails every comparison, welding the relax state machine shut until `max_steps` |
| livelock | the freeze fires on a *collisional* pressure spike ~200 steps after each restart; 119 freeze/thaw cycles advanced `phi` by 1e-4 each and never jammed |

The livelock fix adds `--phi-step` (default 2e-3): a jamming test costs a full
relaxation, so require `phi` to have advanced before doing another. The price is
that `phi_J` is resolved only to `phi_step` and the pack ends up compressed past
`p_jam` — in the test run the relaxed pressure was 2.6e4 against `p_jam = 1e3`,
i.e. residual overlap `delta/R ~ 4e-3` rather than the ~5e-4 the `--p-jam`
docstring advertises. Lower `--phi-step` tightens both, at more compute. **If the
permeability check comes out off, this overlap is the first thing to suspect.**

Sanity check that the DEM is doing what it claims: the 4×4×6 pack jams at
`phi = 0.615` with coordination number **Z = 6.3**, against the isostatic value of
exactly 6 for frictionless spheres.

Two things to expect when scaling up. The outer loop on `N` cannot reach its 0.1%
box tolerance on small packings — one sphere is ~0.3% of the volume at N = 114 —
so it warns and returns its best, which is correct behaviour, not a failure. And
each compression here took ~2e6 DEM steps for 117 spheres; a 20×20×30 box is
N ≈ 14,700 and the outer loop runs up to 12 of them, so **time the first one before
queueing a campaign.**
