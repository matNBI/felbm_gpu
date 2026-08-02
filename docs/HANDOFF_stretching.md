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
| §3.4 benchmark | **mostly answered** — 50k stretching tracers are free at 9.17M sites |
| §3.5 `lambda ~ 0.21` against Heyman | **converged via extrapolation**: `lambda_inf` ~= 0.165 ± 0.01 vs 0.21. See §3.5b for the 12–15 t_a protocol |
| §5 geometry pipeline | **runs**, ladder measured on a real RCP pack and reconciled with the paper's Table S1 — target `d = 40`, `interface_width = 5.66` (19.35 GiB) |
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
| `scripts/geometry/` | Yade RCP generator + periodic voxelizer → the existing `3d_image` path. No C++ change. felbm_gpu `34606c0` + `60c3288`. |
| stretching patch | `rho` evolution in `ParticleManager` + Eq. (12), across `felbm_local` and `felbm_gpu`. felbm_local `38e420d`, felbm_gpu driver half `e521632`. |

The two arrived on this machine as *files*, ahead of the commits that carried
them, so the felbm_gpu driver half was independently re-implemented here before
`e521632` was fetched. The two versions were equivalent; `e521632` is kept as the
canonical one. `voxelize_spheres.py` was used as received (bar the
`--bytes-per-site` correction), but `yade_rcp.py` had never successfully run
anywhere and needed five fixes (see §6).

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

**Partly answered by the §3.5 run**, which is a 9.17M-site image — the scale row
(d) was written for:

| | MLUPS |
|---|---|
| (a) no tracers | 217.5 |
| (d') 50k tracers + stretching | **224.0** |

**50,000 stretching tracers cost nothing measurable** at 9.17M sites: `pm_update`
was 8 ms/step against a 41 ms GPU step, entirely hidden by the overlap. The
scoping claim holds, and GPU-side particle advection does *not* move up the
optimisation list. Rows (b) and (c) were not run separately, and 200k was not
reached, but the trend at 50k is unambiguous.

Caution when reading the driver's own breakdown: the `d2h=` figure is nonsense as
a copy cost — the D2H synchronises the pipeline, so that timer absorbs GPU wait
(2240 s of a 4093 s run). Worth fixing the instrumentation before anyone sizes a
decision on it.

### 3.5 First physics check — run, and INCONCLUSIVE

Before any Ca sweep, reproduce a known number. Steady single-phase flow through a
random bead pack has a measured Lyapunov exponent `lambda ~ 0.21` (Heyman et al.,
ref. 10 of the paper; the paper quotes exactly this for "steady 3D flows through
random bead packs"). Run single-phase in an RCP pack from `scripts/geometry/`, and
see whether the Eq. (12) estimator lands there.

Note `lambda` in `stretching.txt` is per **lattice step**; multiply by
`t_a = d/U` in lattice units to compare with published `lambda t_a`.

**Setup.** `felbm_gpu` refuses `model = single_phase`, so single-phase is done as
`fluid_initializer = uniform` with `concentration = 1` and `phi = 0` (neutral
wetting), which makes the capillary and wetting terms identically inert. Confirmed
empirically, not just by argument: `total_c` held at 9.16919e6 against 9,168,806
fluid sites, i.e. c = 1 to five figures. Geometry was the 8×8×12 pack at `d = 32`
(9.17M sites, **9.89 GiB actually allocated** — this is the measurement that
later corrected `--bytes-per-site` from 1700 to 1160, see §5). 50k
tracers, `volume` seeding, `Dm = 0`, `particles_velocity_skip = 100` (exact here,
the flow is steady). `nu = 0.1333`, interstitial `<u_z>` = 2.219e-3, **Re = 0.53**,
`t_a` = 14,420 steps, 100k steps = 6.93 t_a.

**Result.** `n_active` = 50,000 on every one of the 100,000 rows — nothing lost to
wall-blocking, so the ensemble is unbiased.

| window | 0–1 | 1–2 | 2–3 | 3–4 | 4–5 | 5–6 | 6–6.94 t_a |
|---|---|---|---|---|---|---|---|
| `lambda t_a` | 1.847 | 0.791 | 0.548 | 0.438 | 0.392 | 0.361 | **0.320** |

It starts at ~0, which is right — isotropic orientations give
`<rho_hat^T J rho_hat>` = tr(J)/3 = 0 in incompressible flow — rises as they
align, then decays. Over the last 3.9 t_a it averages 0.378, and the independent
`d<log rho>/dt` estimator gives 0.376: **agreement to 0.6%, so the machinery is
self-consistent.**

But it is **still falling at -11% per t_a at the end of the run**, so 0.320 is an
upper bound, not a measurement. Fitting the approach does not settle it: a power
law gives `lambda_inf` = 0.137 (rms 0.015), an exponential gives 0.353 (rms 0.029).
Those bracket 0.21 without resolving it, and the power law — the better fit, and
the one consistent with still drifting at 6.9 t_a — implies ~160 t_a to get within
10% of the asymptote. **Not confirmed, not refuted.**

**The part that matters beyond this check.** §1 and §4 claim Eq. (12) "converges
with ensemble size rather than integration time — a few advective times per point
instead of tens", and that is the whole affordability argument for the Ca sweep.
At 50,000 tracers and 6.9 t_a it does not hold. The *statistical* error does
converge with ensemble size, but the orientation-alignment **bias** decays in time
and no ensemble size fixes it.

Likely cause, worth testing before accepting the cost: the flow is incompressible,
so a volume-uniform ensemble stays volume-uniform and does *not* drift into slow
zones — but tracers in slow pores relax toward the Lyapunov direction at their own
local rate, far longer than `t_a` set by the mean velocity. The ensemble average
then drags on a heavy tail of slow tracers, which is what a power-law approach
looks like. If that is right, **flux-weighted seeding, or weighting the Eq. (12)
average by local speed, should converge far faster** — neither exists today
(`volume`, `plane`, `line`, `point`, `pairs`, `sheet`), so it is a code change.

**The normalisation, now checked against Heyman et al. directly** (ref. 10 = PNAS
117, 13359 (2020); open access via PMC7306761). It is *not* defined per unit time:

```
lambda  ==  d(log l) / d(X/d)  =  log2 / (Xc/d)  ~=  0.21
"can be converted into a mean stretching rate per unit time as lambda u/d"
```

with `u` the **mean pore (interstitial) velocity**, "determined from flow rate
together with the knowledge of the packing porosity". So a temporal rate
`lambda u/d` multiplied by `t_a = d/u` returns exactly 0.21. **The interstitial
reading used above is the correct one**, and the superficial alternative (0.90) is
ruled out. Also confirmed there: `lambda ~ 0.21` is the Lyapunov exponent, while
`mu = 0.29 +- 0.01` is the topological entropy — Eq. (12) targets the former.

**But the comparison is not like-for-like on geometry, and that is new.** Heyman's
medium is a *random loose* pack of beads settled under gravity at porosity ~0.5,
at Re ~ 7e-3. The measurement above is RCP at porosity 0.364, at Re = 0.53. A
tighter pore space plausibly stretches more, so some of the excess over 0.21 may
be physical rather than numerical. Two ways to settle it, in order of directness:

- voxelize a *loose* pack at porosity ~0.5 (`--shrink` opens an RCP pack toward it,
  see the levers below) and repeat — same code, same estimator, matched geometry;
- lower the forcing to Re ~ 1e-2 to match, at the cost of a proportionally longer
  run in lattice steps.

Neither is worth doing until the convergence problem above is fixed, since the
present number is an upper bound that has not settled.

### 3.5b The convergence problem — diagnosed and solved (no code change)

Run: 8×8×12 pack at **`d = 20`** (2.24M sites, 2.42 GiB — a cheap test vehicle),
single-phase as above, Re = 0.46, `t_a` = 6,553 steps, **20 t_a**, 50k tracers,
`n_active` = 50,000 throughout, with per-particle HDF5 dumps every `t_a`.

**Diagnosis.** The earlier guess — that slow tracers lag — is only half right.
Stretching is *non-monotonic* in local speed:

| speed decile | 0–10% | 30–40% | 60–70% | 80–90% | 90–100% |
|---|---|---|---|---|---|
| `<lambda t_a>` | 0.028 | 0.132 | **0.491** | 0.177 | 0.081 |

It peaks at intermediate speeds (shear layers along channel walls) and is small at
*both* ends: stagnant corners have no strain, fast channel cores are nearly uniform
flow. Local speed spans **564×** between the 10th and 90th percentile, and that
breadth is what sets the convergence: `lambda(t)` decays as **`lambda_inf + A/t`**
(alpha ~= 1.0). Reaching 5% by brute force would need ~80–125 t_a.

Note also *why* it decays from above. Over short times `rho_hat` locks onto the
largest eigenvector of the instantaneous strain-rate tensor, so Eq. (12) starts
near `<s_max>`, a shear-rate-scale quantity far above any Lyapunov exponent
(measured 1.77 in the first t_a). The decay is the **decorrelation of the
stretching direction along trajectories**, not orientations "still aligning".

**The fix is extrapolation, not a longer run and not a code change.** The 1/t form
is stable enough to fit and to validate out of sample — fit on `t <= tmax`, predict
the measured value at 19.3 t_a:

| fit on | predicts | vs measured 0.1984 |
|---|---|---|
| ≤ 6 t_a | 0.2447 | **+23%** |
| ≤ 8 t_a | 0.1949 | −1.8% |
| ≤ 12 t_a | 0.2021 | +1.9% |
| ≤ 20 t_a | 0.2002 | +0.9% |

and `lambda_inf` itself settles: 0.149 (10 t_a) → 0.171 (12) → 0.169 (15) →
0.165 (20).

> **Protocol for the Ca sweep: run 12–15 t_a per point and fit
> `lambda_inf + A t^-alpha` from `t0 = 3`.** That gives `lambda_inf` to ~±3%,
> against ~100 t_a for the same accuracy directly — an 8× saving with no change to
> `ParticleManager`. **6 t_a is definitively too short**, which is exactly where
> §3.5 stopped, and why it read 0.32.

**Consequences.**

- The apparent agreement with 0.21 in §3.5 was a **coincidence**: `lambda t_a`
  passes *through* 0.21 near 15 t_a on its way down. Extrapolated for this RCP
  pack, **`lambda_inf` ~= 0.165 ± 0.01**, ~21% *below* Heyman — and note that is
  the opposite direction from what the porosity argument predicts, since our pack
  is tighter (0.364 vs ~0.5) and should stretch more, not less.
- A **definitional mismatch worth settling before trusting §3.5 as validation.**
  Heyman's `lambda` = log2/(Xc/d) comes from a *folding-rate* argument, and their
  `mu` = 0.29 is the growth rate of total line length ln(L/L0). Total length grows
  as `log<rho>` = `lambda + sigma^2/2`, not `<log rho>` = `lambda`. Here
  `Var(log rho)` grows 0.617 per t_a, so `lambda + sigma^2/2` = 0.198 + 0.308 =
  **0.507 per t_a**. Linga's Eq. (12) is unambiguously the `<log rho>` one, so we
  are consistent *with the paper* — but the paper's own comparison against Heyman's
  0.21 may not be like-for-like.
- Speed-conditioning is a red herring for this purpose but worth knowing: the
  **top 50% by speed converges by ~7 t_a** and sits flat at 0.293 (drift −0.1% over
  the last half), while the unweighted mean is still moving at −9.5%. It converges
  because it drops the slow tail, but it measures a *different quantity* than
  Fig 5's `lambda`.

**Cost implication.** `t_a ∝ 1/Ca`, so at 12–15 t_a per point the sweep is
dominated by low Ca: ~6 h/card at Ca = 1e-2, but ~3 days/card at Ca = 1e-3 even on
an A5000. **The low-Ca end is both the expensive end and the end where clusters may
outgrow the box** — the two hard constraints coincide there.

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
- **Phase split — implemented** (was "left out on purpose"). `lambda_w` /
  `lambda_nw` / `n_w` / `n_nw` are now columns 6–9 of `stretching.txt`, on both
  the CPU and GPU paths. The concentration is sampled at the tracer position
  once per substep, at the same place the Eq. (12) contribution is taken, so a
  tracer crossing the interface mid-update contributes each substep to the phase
  it was actually in. Split at **`c < 0.5` = wetting** (`c` = 1 is fluid 0, and
  `rho1` = wetting). Sampling is nearest-node, not trilinear: the label is a
  threshold on a field whose interface is several lattice units wide, so
  interpolation cannot sharpen a decision that is genuinely ambiguous inside the
  diffuse interface.
  - `lambda_w` is `nan` **exactly when** `n_w` is 0 — no concentration attached,
    or that phase holds no tracers this step. Never 0, which would quietly bias
    an average over rows. Recombine by weighting with `n_w` / `n_nw`.
  - Cost, measured: the GPU path needs a 4th D2H + scatter, which is **+29% on
    `pt_scatter` and +3% on `pm_update`** (§4's estimate of +34% on the scatter
    was close). Disable with `particles_phase_split = false`. On the **CPU path
    it is free** — `Scheduler::accumulate_macroscopic_fields()` already gathers
    `m_concentration` every step, so it is simply attached.
  - Verified: `n_w + n_nw == n_active` on every row;
    `(n_w*lambda_w + n_nw*lambda_nw)/n_active == lambda` to 1.5e-6; and in a
    single-phase run every tracer lands in one phase and that phase's lambda
    equals the global lambda identically.
- **Isotropic initial orientations** need a few advective times to align with the
  Lyapunov direction. Discard that transient before averaging; `n_active` in the
  timeseries will tell you if wall-blocking is also biasing the ensemble.

---

## 5. Geometry pipeline — the ladder, now measured

`scripts/geometry/README.md` has the detail. The pipeline runs: `--self-test`
passes, and voxelising a real pack gives a porosity discretisation error of
3.7e-5 at `d = 20`, pore space percolating on all three axes, 0.003% dead volume,
and a `domain.cfg` whose key set matches `DomainInitializer_3DImage` exactly.

The ladder was run on a **real Yade RCP pack: 8×8×12 d, N = 930, `phi_solid` =
0.6357, porosity 0.3643, Z = 5.91** (isostatic for frictionless spheres is 6;
the deficit is the rattlers, which count in N but carry no contacts). That
porosity is what the original note guessed for RCP, against the 0.50 test pack
its first table came from. The prediction that RCP would be tighter was right,
and by more than the intermediate 114-sphere pack suggested — that pack sat at
porosity 0.385 and was optimistic at every rung:

| d (lu) | fluid sites | GPU mem | <1W | <2W | **<3W** | (<3W on the 114-sphere pack) |
|---|---|---|---|---|---|---|
| 12 | 483,722 | 0.52 GiB | 0.761 | 1.000 | **1.000** | 1.000 |
| 16 | 1,146,458 | 1.24 GiB | 0.501 | 0.974 | **1.000** | 1.000 |
| 20 | 2,239,241 | 2.42 GiB | 0.338 | 0.857 | **1.000** | 1.000 |
| 24 | 3,868,159 | 4.18 GiB | 0.236 | 0.703 | **0.985** | 0.929 |
| 32 | 9,168,806 | 9.91 GiB | 0.126 | 0.439 | **0.846** | 0.773 |
| 40 | 17,908,006 | **19.35 GiB** | 0.084 | 0.288 | **0.638** | 0.543 |
| 48 | 30,946,646 | 33.43 GiB | 0.060 | 0.207 | **0.475** | 0.389 |

(`W = interface_width = 5` lu; site counts are for the 8×8×12 d box itself.)

Porosity comes out 0.36434–0.36449 at every one of the seven grids against the
analytic 0.36435, and the pore space percolates on all three axes throughout.
That consistency is worth as much as the pore-size numbers: the rasterisation is
sound across the whole range.

The original table put 77.9% at `d = 20`. At true RCP that needs **`d ≈ 35`**.

### Where it crosses — and what the paper actually used

At porosity 0.364 and **1160 B/site** (measured — see below), against ~22 GiB
usable of a 24 GB card:

| d | MiB per d³ | max box | as a cube | this 8×8×12 box | paper's 4×4×8 |
|---|---|---|---|---|---|
| 20 | 3.22 | 6987 d³ | 19.1 d | 2.42 GiB | 0.40 GiB |
| 24 | 5.57 | 4043 d³ | 15.9 d | 4.18 GiB | 0.70 GiB |
| 32 | 13.21 | 1706 d³ | 11.9 d | 9.91 GiB | 1.65 GiB |
| 40 | 25.80 | 873 d³ | 9.6 d | **19.35 GiB** | 3.22 GiB |
| 48 | 44.57 | 505 d³ | 8.0 d | 33.43 GiB | 5.57 GiB |

> **These figures were all ~47% too high until 2026-08-02.** The voxelizer assumed
> 1700 B/site; `felbm_gpu` actually reports 10,129.8 MiB in use after init for
> 9,168,806 fluid sites on the fused + `stream_inplace` path in single precision,
> i.e. **1158 B/site**. Corrected in `voxelize_spheres.py` (`--bytes-per-site`
> default now 1160). Raise it without `stream_inplace` (+~150 B/site for the h2/g2
> ping-pong at D3Q19 single) or for a double build.
>
> The correction changes a conclusion: **`d = 40` on the 8×8×12 box is 19.35 GiB
> and fits a 24 GB card**, where the old figure of 28.35 GB ruled it out. That is
> 40 points per grain diameter against the paper's ~36, on a box six times larger
> than theirs — the best resolution/box combination available, and the one to aim
> the Ca sweep at, with `d = 32` (9.91 GiB) as the fallback if the two-phase solver
> needs more headroom than the single-phase run did.

An earlier revision of this section led with "the 20×20×30 box does not fit, and
that sets the low-Ca limit." **That was wrong**, and so was the first correction to
it. Both are recorded below because the second error is easy to repeat.

Table S1 of the paper (arXiv 2604.10382v1) gives the 3D run exactly:

| | paper (Table S1) | this work |
|---|---|---|
| packed region | 4 × 4 × 7 d, padded 0.5 d each end → **4 × 4 × 8 d³ = 128 d³** | 8 × 8 × 12 = 768 d³ |
| porosity | **0.39** (excl. buffer) | 0.364 |
| method | Navier–Stokes–Cahn–Hilliard, P1 FEM | conservative Allen–Cahn LBM |
| resolution | Δx = 2.8e-2 d = (V_f/N)^(1/3), ~36 pts per d | d = 32 lu |
| interface | **ε = 5e-2** | `interface_width` |
| bead diameter | target 1.0, actual **d' = 0.996** | target 1.0, actual 1.0008 |
| inlet velocity | u0 = 1e-3, σ = 20, θ0 = π/2, M = 1e-4, μ = ϱ = 1 | — |

Two things follow, and the second reverses the first correction.

**The box is not the constraint.** The paper's 3D domain is 128 d³, one sixth of
the 8×8×12 pack built here, and 1.65 GiB at `d = 32`. The 20×20×30 ambition comes
from `scripts/geometry/README.md` — a *steady* co-flow wants to hold the largest
clusters — not from the paper. That part of the correction stands.

(Note also that the paper's packed region is 4×4×**7** and the 8 comes from
padding. `--box 4 4 8` is therefore *not* the paper's geometry; use
`--box 4 4 7 --pad 0.5`. And the paper's pack is looser than ours, Φ = 0.39
against 0.364–0.367, so our constriction numbers are pessimistic relative to
theirs. Reassuringly, their target-vs-actual bead diameter mismatch, 0.996 against
1.0, is the same box-fitting rescale `yade_rcp.py` produces.)

### Interface width: ε and `interface_width` are not the same quantity

The previous revision claimed `interface_width = 5` lu at `d = 32` was "3.1× fatter
than the paper's ε = 0.05 d" and concluded that `d = 60` was needed. **That was an
apples-to-oranges error** — ε is a length inside the tanh argument, `interface_width`
is a full-width parameter. The conventions are:

```
paper   phi in [-1,1] :  phi = tanh( z / (sqrt(2) eps) )        (Eq. 12 of the SI)
felbm   c   in [0, 1] :  c   = 0.5 (1 + tanh( 2 x / W ))  =>  2c-1 = tanh( 2x / W )
```

Equating the arguments gives **W = 2√2 ε ≈ 2.83 ε**. So the paper's ε = 0.05 d is
an `interface_width` of **0.1414 d**, and the `d = 32`, W = 5 lu run was 0.156 d —
**1.10× the paper's, not 3.1×.** The `d = 60` / W = 3 recommendation was
over-resolving by a factor 2.8 for no reason.

**Corrected target: `d = 40`, `interface_width = 5.66`** (= 0.1414 × 40) on the
8×8×12 box — the paper's interface sharpness at *better* than its resolution (40
points per grain diameter vs ~36), on a box six times larger, at **19.35 GiB**.
Only the `--bytes-per-site` correction made this reachable; at the old 1700 it
scored 28.35 GB and was ruled out. Fall back to **`d = 32`,
`interface_width = 4.53`** (9.91 GiB) if the headroom proves too thin. Measured
at `d = 32`:

```
fluid sites 9,168,806    9.91 GiB
below 1W (4.5 lu) 0.111   below 2W (9.1 lu) 0.403   below 3W (13.6 lu) 0.762
```

Equivalently `W = 5` lu at `d = 35.4`, or `W = 4` lu at `d = 28.3`.

**What raising `d` at fixed `W/d` does and does not buy.** The constriction
fractions above depend only on the *ratio* `W/d`, so going 32 → 40 leaves them
essentially unchanged — it does not open up the pore space. What it buys is
numerical: 5.66 lattice points across the interface instead of 4.53, and a finer
representation of the grain surfaces. Choose it for accuracy, not to improve the
`<3W` number, which is fixed once ε is matched.

### What that says about the "<3W" criterion

The paper's own 3D simulation therefore runs at an interface-to-grain ratio where
**roughly three quarters of the pore volume is narrower than 3W** — and it works,
and is the published result. `scripts/geometry/README.md` treats the <3W fraction
as "the number to drive down"; the paper operates deep inside it. Treat that
criterion as advisory, not as a feasibility bound, and do not spend resolution
chasing it. The honest resolution requirement is simply to match ε: everything
else in this section was a consequence of mis-reading it.

```bash
yade -j 32 -x -n scripts/geometry/yade_rcp.py -- \
    --box 8 8 12 --phi-init 0.2 --seed 1 --out pack_8x8x12
scripts/geometry/voxelize_spheres.py pack_8x8x12 --res 40 \
    --interface-width 5.66 --out geom_d40
# then in params.cfg:  interface_width = 5.66      (d = 32 / 4.53 is the fallback)
```

For a like-for-like reproduction of the paper's geometry instead, use
`--box 4 4 7 --pad 0.5 --res 32`, which is 1.65 GiB and breaks z-periodicity (it
needs the open-boundary keys — that is the paper's *drainage* setup, not a
periodic co-flow).

### The two levers

**Narrowing the interface is much the stronger one.** At `interface_width = 3`:

| d | <3W (W=5) | <3W (W=3) | box memory |
|---|---|---|---|
| 20 | 1.000 | 0.818 | 2.42 GiB |
| 24 | 0.985 | 0.644 | 4.18 GiB |
| 32 | 0.846 | 0.403 | 9.91 GiB |
| 40 | 0.638 | 0.263 | 19.35 GiB |

W=3 at d=24 reaches 0.644 in 4.18 GiB, beating W=5 at d=32 (0.846 in 9.91 GiB) on
constrictions *and* memory at once, for a quarter of the card. **This is the first
thing to try**, and it is worth more than any amount of extra resolution.

**Shrinking the grains**, the paper's 2D analogue, at d=32 and W=5:

| shrink | porosity | fluid sites | mem | <3W |
|---|---|---|---|---|
| 0.00 | 0.364 | 9,168,806 | 9.91 GiB | 0.846 |
| 0.05 | 0.455 | 11,451,267 | 12.37 GiB | 0.749 |
| 0.10 | 0.537 | 13,505,093 | 14.59 GiB | 0.616 |

This is the poor lever. 10% shrink buys 0.846 → 0.616 while eating almost the
whole card and pushing porosity to 0.537, which is no longer a bead pack. It also
needs the single-phase permeability check against the FEM value before it is
defensible at all. Prefer the interface width.

Worth chasing separately: ref. 82 of the paper deposits *"Meshes and analysis
scripts"*. If the original Yade sphere list is in that dataset, voxelizing **it**
makes the LBM-vs-FEM comparison a same-geometry comparison instead of a
statistically-equivalent one. Given how tight §5 turned out, a sphere list that
fixes the geometry is worth more than it was when this was first written.

---

## 5b. Cluster sizes in 3D — the box is NOT the constraint

Run: 8×8×12 pack at `d = 20`, `W = 3` (0.150 d, vs the paper's 0.1414 d),
θ₀ = 60° (`phi = 6 σ cos θ` — see `felbm_local/docs/fluid_initializers.md`),
s_w = 0.5 from a slab IC, 50k tracers. Measured **Ca = 0.18, Re = 0.90**,
`t_a` = 3,323 steps, run 36 t_a. Analysed with `scripts/cluster_sizes.py`.

**Both phases percolate, and it is not a finite-size artefact.**

| phase | saturation | clusters | largest | spans (x,y,z) |
|---|---|---|---|---|
| non-wetting | 0.479 | 2487 | 132.4 d³ = **0.9932 of the phase** | (T, T, T) |
| wetting | 0.521 | 1003 | 144.1 d³ = **0.9921 of the phase** | (T, T, T) |

Each phase is one spanning network holding ~99% of its volume, plus a few
thousand specks. Already true at 3.6 t_a and unchanged to 36 t_a — it never
fragments. **A larger box will not change this**: 3D site percolation threshold
is ~0.31 and both phases sit near 0.5, well above it, so the spanning cluster is
genuine rather than an artefact of the domain.

**This says nothing about low Ca, and an earlier revision of this section wrongly
claimed it did.** Ca = 0.18 is the *high*-Ca regime, where system-spanning bands
are the known and expected state (connected pathway flow, Avraam & Payatakes).
Reproducing it is a sanity check that the solver gets a known regime right — not
evidence about the rest of the sweep. At low Ca capillary forces dominate,
snap-off and trapping set in, and discrete ganglia are what should appear. The
reasoning that was here — "`<A_c> ~ Bo^-1`, so if the highest Ca does not
fragment, none will" — inverts the regime physics and is withdrawn.

**Two consequences, and the second is the useful one.**

1. The 2D cluster framework does not transfer. `<A_c> ~ Bo^-1` and `N_c ~ Bo` are
   consequences of 2D topology, where two phases cannot both percolate at 50/50.
   A 3D `<V_c>(Bo)` at s_w = 0.5 is not a measurable quantity.
2. **In this regime the box-size objection does not bite** — you cannot fail to
   contain a cluster when the phase is one spanning network. But that is a
   statement about high Ca only. **At low Ca, where ganglia are expected, their
   size is a real physical quantity that must fit the box, and the concern
   returns.** Answering it needs a genuinely low-Ca run; this one cannot.
   Separately, the length scale the paper's model uses is `Delta = 1/s` with `s`
   the specific interface area — a *local* quantity, measured here (high Ca only),
   steady over 36 t_a:

   ```
   s = 3.27 / d      Delta = 1/s = 0.31 d
   ```

   i.e. sub-grain, against a box of 8–12 d, so `Delta` itself never needs a large
   domain. Whether the low-Ca end is limited by time (`t_a ~ 1/Ca`) alone, or by
   box size as well once ganglia form, is **open** — it is the first thing a
   low-Ca run should measure, with `scripts/cluster_sizes.py` on the dumps.

> Caveat on `s`: the voxel-face estimator overcounts area by ~1.5x for an
> isotropic surface (the standard staircase correction), so the true `Delta` is
> nearer 0.45 d. Systematic, and it does not affect the conclusion.

### The phase split, first physics

The same run gives Fig. 5's other claim directly. Over the last 11 t_a:

```
lambda = 0.2340    lambda_w = 0.2440    lambda_nw = 0.2205    ratio 1.107
```

**`lambda_w` and `lambda_nw` agree to 11%** — "filaments in the wetting and
non-wetting phases undergo similar stretching", reproduced in 3D.

Two caveats. `lambda` was still drifting -9.2% per 10 t_a at 36 t_a, exactly the
1/t behaviour of §3.5b, so these are unconverged and need that protocol. And the
tracer phase fractions (n_w/(n_w+n_nw) = 0.571) exceed the field saturation
(0.521) by 5 points: tracers accumulate in the wetting phase, which is the
interface-velocity mismatch the paper itself flags ("particles may, albeit rarely,
move from one phase to the other"). Worth quantifying before `lambda_w` is quoted.

### Performance: the host worker is the critical path, badly configured

This run held the GPU at **40–50% utilisation**. The tracking totals say why:
`scatter` 2462 s and `pm_update` 1470 s of a 4194 s wall, i.e. the worker is busy
94% of the time and the GPU waits for it.

The cause is a misconfiguration, not the code. `settings.threads()` is parsed but
`felbm_gpu` never applies it, `OMP_NUM_THREADS` is unset, and `particles_threads`
was not set in any config used here — so OpenMP defaults to `nproc` = 128 threads
on a **64-physical-core** host. The scatter is a random-access write
(`out[l2g[i]] = v`) into a 12 MB array; running it 2x oversubscribed onto SMT
siblings, which share L1/L2 and the TLB, is close to the worst case. It measured
3.5 ms per array, about 3.6 GB/s.

**Swept, and the effect is much smaller than the above implied** (2D Fig-5
benchmark, 1.58M sites, 50k tracers, 3000 steps):

| `particles_threads` | ms/step | scatter | pm_update |
|---|---|---|---|
| 4 | 74.24 | 24.6 | 48.2 |
| 8 | 41.19 | 14.8 | 24.9 |
| 16 | 25.72 | 10.2 | 14.2 |
| 32 | 17.23 | 7.2 | 8.6 |
| **64** | **13.48** | **5.8** | **6.4** |
| default (128) | 14.4–15.1 | 6.8–7.4 | 6.2–6.3 |

**Use `particles_threads = 64`** — one per physical core. The host work scales
cleanly to 64; the extra 64 hyperthreads cost ~10%, not the factor the
oversubscription argument suggested. Two corrections worth keeping:

- The 20.4 ms/step this section originally quoted was **contaminated** — that
  benchmark ran while a Yade pack generated at `-j 32`, stealing the cores the
  worker needs. Clean it is 14.4 ms at default, 13.5 ms at 64. Never benchmark
  against a busy machine.
- It is **still worker-bound at 64** (scatter 5.8 + pm_update 6.4 = 12.2 ms of a
  13.5 ms step). Thread tuning is worth ~8%; the remaining factor would have to
  come from fewer tracers or a cheaper scatter, not from the thread count.

Note `pm.update()` reduces with `reduction(+:...)`, so `lambda` is not
bit-reproducible across thread counts; pin the value for a campaign.

---

## 6. Environment, and the state of `yade_rcp.py`

Set up on this machine: `sudo apt install yade python3-scipy python3-tifffile`.
Yade is 2026.1.0 from Ubuntu 26.04 universe. The system Python 3.14 is
`EXTERNALLY-MANAGED` and `python3.14-venv` is not installed, so the apt packages
are the path of least resistance — and they keep scipy/tifffile on the same
interpreter Yade embeds.

`voxelize_spheres.py` needed nothing. `yade_rcp.py` had never successfully run,
and needed five fixes (felbm_gpu `60c3288` and `9006821`), three of which are
algorithmic rather
than environmental and would have bitten on any machine:

| | |
|---|---|
| `minieigen` | folded into `yade.minieigenHP`; packaged builds no longer ship the old module |
| arg parsing | Yade 2026.1.0 consumes the `--` separator, and the parser's "no `--` ⇒ no args" fallback then **silently ignored every flag** and produced a default pack |
| `nan` deadlock | `unbalancedForce()` is `0/0` with no contacts, and `nan` fails every comparison, welding the relax state machine shut until `max_steps` |
| livelock | the freeze fires on a *collisional* pressure spike ~200 steps after each restart; 119 freeze/thaw cycles advanced `phi` by 1e-4 each and never jammed |
| relax stall | below jamming `unbalancedForce()` need never converge at all, so a single relaxation eats the whole budget — **only appears as N grows**, see below |

The livelock fix adds `--phi-step` (default 2e-3): a jamming test costs a full
relaxation, so require `phi` to have advanced before doing another. The price is
that `phi_J` is resolved only to `phi_step` and the pack ends up compressed past
`p_jam` — in the test run the relaxed pressure was 2.6e4 against `p_jam = 1e3`,
i.e. residual overlap `delta/R ~ 4e-3` rather than the ~5e-4 the `--p-jam`
docstring advertises. Lower `--phi-step` tightens both, at more compute. **If the
permeability check comes out off, this overlap is the first thing to suspect.**

### The relax stall, and why small packs hide it

The fifth fix is the one to remember, because everything looked fine without it.
**Cundall damping is proportional to contact force, so a sphere with no contacts
is completely undamped.** At constant volume below `phi_J` most of the packing
stays ballistic indefinitely, and `unbalancedForce()` — which normalises by the
mean contact force — pins at ~4e-2 on the handful of sporadic contacts. It never
reaches `--unbalanced`, and it never hits exactly zero contacts either, so the
`nan` guard cannot fire. A 939-sphere pack sat in one relaxation for 800k steps at
`p ~ 1e-5` and died at `max_steps`.

At 117 spheres the system *does* reach zero contacts often enough to escape
through the `nan` path, which is why the first three fixes appeared sufficient.
**Anything tuned on a small pack should be re-checked at production N.**

`--relax-steps` (default 50000) bounds the relaxation, but the two exits are
deliberately *not* symmetric: it gives up only when the pressure has also
collapsed, which is unambiguously below jamming and means the answer is to keep
compressing. When `p` is still up, real contacts carry real forces, damping works,
and the unbalanced force does converge — so the test is allowed to finish.

Sanity checks that the DEM is doing what it claims: the 8×8×12 pack jams at
`phi = 0.6357` (RCP is ~0.64) with coordination number **Z = 5.91** against the
isostatic 6, the deficit being rattlers. The overshoot past `p_jam` that §5's
`--phi-step` note warns about is also much milder at realistic N — 2.2× here
versus 26× on the 114-sphere pack — so the residual-overlap worry is a
small-pack artefact more than a real one.

### Cost, and threads

Yade has OpenMP (`yade -j N`), and it matters, but it **does not scale far**.
Measured on contact-rich lattices with the same engine list, steps/s:

| threads | 1000 bodies | 13,824 bodies |
|---|---|---|
| 1 | 2079 | 53 |
| 8 | 4159 | 287 |
| 16 | 6238 | 399 |
| **32** | **7127** | **543** |
| 64 | 6239 | 475 |
| 128 | 53 | 46 |

Peak is ~32; 64 is worse, and 128 collapses below single-threaded (OpenMP threads
spinning on barriers, oversubscribed). Parallel efficiency at the peak is only
11% (1000 bodies) to 32% (13,824). **Use `-j 32`.** On a many-core host the better
use of the rest is several independent packs at `-j 16` concurrently — different
`--seed` values are the realisations you want anyway.

Wall time is not the obstacle it looked like: the 8×8×12 pack (N = 930) took two
outer iterations at ~7 min each at `-j 16`, and converged inside the 0.1%
tolerance on the second. The outer loop *can* reach that tolerance at this N,
unlike the 114-sphere pack where one sphere is ~0.3% of the volume and it
correctly warns and returns its best instead.

One trap left: **`--phi-init`'s default of 0.30 cannot generate this box.**
`makeCloud` is RSA with a 1000-try cap, so the usable initial solid fraction falls
as N grows; at N = 939 it placed 897 of 939 and the script treats a short
placement as fatal. Pass `--phi-init 0.2`. Worth either lowering the default or
having it retry at a lower value rather than dying.
