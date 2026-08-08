# SPEC — frozen-field tracer tool (`felbm_trace`)

Status: **proposal, not implemented.** Written 2026-08-08 so the design can be
argued with before any code exists.

## What problem this solves

The single-phase Lyapunov control against Heyman et al. (`lambda ~ 0.21`, docs
§3.5) is currently unaffordable and inconclusive for one reason: the
**orientation-alignment bias**. Eq. (12)'s statistical error converges with
ensemble size, but the bias decays *in time*, and no ensemble size fixes it. The
measured run reached 0.320 at 6.9 t_a while still falling at -11%/t_a, and the
power-law fit — the better of the two — implies **~160 t_a** to get within 10% of
the asymptote.

Two independent costs make that expensive, and this tool removes both:

1. **The flow is re-solved every step even though it is steady.** For a
   single-phase control the velocity field stops changing after a few t_a, yet
   `gpu.step()` runs for all 160 t_a.
2. **The ensemble is seeded uniformly by volume.** §3.5's diagnosis: tracers in
   slow pores relax toward the Lyapunov direction at their own local rate, far
   slower than t_a set by the *mean* velocity, so the ensemble average drags on a
   heavy tail of slow tracers. That is what the power-law approach IS.

## What already exists — do not rebuild it

Checked against the source, not assumed:

| capability | where | state |
|---|---|---|
| velocity field on disk | `felbm_gpu_main.cu:79` writes `u_x`, `u_y`, `u_z`, `pressure` into `output_<step>.h5` | works, `output_all = true` |
| tracer advection + Eq. (12) stretching | `ParticleManager::update()`, `StretchRow` | works |
| tracer state save/restore | `ParticleManager::initialize_from_hdf5()` | works, used by every extension |
| **flux-weighted seeding** | `weight_accept()`, cfg key `particles_init_weight = none\|ux\|uy\|uz\|u` | **exists and is unused** |
| holding the velocity constant | `particles_velocity_skip` (`felbm_gpu_main.cu:563`) | works; skips `d2h` + `scatter`, NOT `gpu.step()` |
| time-interpolated velocity | `ParticleInterpolator` holds *references* to `ucx/ucy/ucz` (current) and `unx/uny/unz` (next) | works |

**§3.5 is out of date where it says flux-weighted seeding "does not exist today
... so it is a code change".** It exists. It is disabled everywhere, and every
run config says why:

```
particles_init_weight = none     # uniform (flux-weighting fails at t=0 when u=0)
```

Seeding happens at t = 0, when the velocity field is identically zero, so
`weight_accept` rejects every candidate and `rand_fluid_point` falls through to
its fallback. **Loading a converged field before seeding is what makes the
existing feature usable** — that is the core of this proposal, and it is why the
frozen-field split buys more than just speed.

## Design

Two stages, no mid-run mode switching.

### Stage 1 — produce the field (no new code)

Run `felbm_gpu` single-phase as §3.5 does (`fluid_initializer = uniform`,
`concentration = 1`, `phi = 0`, so capillary and wetting terms are inert) for
long enough to reach steady state, with `particles_enable = false`. Keep one
`output_<step>.h5`. Steadiness is asserted, not assumed: require
`max|u(t) - u(t - 1 t_a)| / max|u| < 1e-3` between the last two dumps, and the
tool refuses to start if the pair it is given fails that.

### Stage 2 — `felbm_trace`, new binary

```
felbm_trace settings.cfg --field out/output_<step>.h5 [--field2 <later dump>]
```

Reads geometry the same way `felbm_gpu` does (so `domain.cfg` and the
`cylinders_repulsive` / image path are unchanged), loads `u_x/u_y/u_z` into the
six interpolator vectors — **both** `uc*` and `un*` set to the same field so the
time interpolation is exact and inert — then loops `pm.update()` and the existing
stretching output. No LBM, no `d2h`, and `scatter` runs **once at load** instead
of every step.

`--field2` is optional and exists only for the steadiness assertion above.

Everything downstream is unchanged: same `stretching.txt` format, same
`series.txt`, so `splice_legs.py`, `export_tables.py` and the analysis snippets
work without modification.

### Config keys — all existing

```
particles_number         = 50000
particles_init_mode      = volume            # or any existing mode
particles_init_weight    = none | u | uz     # THE point of the exercise
particles_stretching     = true
particles_dt             = 1.0
particles_dm             = 0.0
max_iterations           = <160 t_a worth>
```

## Expected cost

From the 2D benchmark (1260x1890, 20k tracers, 32 threads): total 246.4 s per 20k
steps, of which `d2h` 24.2 + `scatter` 133.7 + `pm_update` 87.7. Stage 2 keeps
only `pm_update`, so ~**2.8x** on host work before counting the LBM that no
longer runs at all. The GPU step cost, currently hidden behind host work and
therefore never measured, disappears entirely.

That is the *lower* bound on the benefit. If flux weighting shortens the
convergence horizon from ~160 t_a, that is the larger win by far, and it is the
one this tool exists to test.

## The definitional question — settle it with the tool, do not assume

Volume seeding and flux seeding agree on `lambda_inf` **only if** the tracer
dynamics are ergodic over a single component. In a porous medium they may not be:
dead-end and stagnant pores hold tracers that essentially never stretch. Volume
seeding includes them and drags `lambda` down; flux weighting excludes them. So
the two can converge to genuinely **different values**, not merely at different
rates.

This matters for both comparisons:

- **Heyman** measures `d(log l)/d(X/d)` by experimental particle tracking —
  elongation per distance *travelled* — which is naturally trajectory/flux
  weighted. Flux seeding is arguably the closer match to 0.21.
- **Our own 2D and 3D sweeps** all run `particles_init_mode = volume` with
  `particles_init_weight = none`. A flux-seeded control is therefore NOT directly
  comparable to our two-phase points.

**So run both on the same saved field and report the pair.** One extra cheap run.
If they agree, ergodicity is not a concern and flux weighting is a free
convergence speedup that the sweeps should adopt. If they differ, that difference
is a result in itself, and it constrains which convention Eq. (12) implies.

## Defects to fix in `weight_accept` before trusting it

Reading it (`lbm_particle_manager.h:610`):

```cpp
return m_uni(m_gen) < val/(val+1.0);
```

1. **The normalisation is only accidentally correct.** Lattice velocities here are
   ~1e-4..1e-3, so `val/(val+1) ~= val` and the sampling is proportional to speed
   as intended. At larger `u` it saturates toward 1 and silently degrades to
   uniform seeding. Should be `val / val_max` with `val_max` precomputed over
   fluid sites — cheap, and the field is already in memory.
2. **The fallback is a silent failure.** `rand_fluid_point` tries 10000 times and
   then returns the fixed point `(m_x0, m_y0, m_z0)`. With acceptance ~1e-3 the
   per-particle failure probability is ~4e-5, so a 50k ensemble expects a couple
   of tracers piled on one location. Should count fallbacks and refuse to run if
   any occurred.
3. **Rejection is over the whole box**, including solid, so the effective
   acceptance is further reduced by porosity. With `val_max` normalisation and a
   fluid-site list to sample from instead of the bounding box, seeding becomes
   O(N) rather than O(N/u).

## Acceptance criteria

1. With `particles_init_weight = none` and a frozen field, `felbm_trace`
   reproduces a `felbm_gpu` run of the same length on the same field to within
   tracer-noise on `lambda(t)` — same seed, same count. This is the regression
   that proves the tool changes nothing but cost.
2. Zero seeding fallbacks reported.
3. The steadiness assertion passes on the field pair used.
4. Both seedings run to >= 60 t_a on the same field, reported together with the
   drift rate, so the reader can see whether either has actually converged.

## Effort and risk

Small and well contained: geometry loading, the interpolator, the tracer loop,
the stretching output and the seeding all exist. The new code is a `main` that
loads a field instead of stepping an engine, plus the three `weight_accept`
fixes. Most of the risk is in the definitional question above, which is a physics
question the tool is designed to answer rather than a coding risk.

The one thing NOT to do is add a "freeze the flow at step N" flag to
`felbm_gpu`. It would couple the two concerns, make the field un-reusable, and
leave the seeding bug in place — the seeding would still happen at t = 0.
