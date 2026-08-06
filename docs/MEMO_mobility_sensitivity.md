# Memo — phase-field mobility does not explain the `lambda` shortfall

**2026-08-04.** Negative result, recorded so it is not re-investigated.

> **2026-08-06 (morning) — upgraded from "null" to "exonerated".** This scan was
> run under the slab IC, where `lambda` was decaying toward zero for an unrelated
> reason, so a null result here proved less than it looked. It now has independent
> support: `scripts/interface_length.py` measures specific interface length within
> ~10% of the paper's Fig. 3B at every Ca up to 2.5e-2.
>
> **2026-08-06 (later) — that exoneration was over-claimed. RETRACTED.** The
> Ca = 0.0965 point has now run 63 t_a and the picture changed. Interface length
> does coarsen as predicted — `sd` peaks at 1.250 (13 t_a) then falls monotonically
> to 0.903, clusters 410 -> 191 — but the paper's Fig. S4 has this Ca flat at 0.06
> from ~27 t_a, while ours is still descending at 63 (`lambda` 0.19, −0.8%/t_a).
> **The difference is a COARSENING RATE, not a run length.**
>
> The agreement that produced "exonerated" was measured at 3–8 t_a, before
> coarsening matters, so it constrains the INITIAL interface structure and says
> nothing about the rate. The low-Ca match (`sd` 0.301 at 3.4 t_a against their
> converged 0.312) may simply be two curves crossing.
>
> **2026-08-06 (later still) — the "conservative Allen-Cahn" claim above was
> wrong; struck.** Checked against the source: there is no sharpening counter-term
> anywhere in `LBM/include/multi_phase/`, and the update carries
> `mobility*m_avg_lapl_mu(k)` (`lbm_force_term_multi_phase.h:166`) — an explicit
> Cahn-Hilliard flux `M grad^2 mu` with a chemical potential built from a double
> well. **felbm IS Cahn-Hilliard (Lee-Liu), the same class as the paper.** So
> there is no model-class escape hatch, and the scan in this memo tested the right
> parameter of the right model.
>
> What the scan did NOT cover: values ABOVE the 0.02 baseline. It only went down,
> toward the NaN floor at ~0.002. Since M sets the Ostwald-ripening rate in a
> Cahn-Hilliard model, RAISING it is the one untested direction with a mechanism
> behind it.

## Why it was tested

The 2D reproduction of Fig. 5 of Linga et al. undershoots badly: at
Ca ~ 1e-2 we measure `lambda t_a` = 0.145 (window mean, 15 t_a) against the
paper's **0.33 +- 0.03**, a factor 2.3 — and our value is still falling, so it is
an upper bound.

The suspicion was the phase-field mobility. felbm derives
`mobility = mobility_coeff / beta` with `beta = 12 sigma / W`, giving
`mobility = 1.181` for the sweep configuration, hence

```
Sc_PF = nu d / mobility = 2.4        (ours)
Sc_PF = nu d / M        = 1e4        (paper, Table S2)
```

a factor ~4200. The paper states Sc_PF "should ideally be as large as numerically
feasible", so being four orders the wrong side of that looked like a real defect,
and mobility plausibly damps the interface dynamics that drive stretching.

## Setup

Single Ca point, everything else held at the sweep configuration, so mobility is
the only variable:

| | |
|---|---|
| geometry | 2D RSA cylinders, 60d x 90d = 1260 x 1890 at d = 21 lu, porosity 0.62 |
| interface | `interface_width` = 3.0 (eps_equiv = 0.0505 d vs the paper's 0.05) |
| `surface_tension` | 0.004233 (Oh = 0.447, so Re = Ca/Oh^2 < 0.5) |
| contact angle | `phi` = 0.012700, i.e. theta_0 = 60 deg |
| saturation | s_w = 0.5, slab IC |
| tracers | 20k, `volume` seeding, `Dm` = 0, stretching + phase split on |
| length | 99,225 steps = 15 t_a at the realised Ca ~ 0.099 |
| hardware | RTX 3090, 64 cores, single precision |

Values tested: `mobility_coeff` = 0.02 (baseline), 0.01, 0.005, 0.002, 1e-3,
1e-4, 5e-6.

## Results

| `mobility_coeff` | x base | realised Ca | `lambda`(window) | `lambda_inf` | Var(log rho) | `lambda` at 3-6 / 6-9 / 9-12 / 12-15 t_a |
|---|---|---|---|---|---|---|
| 0.020 | 1.00 | 9.933e-2 | 0.1268 | 0.075 | 3.46 | 0.236 / 0.165 / 0.137 / 0.125 |
| 0.010 | 0.50 | 9.926e-2 | 0.1263 | 0.078 | 3.51 | 0.233 / 0.162 / 0.137 / 0.127 |
| 0.005 | 0.25 | 9.913e-2 | 0.1250 | 0.073 | 3.45 | 0.233 / 0.163 / 0.138 / 0.124 |
| 0.002 | 0.10 | 9.890e-2 | 0.1258 | 0.079 | 3.41 | 0.234 / 0.160 / 0.138 / 0.125 |
| 1e-3 | 0.05 | — | **NaN by step 1000** | | | |
| 1e-4 | 0.005 | — | **NaN by step 1000** | | | |
| 5e-6 | 2.5e-4 | — | **NaN by step 1000** | | | |

**Null.** Over a 10x reduction `lambda` changes by -1.4%, non-monotonically, so
that is scatter and not a trend. `lambda_inf`, Var(log rho) and the realised Ca
are flat, and the decay trajectory is superimposable window by window. `lambda`
needs **+128%** to reach the paper; the whole accessible range of this parameter
buys ~1%.

Below ~0.002 the solver NaNs within 1000 steps. In this Lee-Liu formulation the
mobility enters as `P.mobility * avg_lap_mu` — the Cahn-Hilliard flux `M grad^2 mu`
— which is what *maintains* the interface at its equilibrium tanh profile against
sharpening by the flow. Remove it and the interface goes singular. So the
mobility is a **regulariser with a hard floor**, not a free parameter, and the
paper's Sc_PF is unreachable here by four orders of magnitude.

## What this rules out, and three corrections

- **Mobility is not the cause of the `lambda` shortfall.** Neither the converged
  value nor the `lambda_inf + A/t` decay shape responds to it.
- **The Sc_PF = 2.4 vs 1e4 comparison was almost certainly invalid.** felbm's
  `mobility_coeff/beta` is evidently not the paper's `M`: a code sitting 4200x
  from a well-posed optimum would not be both stable and insensitive.
- **The reasoning that lower mobility should *raise* `lambda`** (less damping of
  interface deformation) had the role backwards — it is the regularising term.
- **A related hypothesis, also dead:** that the paper's refined material strip
  gives a rho-weighted ensemble measuring `d log<rho>/dt = lambda + sigma^2/2`
  rather than our `d<log rho>/dt`. The paper's own text rules it out — the strip
  is the Fig. 4 visualisation, while `lambda` comes from "10^4 passive particles
  ... distributed uniformly in space throughout both phases with random initial
  orientations", i.e. **identical sampling to ours**.

## What survives

The real discrepancy is qualitative, and untouched by any of the above: the paper
reports `lambda~(t)` tending to a **constant** after a transient, averaged over
t in [30 t_a, 60 t_a]. Ours **decays as `lambda_inf + A/t`** and is still falling
at 15 t_a — so running to their window would push our number *down*, not up. At
matched time we are ~2.3x low.

Remaining candidates, in order: **resolution** (they use dx = 0.028 d, i.e. 36
points per grain diameter, against our 21 — our interface spans ~1 lattice unit
against their ~1.8 elements); and the solver itself (conservative Allen-Cahn LBM
vs Navier-Stokes-Cahn-Hilliard FEM). A single-phase 2D check separates "our
`lambda` estimator reads low" from "our two-phase dynamics are too weak" and is
cheaper than the resolution test.

Raw runs: `~/runs/2d_mobscan/mob*` and `~/runs/2d_mixing_local/ca8.6e-2`.
