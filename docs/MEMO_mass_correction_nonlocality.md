# Memo: non-locality of the order-parameter mass correction

Status: **open issue, diagnosed and quantified, not fixed.** Nothing here changes the
code as it stands. Written 2026-07-26.

Related: `felbm_local/docs/mass_correction.md` (what the code does),
`felbm_gpu/README.md` (GPU implementation), `docs/si_template_2019.tex` (write-up).

---

## 1. The issue

The corrector uses a single global scalar

    lambda = (M - M0) / W,     dC_i = -lambda * phi_i,     phi_i = C_i (1 - C_i)

It restores the global total exactly, adds no momentum, and acts only inside the
diffuse interface. But `lambda` does not know *where* the error was generated. If
region A generates spurious mass and region B does not, both are corrected in
proportion to their interface weight, so B is drained to pay for A. Global
conservation holds; **local conservation does not**. Mass moves between regions with
no transport process connecting them — in the limit, between disconnected ganglia.

The continuum Cahn-Hilliard equation is in flux form (`dC/dt = M grad^2 mu`) and so
conserves C locally. Our correction is a source term, which is strictly weaker.

For a study whose subject is *dynamic fluid connectivity*, a numerical operator that
transfers phase between disconnected clusters is exactly the wrong artifact to have.

## 2. Diagnosis (measured, not estimated)

Method: for a region bounded by planes lying in **bulk** phase, the Cahn-Hilliard flux
`M grad(mu)` vanishes there and the advective flux is measurable, so the residual of
the mass budget is the locally generated error:

    e_R = dM_R/dt - Phi(y_lo) + Phi(y_hi),     Phi(y) = sum over plane of C * u_y

Run: 2D cylinder pack 1024x2048, slab IC (interfaces at y=512 imbibition, y=1536
drainage), corrector OFF, 12000 steps, field output every 500. Flux planes in bulk at
y = 256, 1024, 1792. Budget closes: `e_L + e_U = 0.0091/step` vs measured global drift
`0.0091/step`.

| interface | weight W | error e (per step) | **e/W** |
|---|---|---|---|
| lower (imbibition) | 954.6 | 0.00266 | 2.79e-6 |
| upper (drainage)   | 1311.3 | 0.00635 | **4.84e-6** |

**The drainage front generates 1.7x more error per unit interface than the imbibition
front.** With a global lambda the corrector removes 0.00380 from the lower region
(which generated 0.00266) and 0.00521 from the upper (which generated 0.00635):

    net: -0.00113/step from imbibition side, +0.00113/step to drainage side

i.e. **13% of the applied correction is spurious transport**, or 0.17% of total phase
volume per 1e6 steps, against the 1.4% drift it removes. An ~8x improvement, not a fix.

Physical reading: truncation error tracks interface acceleration; drainage fronts are
capillary-destabilised and move in jumps, imbibition fronts are film-stabilised. This
matches the velocity-variance asymmetry measured independently on the same two planes
(normalised Var(|v|): 0.71 drainage vs 0.56 imbibition, growing to 0.94 vs 0.67 at
density ratio 16). So the bias is systematic and always runs from the quiescent side
toward the active side.

Caveats: (i) these two interfaces are connected through bulk, so this demonstrates
*mis-allocation*, not directly inter-ganglion transfer — the decisive test needs
solid-isolated pores and has not been built; (ii) the ratio was still drifting (0.73 ->
0.54) at 12000 steps, so 1.7x is likely a lower bound for developed flow.

Reproduce: `/tmp/diag` recipe — cfg from `felbm_gpu/cfg/2d_cylinders/`, set
`correct_op_mass = false`, `file_skip = 500`, `particles_enable = false`.

## 3. Literature position (checked 2026-07-26)

Our scheme is **not** naive. The relevant lineage:

- **Rubinstein & Sternberg (1992)**, IMA J. Appl. Math. 48, 249: `phi_t = ... + beta(t)`,
  correction added *uniformly everywhere including bulk*. This is the one with the
  documented coarsening / critical-droplet-radius failure.
- **Brassel & Bretin (2011)**, Math. Methods Appl. Sci. 34(10), 1157: adds the interface
  weight, `phi_t = ... + beta(t) sqrt(2F(phi))`.
- **Kim, Lee & Choi (2014)**, Int. J. Eng. Sci. 84, 11-17 ("space-time dependent
  Lagrange multiplier"). PDF: `~/code/A_conservative_Allen_Cahn_equation_with.pdf`.

**Key finding: we already implement Kim et al. exactly.** Their title is misleading —
beta is still a single scalar per step (their Eq. 14); the "space-time dependence" is
in the *weight*. With their potential `F = 0.5 phi^2 (1-phi)^2`,

    sqrt(2F) = phi(1-phi)

which is literally our `phi_i`. Their update `phi^{n+1} = phi^{n+1,2} + dt*beta*sqrt(2F)`
expands to `phi_i - (M-M0) phi_i / W` — algebraically identical to ours, term for term.

**Consequence: Kim et al. is not a fix for section 2.** They cure RS's bulk correction;
they do not address a spatially non-uniform error density along the interface, because
their test cases are droplets in quiescent fluid where it is roughly uniform. In driven
porous flow it is not. That literature line is exhausted for our purposes.

## 4. Options if this needs fixing

Cost baseline: 300^3 step ~57 ms; one stencil pass ~4 ms; current corrector ~3 ms (6%).
Costs below are estimates from pass counts, NOT measured.

| option | fixes sec. 2? | every step | every 100 steps |
|---|---|---|---|
| flux form, `dC = div(D grad psi)` (Poisson solve) | yes | +90-260% | +2-3% |
| per-connected-component beta | only disconnected case | +70-140% (CCL dominates) | +1% (but labels stale exactly when ganglia merge/snap) |
| conservative Allen-Cahn (Chiu & Lin 2011; Geier et al. 2015 LBM) | by construction | model change | - |

Amortising is legitimate: the drift is 1.4e-8 relative per step, so correcting every
100 steps still holds it at 1.4e-6. Nothing physical changes.

Warm-started CG for the flux form is the number worth measuring first — the drift field
evolves slowly, so ~10 iterations may suffice, which would make flux-form-every-50-steps
~2% and strictly the best answer.

The 2025 comparative study (arXiv:2511.11360) benchmarks conservative AC / nonlocal AC /
hybrid AC / Cahn-Hilliard LB models and favours conservative AC overall — but reports
that *all* families still lose droplet volume in 3D turbulence at high Weber number.

## 5. What to do before it matters

Cheap and worth doing regardless:

1. **Bound the artifact for the current results.** Track the ganglion size distribution
   with corrector on vs off over a production-length run. If the distributions agree,
   the artifact is below the level that affects the connectivity conclusions and can be
   reported as such.
2. **Build the disconnected-ganglion test** (two pores isolated by solid, one with a
   moving interface, one quiescent) to demonstrate or bound inter-ganglion transfer
   directly. This is the test the paper's thesis actually needs.
3. Note in any write-up that the correction conserves globally, not locally, with the
   0.17%-per-1e6-steps bound. Better stated by us than found by a referee.

## 6. Unrelated but noted from Kim et al.

Their operator splitting solves the nonlinear sharpening term `phi_t = -F'(phi)/eps^2`
**analytically** (their Eq. 11, separation of variables), which is what makes the scheme
unconditionally stable in that substep. Independent of the mass question; worth a look
if interface stiffness ever limits the time step.
