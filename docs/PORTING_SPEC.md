# felbm_gpu porting spec — CPU → GPU mapping

This maps each CPU multiphase routine to its GPU counterpart and records the exact
formulas, so the kernels can be checked against `felbm_local` line-by-line and the
port can be completed/optimised with confidence.

## One time step (order matters)

CPU `TimeStepperMultiPhase::do_time_step`:
`compute_fields → [apply_open_bnd] → compute_body_forces → collide → stream`.

The GPU driver (`felbm_gpu_main.cu`) reproduces this exactly:

1. **compute_fields** (`FieldManagerMultiPhase::compute_fields`)
   - `k_moments` — pass 1: `c=Σ hₖ`, `p=Σ gₖ`, `u=Σ eₖ gₖ`, then
     `ρ=ρ1+Δρ·c`, `u /= cs²ρ`, `μ_bulk=4β·c(1−c)(0.5−c)`,
     `τ=1/(c⁺/τ0+(1−c⁺)/τ1)` with `c⁺=clamp(c,0,1)`, `rp=1/(τ+0.5)`.
   - `A_grad_cd·c`, `A_grad_bd·c`, `k_grad_md` → `∇c_cd, ∇c_bd, ∇c_md=½(cd+bd)`.
   - `A_lap·c`, then `k_mu_axpy`: `μ += −κ·∇²c`  → `μ = 4β·c(1−c)(0.5−c) − κ∇²c`.
   - `A_lap·μ` → `∇²μ`.
   - `k_vel_press_corr`: `u += ½(g + f·ff/ρ + (μ/ρ)∇c_cd)`,
     `p += ½·Δρ·(u·∇c_cd)·cs²`.
   - `A_grad_cd·p`, `A_grad_bd·p`, `k_grad_md` → `∇p_cd, ∇p_bd, ∇p_md`.
   - `A_cd_dir·c`, `A_cd_dir·p` → per-direction `∂c_k, ∂p_k` (central).
2. **compute_body_forces** (`ForceTermMultiPhase::update`)
   - `A_bd_dir·c`, `A_bd_dir·p` → per-direction biased derivatives.
   - `A_avg_dir·∇²μ` → direction-averaged Laplacian of μ.
   - `k_force_term` → `force_h`, `force_g` (Guo forcing + surface tension).
3. **collide**
   - `k_equilibria` (`CollisionModelMultiPhase::compute_all_equilibria`) → `eq_h, eq_g`.
   - `k_collision_term`: `coll_h = eq_h − h`, `coll_g = rp·(eq_g − g)`.
   - `k_collide_apply`: `h += coll_h + force_h`, `g += coll_g + force_g`.
4. **stream**: `A_stream·h`, `A_stream·g`, swap buffers (halfway BB baked into the operator).

## Equilibrium constants (D3Q19, cs²=1/3)

`ALPHA = 1/cs² = 3`, `GAMMA = 1/(2cs²) = 1.5`, `BETA = 1/(2cs⁴) = 4.5`.
`u_term = uₖ(ALPHA + BETA·uₖ) − GAMMA·u²`, `G = wₖ(1+u_term)`.

## Per-direction operator index convention (IMPORTANT)

The CPU `FieldOperator` per-direction operators (`cd_dir`, `bd_dir`) have
`(Q−1)·n` rows: direction 0 (rest) has no gradient. The results are stored into a
`Q·n` buffer starting at **direction 1** (`grad_c_cd_dir(1u)`), leaving slot k=0 as
zero. The driver reproduces this by `cudaMemset(buf,0,Q·n)` then
`spmv(A_cd_dir, field, buf + n)`. The kernels then read `buf[k*n+i]` for `k=0..Q−1`
with slot 0 = 0.

`avg_dir` has `Q·n` rows (`m_size_var + m_size_sites`), maps the scalar `∇²μ`
(length n) to a `Q·n` per-direction field; written from base (no offset). Its rows
are ordered `row = m·n + i` for `m = 0..Q−1` (verified against
`create_averaging_matrix_dir`), which matches the `[k*n+i]` reads in `k_force_term`.
**Confirmed correct** by the porous validation below.

## Node-centred Laplacian: length-2n extended input (IMPORTANT)

`compute_laplacian` is the one operator whose matvec has **`2n` columns**, not `n`:

```
matrix_vector_product( n, 2*n, m_rows_lap, m_cols_lap, m_laplacian, field, out );
```

Its input must be a length-`2n` buffer `[ field(n) , boundary(n) ]`. The second half
carries the wetting/contact-angle boundary condition, and **near-wall rows reference
columns ≥ n**. In the CPU code the two halves are the consecutive scratch buffers
`buffer(7)`/`buffer(8)`:

- `∇²c` : boundary half = `bnd_cond = boundary_coefficient · c(1−c)`  (`k_pack_lap_c`)
- `∇²μ` : boundary half = `0`                                          (`k_pack_lap_zero`)

Passing only the length-`n` field reads past the array end for near-wall rows → wrong
`mu` (and instability) in porous domains. This was the one real port bug the obstacle
harness caught; fixed by packing the 2n input before every `A_lap` SpMV. It is also
required for non-neutral wetting (`phi ≠ 0` ⇒ non-zero boundary half), independent of
the out-of-bounds issue.

## Validation status

Both regimes reproduce the CPU `EngineMultiPhase` to **machine precision** (`max|Δ|
~ 2e-16` on `h`/`g` after 20 steps, double precision), via `compare_cpu_gpu`:

- **all-fluid periodic** (`geom=fluid`) — bulk physics.
- **porous** (`geom=spheres`, `use_halfway_bb=true`) — bounce-back streaming, biased
  near-wall stencils, and the node-centred Laplacian + wetting BC.

Scope validated: BGK, body-force/periodic, double precision (the GRL regime, minus
MRT and particle tracking — see below).

## Memory (naive v0, per fluid site, doubles)

Distributions 4·Q, per-direction temporaries ~13·Q, ~30 scalar fields, plus the
uploaded CSR operators. Roughly ~4–5 KB/site — matches the "naive port" estimate in
`../felbm_local/docs/gpu_and_coolbm_notes.md`. Fine for correctness; see below to
shrink it.

## Optimisation roadmap (after validation)

1. **Matrix-free stencils.** Replace the CSR SpMV operators with direct D3Q19
   stencil kernels (compute neighbours on the fly from a small neighbour table or
   coordinates + bounce-back flag). Removes the largest memory cost and the SpMV
   indirection. This is where most of the speed-up and the "full 300³ on 32 GB"
   headroom comes from.
2. **Fuse passes.** Merge the many small per-direction kernels; keep distributions
   resident; recompute cheap temporaries instead of storing them.
3. **Single precision** where the physics tolerates it (`-DFELBM_SINGLE`).
4. **AoSoA / coalescing** tuning of the `k*n+i` layout for the memory-bound loops.
5. Then add MRT, open boundaries, mass correction, and GPU particle tracking.

## MRT collision (ported)

`use_mrt = true` uses the multiple-relaxation-time `g`-collision. The `h`-collision
is unchanged; only `g` replaces the BGK `rp·(eq_g−g)` with a moment-space relaxation
`M⁻¹·S·M·(eq_g−g)` per site (`k_mrt_relax_g`). The MRT **equilibrium** also differs
from BGK — `eq_g` flips the sign of the `mu·∂c_k + f_proj` term — so there is a
dedicated `k_equilibria_mrt`. `M` / `M⁻¹` (the D3Q19 mass matrices) are copied from
the CPU `EquilibriumDistributionMRT` statics into `__constant__` memory at init.
The moment relaxation runs in double regardless of `real_t` for accuracy.

## Particle tracking (ported, host-side)

The driver builds the CPU `ParticleManager` on the full `Domain` and advects tracers
on the host. Each step the GPU velocity (`d_ux/d_uy/d_uz`, compressed by the
subdomain) is downloaded and scattered into the global velocity arrays via
`SubDomain::local_to_global_index()`, then `pm.update()` runs. Output honours
`particles_file_skip` / `particles_format`. This reuses the validated CPU advection
and interpolation; promoting it to a device kernel is a future optimisation only if
the per-step velocity copy becomes a bottleneck.

## Order-parameter mass correction (ported)

`correct_op_mass = true` removes the scheme's intrinsic order-parameter drift each
step (CPU `MassConservationCorrector`). After the step: reduce `M = Σ c_i` and
`W = Σ c_i(1−c_i)` over streamed sites (`k_mass_weight`, block reduction +
`atomicAdd`), then inject `δc_i = −λ φ_i` with `λ = (M−M₀)/W` via
`h_k[i] += w_k δc_i` (`k_inject_mass`) — restores the total exactly, interface-
localized, adds no momentum. `M₀` is recorded once after the IC upload
(`record_target_mass`). The reduction uses double `atomicAdd`, so the sum order (and
hence the injected correction) differs from the CPU's serial sum at the ~1e-12 level
— immaterial for a drift-removal nudge. Requires compute capability ≥ 6.0 (double
atomics); the default `-DCMAKE_CUDA_ARCHITECTURES=80` satisfies this.

## Not ported yet (guards emit warnings)

- `use_open_bnd` → periodic/body-force only (add `OpenBoundaryOperator` + BC field enforcement).
