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
(length n) to a `Q·n` per-direction field; written from base (no offset).

> **TODO / verify on hardware:** the `avg_dir` row count is `(Q−1)n + n + 1`. The
> driver allocates the target as `Q·n` and indexes `[k*n+i]`. Confirm the CPU
> `compute_average_dir` writes exactly `Q·n` rows into `m_avg_lapl_mu` (base) — if
> the mapping differs, adjust the `d_avg` size / indexing accordingly. This is the
> one operator whose output layout wasn't fully traced from source.

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

## Not ported yet (guards emit warnings)

- `use_mrt` → BGK only (add `CollisionModelMRTMultiPhase`).
- `use_open_bnd` → periodic/body-force only (add `OpenBoundaryOperator` + BC field enforcement).
- `correct_op_mass` → add the global order-parameter mass reduction + rescale.
- Particle tracking → host subsystem; run on the CPU build or port `ParticleManager`.
