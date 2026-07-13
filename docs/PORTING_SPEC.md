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

## Memory (per fluid site, doubles)

**CSR path (`*_matrix_free = false`):** distributions 4·Q, per-direction temporaries
~13·Q, ~30 scalar fields, plus the stored CSR operators. Roughly ~5.6 KB/site —
matches the "naive port" estimate in `../felbm_local/docs/gpu_and_coolbm_notes.md`.

**Matrix-free path (`*_matrix_free = true`):** the stored operators become compact
index tables (~0.7 KB/site total) → ~3.4 KB/site double, ~2.1 KB/site with
`-DFELBM_SINGLE`. The remaining bulk is the ~13·Q per-direction temporaries; **fusion**
removes those, → ~1.7 KB/site double / ~1 KB/site single.

**Host memory.** In the matrix-free / fused path the CSR field operators
(`FieldOperatorGPU`) are *not* built — they were the dominant host cost (~100 GB at
300³) and are only constructed inside the `!mf_grad` branch. The driver also calls
`malloc_trim(0)` after `init()` so the transient build buffers (streaming CSR, mf
table staging) are returned to the OS rather than retained by glibc. Steady-state
host RES then reflects the domain masks + tracers, not the init peak.

**GPU memory (300³).** A 0.37-porosity 300³ (~10 M fluid sites) fully fused runs in
~12 GB in single precision (fits a 24 GB card comfortably). Double does not fit: the
4·Vn distribution arrays (h/g/h2/g2) plus the int tables dominate. To reach 300³ in
double, shrink `d_relax` (Vn→n) and drop the h2/g2 streaming ping-pong (in-place).

## Matrix-free operators (done, validated)

Two cfg keys replace the stored CSR SpMV operators with compact table + on-the-fly
kernels. All reproduce the CPU operators to machine precision (validate with
`compare_cpu_gpu`; args 6/7 toggle them):

- `stream_matrix_free = true` — streaming via a source-code table (`k_stream_gather`):
  one int `src[j]` per distribution index (`>=0` gather, `-1` corner `0.5/0.5`
  average, `-2` empty). ~76 B/site vs ~300 B/site CSR. (Cheap operator; ~neutral
  speed — streaming was never the bottleneck.)
- `grad_matrix_free = true` — **all six field operators**: `grad_cd`/`grad_bd`
  (column-pair / case tables, `k_grad_cd_mf`/`k_grad_bd_mf`), `cd_dir`/`bd_dir`
  (reuse the gradient tables, `k_cd_dir_mf`/`k_bd_dir_mf`), `avg_dir` (`k_avg_dir_mf`),
  and the node-centred `lap` with its 2n `[field,bnd]` input (`k_lap_mf`). Weights are
  recomputed from the D3Q19 constants; near-wall central/backward/forward/3-point
  cases are encoded exactly. Also eliminates the ~22 GB/run of dir-buffer memsets.

**Measured (A5000, 150³ porous, double):** ~19.6 -> **36.96 MLUPS (1.88x)**, exact
(max|Δ| ~5e-16). Per-site memory ~5.6 -> ~3.4 KB (double); with `-DFELBM_SINGLE`
~2.1 KB, so a 300³ porous run (~10 M fluid sites) ~= 20 GB — fits a 24 GB card, which
the naive CSR port did not.

## Kernel fusion (done, validated)

Two further cfg keys collapse the per-direction temporaries into register
computation. Both validate to machine precision (`compare_cpu_gpu`, args 8/9):

- `fused = true` (step 1) — recompute the directional derivatives
  (`cd_dir`/`bd_dir`/`avg_dir`) inside the equilibria + force kernels
  (`k_*_fused`), eliminating 5 Q*n temporaries (`gc_cd, gp_cd, gc_bd, gp_bd, avg`).
- `fuse_collision = true` (step 2, implies `fused`) — one per-site kernel
  (`k_collide_fused_bgk`/`_mrt`) does equilibria + force + collision + apply, so
  `eq_h/eq_g/force_h/force_g/coll_h/coll_g` (6 Q*n) are never materialised. The `h`
  update collapses to `eq_h + force_h`; the MRT `g` path holds the 19 `coll_g` in a
  register array and applies `M^-1 S M` in place (folding the old `k_mrt_relax_g`).

**Measured (A5000, 150³ porous, double):** 36.96 -> 38.2 (step 1) -> **45.8 MLUPS**
(step 2) = **2.34x over the CSR baseline**, exact (max|Δ| ~5e-16) for BGK+MRT,
spheres+fluid. Per-site memory drops another ~209 vals/site to **~1.7 KB (double)**,
so a **300³ porous run (~17 GB) now fits a 24 GB card in double precision**.

## Optimisation roadmap (next)

The operator + fusion work has captured the memory-bandwidth wins. Remaining ideas,
lower priority:

1. **Shrink `d_relax` to n** (one value per site, not Q*n) and consider in-place
   streaming to drop the `d_h2/d_g2` ping-pong — squeezes memory further toward
   ~1 KB/site.
2. **Multi-GPU / domain decomposition** for domains beyond a single card.
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
