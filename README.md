# felbm_gpu

A **separate, GPU-only** CUDA port of the felbm multiphase (Lee–Liu free-energy)
solver. It reuses the `felbm_local` host code for everything that is not
performance-critical (config parsing, geometry/TIFF init, the multiphase initial
condition, and the sparse stencil operators) and replaces the compute engine with
CUDA kernels. Single GPU, one subdomain, `felbm_gpu` binary only.

> **Status: v0 — first cut, not yet validated on hardware.**
> This scaffold was written without a GPU/`nvcc` available, so it has **not been
> compiled or run**. Treat it as a reviewable, buildable-on-your-box starting
> point, not a finished solver. The intended workflow is: build on the GPU
> machine, fix any compile issues, then validate against the CPU reference (see
> "Validation plan").

## Design: correct first, fast later

The CPU code precomputes every **stencil** — streaming, gradients, Laplacian, the
per-direction operators — as a sparse CSR matrix and applies it as a matrix–vector
product each step. felbm_gpu **uploads those exact CSR matrices** and applies them
with a device SpMV, so the stencils — including all the halfway bounce-back
near-wall terms, which are the easy thing to get subtly wrong — are **bit-for-bit
the CPU operators**. The CSR arrays are reached via thin `felbm_gpu` subclasses
(`operator_access.h`) that expose the operators' `protected` members, so **no edit
to `felbm_local` is required** — it builds against an unmodified checkout.

The genuinely new GPU code is therefore only the **pointwise, embarrassingly
parallel physics** (moments, chemical potential, equilibria, force term, collision
combine), ported kernel-by-kernel in `include/felbm_gpu/device_engine.cuh` with the
CPU source function named above each kernel.

This is the "naive but provably correct" port. It uses more memory than necessary
(stores the operators + many field temporaries) and SpMV is not the fastest GPU
pattern. **Once validated**, the roadmap is to replace the SpMV operators with
matrix-free stencil kernels and fuse passes — see `docs/PORTING_SPEC.md`.

## What is and isn't ported

| Ported (v0) | Not yet ported |
|---|---|
| BGK multiphase collision, Guo forcing | MRT collision (`use_mrt`) |
| Body-force / fully periodic runs | Open inlet/outlet boundaries (`use_open_bnd`) |
| Streaming + halfway bounce-back (CPU CSR) | Order-parameter mass correction (`correct_op_mass`) |
| Gradients / Laplacian / dir operators (CPU CSR) | Particle tracking (host subsystem) |
| Compile-time `double` (default) / `float` (`-DFELBM_SINGLE`) | Multi-GPU / domain decomposition |
| Minimal HDF5 field dump | XDMF metadata (compressed-index → xyz) |

The GRL dispersion runs are body-force-driven and periodic, so they fall in the
**ported** regime — except particle tracking (D⊥), which for now stays on the CPU
build or is added to the GPU path later.

## Build

Requires the CUDA toolkit (`nvcc`), plus HDF5 (C++) and libtiff — the same
libraries as `felbm_local`.

```bash
cd felbm_gpu
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80     # 80=A100, 86=RTX30, 89=RTX40, 90=H100
cmake --build . -j
```

Single precision (≈2× memory/throughput, larger domains, but changes numerics):

```bash
cmake .. -DFELBM_SINGLE=ON
```

`FELBM_LOCAL` defaults to `../felbm_local`; override with `-DFELBM_LOCAL=/path`.

## Run

```bash
cd bin && mkdir -p out
./felbm_gpu ../felbm_local/bin/settings.cfg      # model = multi_phase, use_open_bnd = false
```

Output: `<output_dir>/<output_name>_<iter>.h5` with compressed-site datasets
`density, concentration, u_x, u_y, u_z, pressure` (length = number of fluid sites).

## Validation plan (do this before trusting results)

Validate the GPU against the CPU reference on the same cfg, in order:

1. **One step, all fields.** Dump `c, rho, u, p, mu` after a single step from both
   builds on a small periodic droplet; compare max abs diff (expect ~1e-12 in
   double — the operators are identical, so only the pointwise kernels can differ).
2. **Static droplet / Laplace law.** Spurious currents bounded, mass conserved,
   Δp ≈ 2σ/R — the same checks as `felbm_local` `test_mp_droplet`.
3. **Density-ratio sweep** at ratios 20 and 100 for stability.
4. **A GRL case** vs the CPU for a few thousand steps; compare bulk statistics.

Kernel-by-kernel diffing (step 1) localises any port bug immediately, because the
stencils are shared.

## Layout

```
include/felbm_gpu/
  precision.h        real_t compile-time flag (double default / float)
  gpu_common.cuh     CUDA error checks, launch config, typed alloc/copy
  d3q19.cuh          D3Q19 constants (constant memory), matches the CPU set
  device_csr.cuh     device CSR + SpMV (single- and triple-value)
  operator_access.h  subclasses exposing the CPU operators' CSR (no felbm_local edit)
  device_engine.cuh  the pointwise CUDA kernels (moments, equilibria, force, ...)
src/felbm_gpu_main.cu  driver: reuse host setup, upload operators, run loop, output
docs/PORTING_SPEC.md   CPU→GPU mapping, formulas, and the optimisation roadmap
```
