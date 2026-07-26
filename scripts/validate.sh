#!/usr/bin/env bash
# Standard validation matrix for felbm_gpu.
#
# Runs compare_cpu_gpu across the operator paths and collision models that
# matter, on both an all-fluid and an obstacle geometry. Every case drives the
# SAME MultiPhaseGPU the app uses, from the same initial distributions as the CPU
# EngineMultiPhase, and reports max|delta| on the h/g distributions -- the
# fundamental state, so if these match, everything derived from them matches.
#
#   scripts/validate.sh [path-to-build]      (default: ./build)
#
# Pass criteria:
#   double build (default)      max|delta| ~ 1e-15 or smaller
#   -DFELBM_SINGLE=ON           max|delta| ~ 1e-6  (pure float rounding)
#
# arg order: steps N ratio geom coll mf mfg fused fusecoll mrtfast inplace
set -u
BUILD="${1:-./build}"
BIN="$BUILD/compare_cpu_gpu"
[ -x "$BIN" ] || { echo "no compare_cpu_gpu in $BUILD (build first)"; exit 1; }

run() {                       # run <label> <args...>
  local label="$1"; shift
  local raw dh dg
  raw=$("$BIN" "$@" 2>&1)
  dh=$(printf '%s' "$raw" | sed -n 's/^  h  *: [^=]*=\([0-9.e+-]*\).*/\1/p' | head -1)
  dg=$(printf '%s' "$raw" | sed -n 's/^  g  *: [^=]*=\([0-9.e+-]*\).*/\1/p' | head -1)
  if [ -z "$dh" ]; then
    printf '%-40s  FAILED (no h/g line)\n' "$label"
  else
    printf '%-40s  max|dh|=%-11s max|dg|=%s\n' "$label" "$dh" "$dg"
  fi
}

echo "=== felbm_gpu validation matrix ($BUILD) ==="
echo "--- CSR reference path (stencils are bit-for-bit the CPU operators) ---"
run "fluid   bgk  csr"                20 48 5 fluid   bgk 0 0 0 0 0 0
run "spheres bgk  csr"                20 48 5 spheres bgk 0 0 0 0 0 0
run "spheres mrt  csr"                20 48 5 spheres mrt 0 0 0 0 0 0
echo "--- matrix-free operators ---"
run "spheres bgk  mf+mfg"             20 48 5 spheres bgk 1 1 0 0 0 0
run "spheres mrt  mf+mfg"             20 48 5 spheres mrt 1 1 0 0 0 0
echo "--- fusion ---"
run "spheres mrt  fused"              20 48 5 spheres mrt 1 1 1 0 0 0
run "spheres bgk  fuse_collision"     20 48 5 spheres bgk 1 1 0 1 0 0
run "spheres mrt  fuse_collision"     20 48 5 spheres mrt 1 1 0 1 0 0
echo "--- mrt_fast_transform + in-place streaming (full production path) ---"
run "spheres mrt  +mrt_fast"          20 48 5 spheres mrt 1 1 0 1 1 0
run "spheres mrt  +mrt_fast +inplace" 20 48 5 spheres mrt 1 1 0 1 1 1
run "fluid   bgk  +inplace"           20 48 5 fluid   bgk 1 1 0 1 0 1
echo
echo "The spheres cases exercise halfway bounce-back, the biased near-wall"
echo "stencils and the node-centred Laplacian with the wetting BC. The all-fluid"
echo "case never touches those, so always check a spheres case before trusting"
echo "porous results."
