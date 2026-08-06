#!/usr/bin/env bash
#
# mobility_scan.sh -- does phase-field mobility control the COARSENING RATE?
#
#   ./mobility_scan.sh SRC_RUN CKPT_STEP OUT_DIR TOTAL_STEPS [MOBILITIES...]
#   DRY_RUN=1 ./mobility_scan.sh ...            # set up, print the plan, launch nothing
#   CALIBRATE=5000 ./mobility_scan.sh ...       # short run per variant to measure ms/step
#
#   ./mobility_scan.sh ~/runs/2d_mixing/ca8.6e-2 14200 ~/runs/2d_mobscan2 280000 0.005 0.08
#
# WHY, AND WHY THE OLD SCAN MISSED IT
# -----------------------------------
# At Ca ~ 0.098 our lambda converges to 0.138 against the paper's 0.058, and our
# specific interface length to 0.753 against their 0.333 -- ratios 2.40 and 2.26,
# equal to 6%. So the estimator is sound and the entire discrepancy is that we
# equilibrate to a state with 2.3x more interface. Duration is ruled out: 149 t_a
# converged (docs/HANDOFF_stretching.md 8b).
#
# docs/MEMO_mobility_sensitivity.md scanned mobility and read null. It judged on
# lambda after ~10 t_a, under the slab IC, where lambda was decaying toward zero
# for an unrelated reason. That could not have detected a coarsening-rate effect.
# This scan judges on d(sd)/dt over tens of t_a instead.
#
# WHAT A NULL WOULD MEAN HERE
# ---------------------------
# In the conservative Allen-Cahn update
#
#     dc/dt + div(c u) = div[ M ( grad c - (4/W) c(1-c) n ) ]
#
# BOTH the diffusive and the sharpening term carry M. Rescaling M may therefore
# change how fast the interface network relaxes without moving its equilibrium.
# If the variants trace different rates to the SAME sd, that is not a null
# result -- it is the answer: mobility cannot reach the paper's morphology, and
# the difference is the model class (Cahn-Hilliard coarsens by Ostwald ripening;
# conservative Allen-Cahn is built to suppress exactly that).
#
# So read the trajectories, not the endpoint alone. Divergent rates AND a common
# plateau is the informative outcome.
#
# START POINT
# -----------
# Restart from an EARLY checkpoint (a few t_a), not the converged one: the
# quantity of interest is the rate, and from an already-flat state there is
# nothing left to coarsen. The baseline mobility trajectory does not need
# re-running -- it is the parent run's own sd(t).
#
# Stability: the solver NaNs below mobility_coeff ~ 0.002 within 1000 steps.
# Use CALIBRATE to shake that out cheaply before committing hours.
set -euo pipefail

SRC=${1:?usage: mobility_scan.sh SRC_RUN CKPT_STEP OUT_DIR TOTAL_STEPS [MOBILITIES...]}
CKPT=${2:?need the checkpoint step to branch from}
OUT=${3:?need an output directory}
TOTAL=${4:?need total steps (absolute, i.e. beyond CKPT)}
shift 4
MOBS=("${@:-0.005 0.08}")
[ $# -gt 0 ] || MOBS=(0.005 0.08)

DRY_RUN=${DRY_RUN:-0}
CALIBRATE=${CALIBRATE:-0}
BIN=${BIN:-$PWD/build/felbm_gpu}
GPU=${GPU:-0}
CORES=${CORES:-48}

CK="$SRC/out/checkpoint_${CKPT}.h5"
PF="$SRC/out/particles_${CKPT}.h5"
[ -f "$CK" ] || { echo "no $CK" >&2; exit 1; }
[ -f "$PF" ] || echo "WARNING: $PF missing -- tracers would be re-seeded" >&2

mapfile -t P < <(lscpu -p=CPU,Core,Socket 2>/dev/null | grep -v '^#' \
                 | awk -F, '!seen[$3","$2]++ {print $3","$2","$1}' \
                 | sort -t, -k1,1n -k2,2n | cut -d, -f3)
CPUSET=${CPUSET:-$( IFS=,; echo "${P[*]:0:$CORES}" )}

STEPS=$(( ${CALIBRATE:-0} > 0 ? CKPT + CALIBRATE : TOTAL ))
echo "mobility_scan: branching from $SRC @ step $CKPT"
echo "  variants   : ${MOBS[*]}"
echo "  to step    : $STEPS  (+$((STEPS-CKPT)) from the checkpoint)$( [ "$CALIBRATE" -gt 0 ] && echo '   [CALIBRATION]')"
echo "  gpu $GPU, $CORES cores"
echo "  baseline   : the parent run's own sd(t) -- not re-run here"

for m in "${MOBS[@]}"; do
  R="$OUT/mob$m"
  mkdir -p "$R/out"
  cp "$SRC/domain.cfg" "$SRC/params.cfg" "$SRC/fluid.cfg" "$R/"
  cp "$SRC/settings.cfg" "$R/settings.cfg"
  # symlink, not copy: 482 MB per variant of identical read-only input
  ln -sf "$(readlink -f "$CK")" "$R/out/checkpoint_${CKPT}.h5"
  [ -f "$PF" ] && ln -sf "$(readlink -f "$PF")" "$R/out/particles_${CKPT}.h5"
  sed -i "s|^mobility_coeff.*|mobility_coeff  = $m|" "$R/params.cfg"
  sed -i -e "s|^max_iterations.*|max_iterations  = $STEPS|" \
         -e "s|^#* *restart_file.*|restart_file    = ./out/checkpoint_${CKPT}.h5|" "$R/settings.cfg"
  grep -q "^restart_file" "$R/settings.cfg" \
    || echo "restart_file    = ./out/checkpoint_${CKPT}.h5" >> "$R/settings.cfg"
  echo "  [mob=$m] $R  ($(grep -m1 '^mobility_coeff' "$R/params.cfg" | tr -s ' '))"
done

if [ "$DRY_RUN" = 1 ]; then echo "(dry run -- set up, nothing launched)"; exit 0; fi

for m in "${MOBS[@]}"; do
  R="$OUT/mob$m"
  echo "  launching mob=$m ..."
  ( cd "$R" && CUDA_VISIBLE_DEVICES=$GPU OMP_PROC_BIND=close OMP_PLACES=cores \
      taskset -c "$CPUSET" "$BIN" settings.cfg > run.log 2>&1 ) || {
    echo "  mob=$m FAILED -- see $R/run.log" >&2; continue; }
  t=$(grep -o 'in [0-9.]* s' "$R/out/log.txt" | tail -1)
  echo "    done $t  ->  $(python3 - "$R" "$CKPT" "$STEPS" <<'PY'
import sys,re,os
try:
    s=open(os.path.join(sys.argv[1],"out/log.txt")).read()
    el=float(re.findall(r"in ([0-9.]+) s",s)[-1]); n=int(sys.argv[3])-int(sys.argv[2])
    print(f"{1000*el/n:.2f} ms/step")
except Exception: print("(no timing)")
PY
)"
done
echo "mobility_scan: done. Compare with"
echo "  for d in $OUT/mob*/; do python3 scripts/interface_length.py \$d --d 21; done"
