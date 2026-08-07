#!/usr/bin/env bash
#
# extend_run.sh -- continue a finished felbm_gpu run from its last checkpoint.
#
#   ./extend_run.sh RUN_DIR TOTAL_STEPS [GPU] [CPUSET]
#   DRY_RUN=1 ./extend_run.sh RUN_DIR TOTAL_STEPS      # show the plan only
#
#   ./extend_run.sh sweep3d/ca1e-1 360000 5 42-47
#
# WHY THIS EXISTS
# ---------------
# felbm_gpu opens log.txt, series.txt and stretching.txt with the default
# ofstream mode, i.e. TRUNCATE. Relaunching in the same directory therefore
# destroys the first leg's history. It is easy to miss: a restart that runs the
# same number of steps produces a file with the same number of ROWS, so nothing
# looks wrong until you plot it and the early times are missing.
#
# On top of that, stretching.txt renumbers from 1 on a restart (it writes
# t - restart_step) while series.txt keeps ABSOLUTE iterations. Splicing the two
# legs therefore needs an offset on one file and not the other.
#
# This script does both: preserves leg 1 under out/leg<N>_*.txt, runs the
# extension, then writes a spliced out/stretching_full.txt with absolute steps.
# The raw per-leg files are never modified, so the splice can be redone.
set -euo pipefail

R=${1:?usage: extend_run.sh RUN_DIR TOTAL_STEPS [GPU] [CPUSET]}
TOTAL=${2:?usage: extend_run.sh RUN_DIR TOTAL_STEPS [GPU] [CPUSET]}
GPU=${3:-0}
CPUSET=${4:-}
BIN=${BIN:-$PWD/build/felbm_gpu}
DRY_RUN=${DRY_RUN:-0}

[ -d "$R/out" ] || { echo "no $R/out" >&2; exit 1; }

# Last checkpoint = highest step number, which is what we resume from.
CK=$(ls -1 "$R/out"/checkpoint_*.h5 2>/dev/null \
     | sed 's/.*checkpoint_\([0-9]*\)\.h5/\1 &/' | sort -n | tail -1 | cut -d' ' -f2- || true)
[ -n "$CK" ] || { echo "no checkpoint in $R/out -- cannot extend, only re-run" >&2; exit 1; }
# NOT `tr -dc 0-9`: that also eats the 5 out of ".h5" and turns 1200 into 12005.
STEP=$(basename "$CK" | sed 's/^checkpoint_\([0-9]*\)\.h5$/\1/')

[ "$TOTAL" -gt "$STEP" ] || { echo "TOTAL_STEPS=$TOTAL is not beyond checkpoint $STEP" >&2; exit 1; }

# Leg number: how many legs are already preserved, plus the one about to move.
# Counted with a glob loop, not `ls | wc -l`: under `set -o pipefail` a glob that
# matches nothing makes ls exit 2, which fails the pipeline and aborts the script.
N=1
for f in "$R"/out/leg*_stretching.txt; do [ -e "$f" ] && N=$((N+1)); done

echo "extend_run: $R"
echo "  resume from : $(basename "$CK")  (step $STEP)"
echo "  extend to   : $TOTAL steps  (+$((TOTAL-STEP)) from the checkpoint)"
echo "  preserving  : out/leg${N}_{log,series,stretching}.txt"
echo "  gpu $GPU${CPUSET:+, cpus $CPUSET}"
[ "$DRY_RUN" = 1 ] && { echo "(dry run -- nothing moved, nothing launched)"; exit 0; }

for f in log series stretching; do
  [ -e "$R/out/$f.txt" ] && cp -p "$R/out/$f.txt" "$R/out/leg${N}_$f.txt"
done

cp -p "$R/settings.cfg" "$R/settings_leg${N}.cfg"
sed -i -e "s|^max_iterations.*|max_iterations  = $TOTAL|" \
       -e "s|^#* *restart_file.*|restart_file    = ./out/$(basename "$CK")|" "$R/settings.cfg"
grep -q "^restart_file" "$R/settings.cfg" || echo "restart_file    = ./out/$(basename "$CK")" >> "$R/settings.cfg"

( cd "$R" && CUDA_VISIBLE_DEVICES=$GPU OMP_PROC_BIND=close OMP_PLACES=cores \
    ${CPUSET:+taskset -c "$CPUSET"} "$BIN" settings.cfg > "run_leg$((N+1)).log" 2>&1 )

# Splice every leg into out/stretching_full.txt with absolute steps.
# Delegated to splice_legs.py rather than done inline: the inline version offset
# only the FINAL leg, which is right for one extension and silently wrong for
# two or more, because every leg after the first also renumbers from 1. It also
# could not cope with a leg re-run over the same span after being killed.
"$(cd "$(dirname "$0")" && pwd)/splice_legs.py" "$R"
