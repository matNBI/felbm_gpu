#!/usr/bin/env bash
#
# prune_runs.sh -- reclaim space from finished sweep points, safely.
#
#   ./prune_runs.sh SWEEP_DIR                 # REPORT ONLY (default)
#   DRY_RUN=0 ./prune_runs.sh SWEEP_DIR       # actually delete
#   KEEP=2 ./prune_runs.sh SWEEP_DIR          # keep the last 2 checkpoints
#   DUMPS=1 ./prune_runs.sh SWEEP_DIR         # also thin field dumps (see below)
#
# Dry run is the DEFAULT. You must pass DRY_RUN=0 to remove anything.
#
# WHAT IT TOUCHES
# ---------------
# Only `checkpoint_<step>.h5` and its paired `particles_<step>.h5`, keeping the
# newest KEEP of them. Those are the bulk: 678 MB per checkpoint in 3D, 482 MB in
# 2D, and a point checkpointed every 2 t_a accumulates dozens.
#
# WHAT IT NEVER TOUCHES
# ---------------------
#   series.txt  stretching.txt  stretching_full.txt  leg*_*.txt  log.txt
#     -- kilobytes to tens of MB, and IRREPLACEABLE: they are the actual results.
#   geometry.h5
#     -- needed by interface_length.py and cluster_sizes.py to rebuild the grid.
#   the newest checkpoint + its particles file
#     -- extend_run.sh resumes from exactly this pair. Delete it and the point
#        can only be re-run from step 0.
#
# Field dumps (`output_<step>.h5`) are left alone unless DUMPS=1, because they
# are what sd(t) is computed from -- the diagnostic that localised the whole
# high-Ca discrepancy. With DUMPS=1 every other dump is removed, halving the
# space while keeping the trajectory readable.
#
# SAFETY: a directory is SKIPPED if any felbm_gpu process has it open. Pruning a
# point that is mid-extension would delete the checkpoint its own restart depends
# on. This check is why the script exists rather than a find -delete one-liner.
set -euo pipefail

SWEEP=${1:?usage: prune_runs.sh SWEEP_DIR}
DRY_RUN=${DRY_RUN:-1}
KEEP=${KEEP:-1}
DUMPS=${DUMPS:-0}

human() { numfmt --to=iec --suffix=B "${1:-0}" 2>/dev/null || echo "${1} bytes"; }

total=0
for R in "$SWEEP"/*/; do
  [ -d "$R/out" ] || continue
  label=$(basename "$R")

  # In use? Match a felbm_gpu whose cwd is this directory.
  busy=""
  for pid in $(pgrep -x felbm_gpu 2>/dev/null || true); do
    cwd=$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)
    [ "$cwd" = "$(readlink -f "$R")" ] && busy="pid $pid"
  done
  if [ -n "$busy" ]; then
    echo "  SKIP $label -- IN USE ($busy)"
    continue
  fi

  mapfile -t cks < <(ls -1 "$R/out"/checkpoint_*.h5 2>/dev/null \
                     | sed 's/.*checkpoint_\([0-9]*\)\.h5/\1 &/' | sort -n | cut -d' ' -f2-)
  n=${#cks[@]}
  if [ "$n" -le "$KEEP" ]; then
    echo "  keep $label -- $n checkpoint(s), at or under KEEP=$KEEP"
    continue
  fi

  drop=$(( n - KEEP ))
  bytes=0
  for ((k=0; k<drop; k++)); do
    f=${cks[$k]}
    step=$(basename "$f" | sed 's/^checkpoint_\([0-9]*\)\.h5$/\1/')
    pf="$R/out/particles_${step}.h5"
    for x in "$f" "$pf"; do
      [ -f "$x" ] || continue
      sz=$(stat -c %s "$x"); bytes=$(( bytes + sz ))
      [ "$DRY_RUN" = 0 ] && rm -f "$x"
    done
  done

  if [ "$DUMPS" = 1 ]; then
    mapfile -t ds < <(ls -1 "$R/out"/output_*.h5 2>/dev/null \
                      | sed 's/.*output_\([0-9]*\)\.h5/\1 &/' | sort -n | cut -d' ' -f2-)
    for ((k=1; k<${#ds[@]}-1; k+=2)); do      # keep first, last, and every other
      sz=$(stat -c %s "${ds[$k]}"); bytes=$(( bytes + sz ))
      [ "$DRY_RUN" = 0 ] && rm -f "${ds[$k]}"
    done
  fi

  total=$(( total + bytes ))
  printf "  %-12s %2d checkpoints -> keep %d, free %s\n" "$label" "$n" "$KEEP" "$(human $bytes)"
done

echo
echo "  TOTAL reclaimable: $(human $total)"
[ "$DRY_RUN" = 1 ] && echo "  (dry run -- nothing deleted; re-run with DRY_RUN=0)"
