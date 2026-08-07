#!/usr/bin/env bash
#
# extend_3d.sh -- take the 3D sweep points to convergence, cheapest first.
#
#   ./extend_3d.sh SWEEP_DIR [BIN]
#   DRY_RUN=1 ./extend_3d.sh SWEEP_DIR     # print the plan only
#   ONLY=ca1e-1,ca3e-2 ./extend_3d.sh ...  # just these
#   CPU_BASE=36 GPU_BASE=1 ONLY=ca1e-1 ./extend_3d.sh ...   # keep clear of other jobs
#
# RUN IT DETACHED. This loop can span tens of hours, and in the foreground it
# dies with the ssh session -- the point being extended survives (it is already
# exec'd) but the loop never advances, so the remaining points silently never
# start. Observed exactly that: ca1e-1 completed, ca3e-2 and ca1e-2 had 0 legs.
#
#   setsid nohup ./extend_3d.sh SWEEP_DIR > extend.log 2>&1 < /dev/null &
#
# WHY, AND WHY THE TARGETS DIFFER PER POINT
# -----------------------------------------
# 2D established that the late-time decay of lambda is strongly Ca-DEPENDENT:
#
#     Ca 9.65e-2 : 0.274 (15 t_a) -> 0.1418 (146 t_a)   -49.6%
#     Ca 6.37e-3 : 0.327 (6.4 t_a) -> 0.3008 (31 t_a)    -8.1%
#
# with convergence horizons to match -- flat to -0.01%/t_a by 31 t_a at mid Ca,
# still moving at 146 t_a at high Ca. A single t_a target across the sweep would
# therefore either waste days at low Ca or stop far short at high Ca.
#
# The 3D points other than ca1e-1 sit at ~9 t_a, which the 2D result shows can be
# 50% high. They are upper bounds, and the 3D lambda(Ca) SHAPE cannot be read
# until they are extended -- which is the whole point of the 3D campaign.
#
# Targets below mirror the 2D horizons by Ca. They are not sacred: watch the
# slope reported by the analysis snippet at the end and stop when it is under
# ~0.3%/t_a, which is where 2D's ca1e-2 was when it stopped moving.
#
# ca1e-3 is DELIBERATELY EXCLUDED. At t_a = 759k steps it needs ~11.3M more
# steps, i.e. ~63 h at 24 cores -- more than the other four combined. Run it
# only if the shape of the other four leaves the optimum genuinely ambiguous.
#
# Each point resumes from its last checkpoint via extend_run.sh, so the cost
# below is incremental, not a re-run. Nothing is lost if you stop partway: every
# extension leaves out/stretching_full.txt spliced to that point.
set -euo pipefail

SWEEP=${1:?usage: extend_3d.sh SWEEP_DIR [BIN]}
BIN=${2:-$PWD/build/felbm_gpu}
HERE="$(cd "$(dirname "$0")" && pwd)"
DRY_RUN=${DRY_RUN:-0}
ONLY=${ONLY:-}
GPU_BASE=${GPU_BASE:-0}
CORES=${CORES:-24}
# CPU_BASE is essential when anything else is on the box. Without it every
# invocation allocates from physical-core index 0, so a 3D extension launched
# alongside the 2D sweep (which runs GPU=0 CORES=24 from CPU_BASE=0) lands on
# cores 0..23 as well and the two OpenMP pools fight for the same cores. That
# costs far more than sharing a GPU, because these runs are ~75% host-bound.
# Check with `taskset -cp <pid>` after launching, do not assume.
CPU_BASE=${CPU_BASE:-0}

# label : total steps : approx t_a reached : approx hours at 24 cores
# ordered CHEAPEST FIRST, so the curve fills in from the top of the Ca range
POINTS=(
  "ca1e-1:900943:150:1.7"
  "ca3e-2:1653960:80:8.1"
  "ca1e-2:2645612:40:11.4"
  "ca3e-3:4739808:20:15.2"
)

mapfile -t P < <(lscpu -p=CPU,Core,Socket 2>/dev/null | grep -v '^#' \
                 | awk -F, '!seen[$3","$2]++ {print $3","$2","$1}' \
                 | sort -t, -k1,1n -k2,2n | cut -d, -f3)
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); [ "$NGPU" -gt 0 ] || NGPU=1

echo "extend_3d: $SWEEP   ($CORES cores per point, GPUs from $GPU_BASE)"
i=0
for p in "${POINTS[@]}"; do
  IFS=: read -r label total ta hrs <<<"$p"
  if [ -n "$ONLY" ]; then
    case ",$ONLY," in *",$label,"*) ;; *) continue;; esac
  fi
  R="$SWEEP/$label"
  [ -d "$R/out" ] || { echo "  SKIP $label (no $R/out)"; continue; }
  cur=$(ls -1 "$R/out"/checkpoint_*.h5 2>/dev/null \
        | sed 's/.*checkpoint_\([0-9]*\)\.h5/\1/' | sort -n | tail -1)
  [ -n "$cur" ] || { echo "  SKIP $label (no checkpoint -- cannot extend, only re-run)"; continue; }
  gpu=$(( (GPU_BASE + i) % NGPU ))
  cpus=$( IFS=,; echo "${P[*]:$(( CPU_BASE + i * CORES )):$CORES}" )
  [ $(( CPU_BASE + (i+1) * CORES )) -le ${#P[@]} ] || {
    echo "  CPU_BASE=$CPU_BASE + $((i+1)) x CORES=$CORES exceeds ${#P[@]} physical cores" >&2; exit 1; }
  echo "  [$label] $cur -> $total steps  (~$ta t_a, ~$hrs h)  gpu $gpu"
  [ "$DRY_RUN" = 1 ] && { i=$((i+1)); continue; }
  # Do NOT let one failure abort the rest: under `set -e` a non-zero exit from
  # extend_run.sh would kill the whole loop and silently leave the remaining
  # points untouched. Each point is independent and resumable, so carry on.
  if ! BIN="$BIN" "$HERE/extend_run.sh" "$R" "$total" "$gpu" "$cpus"; then
    echo "  [$label] FAILED -- continuing with the next point" >&2
  fi
  i=$((i+1))
done

[ "$DRY_RUN" = 1 ] && echo "(dry run -- nothing launched)"
cat <<'EOT'

Read the result with (note stretching_full.txt, not stretching.txt):

  for d in */; do python3 - "$d" <<'PY'
import numpy as np, sys, os
NU,GAM,D = 0.13333, 0.004444, 20.0
r = sys.argv[1]
s = np.loadtxt(r+"out/series.txt")
if os.path.exists(r+"out/leg1_series.txt"):
    s = np.vstack([np.loadtxt(r+"out/leg1_series.txt"), s])
f = r+"out/stretching_full.txt"
st = np.loadtxt(f if os.path.exists(f) else r+"out/stretching.txt")
U = np.abs(s[len(s)//2:, 4]).mean(); ta = D/U
t = st[:,0]/ta; lam = st[:,1]*ta; m = t >= t[-1]*0.8
sl = 100*np.polyfit(t[m], lam[m], 1)[0]/lam[m].mean()
print(f"{r:10s} Ca={U*NU/GAM:.3e}  {t[-1]:6.1f} t_a  lambda={lam[m].mean():.4f}  slope={sl:+.2f}%/t_a")
PY
  done

STOP when slope is under ~0.3%/t_a -- that is where 2D's ca1e-2 had stopped
moving. A point still at -3%/t_a is nowhere near converged and its lambda is an
upper bound, not a measurement.
EOT
