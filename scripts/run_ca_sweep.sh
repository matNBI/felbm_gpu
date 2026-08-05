#!/usr/bin/env bash
#
# run_ca_sweep.sh -- launch a lambda(Ca) sweep for the 3D analogue of Fig. 5.
#
#   ./run_ca_sweep.sh GEOM_DIR OUT_DIR [BIN]        # launch
#   DRY_RUN=1 ./run_ca_sweep.sh GEOM_DIR OUT_DIR    # print the plan only
#   ONLY=ca1e-2,ca3e-3 ./run_ca_sweep.sh ...        # run just these points
#   GPU_BASE=3 ./run_ca_sweep.sh ...                # lanes on GPUs 3,4,5 (not 0,1,2)
#   CPU_BASE=24 CORES=24 ./run_ca_sweep.sh ...      # and confine to cores 24..47
#
# Use GPU_BASE/CPU_BASE to run alongside another job: setting CUDA_VISIBLE_DEVICES
# in the environment does NOT work, because each lane sets it itself and that
# overrides the outer value.
#
# GEOM_DIR needs image/ and domain.cfg from voxelize_spheres.py.
# BIN defaults to ./build/felbm_gpu -- make sure that is the SINGLE-precision
# build (-DFELBM_SINGLE=ON): measured 36% faster than double on this workload,
# and validate.sh certifies it at 1.4e-7..2.9e-7 against the CPU double reference.
#
# ---------------------------------------------------------------------------
# Why this is not simply "one point per GPU"
# ---------------------------------------------------------------------------
# The run is HOST-bound, not GPU-bound. Per step the host does a scatter over
# every fluid site plus the tracer update, and on a measured sweep those took
# 5.8 + 6.4 ms at 64 threads against an ~8 ms GPU step. So the limiting resource
# is PHYSICAL CORES, not cards, and launching 8 jobs on a 48-core box just
# oversubscribes it. Measured scaling of ms/step against OpenMP threads:
#
#     4 -> 74.24    8 -> 41.19    16 -> 25.72    32 -> 17.23    64 -> 13.48
#
# Sublinear, so more concurrency raises aggregate throughput but slows each job.
# What matters is MAKESPAN, and that is set by the slowest point: t_a ~ 1/Ca, so
# Ca = 1e-3 is 9e6 steps against 9e4 for Ca = 1e-1 -- a hundredfold spread. An
# even split therefore wastes cores on jobs that finish in an hour while the long
# one crawls. Instead the points are grouped into LANES sized by their cost, and
# the cheap points run SEQUENTIALLY inside one lane.
#
# Lanes are pinned with taskset to whole sockets where possible. On a 2-socket
# EPYC the scatter is a random-access write, so a job spanning both sockets pays
# a NUMA penalty that can undo the benefit of the extra cores. Only one CPU per
# physical core is used -- SMT siblings share L1/L2/TLB, which is exactly what a
# scattered write is limited by, and 2x oversubscription measured ~10% WORSE.
#
# ---------------------------------------------------------------------------
# Tracers are required
# ---------------------------------------------------------------------------
# Fig. 5 is lambda(Ca) from Eq. (12), an average of rho_hat^T J rho_hat over the
# ensemble -- there is no lambda without particle tracking. (The cluster-size
# and interface-area diagnostics need only the field dumps, so THOSE could run
# with particles_enable = false and saturate all 8 cards; this sweep cannot.)
#
# particles_number is deliberately the SAME at every point so the statistical
# error on lambda is comparable across the sweep. 20k rather than 50k: at low
# thread counts pm_update dominates the scatter about 2:1, so it is the bigger
# lever, and lambda's limitation is the 1/t convergence bias (docs section 3.5b),
# not ensemble noise.
#
# ---------------------------------------------------------------------------
# Ca is DIAGNOSED, not imposed
# ---------------------------------------------------------------------------
# The accelerations below are SEEDS, not targets. Ca is diagnosed, and the curve
# only needs a good SPREAD of Ca -- hitting round numbers buys nothing. Measured
# in 2D, the checkerboard IC shifts mobility relative to these slab-era values by
# 1.10x at Ca ~ 0.09, 0.87x at 0.026 and 0.73x at 0.010: a fragmented state costs
# mobility, and more so as the forcing weakens. Expect the realised Ca to land
# below the label at the low end, and simply plot lambda against what you measure.
#
# The one real consequence is RUN LENGTH, since t_a ~ 1/Ca: a point whose Ca comes
# in at 0.7x its label has a t_a 1.4x longer, so a fixed step count buys fewer
# advective times than intended. The step counts below are therefore sized for
# 10 t_a NOMINAL, which still delivers ~7 t_a at 0.7x and ~5 at 0.5x. In 3D the
# realised Ca lands at 0.80-0.92x its label -- much milder than 2D's 0.29-0.72x,
# because a 3D pore space traps far less -- so 10 t_a nominal buys 8-9 real ones.
#
# ---------------------------------------------------------------------------
# 8-9 t_a is probably NOT enough, and how much is enough is not yet known
# ---------------------------------------------------------------------------
# An earlier version of this header claimed lambda plateaus by 3-4 t_a with the
# checkerboard IC. That was measured in 2D and does not transfer: at Ca = 9.2e-3
# the 3D lambda was still falling at 6 t_a (per-t_a bins 0.574, 0.545, 0.530,
# 0.517) where 2D had been flat since 4. Fitting lambda_inf + A/t over t > 2, 3
# and 4 gives 0.439, 0.467, 0.461 -- stable, so it converges to a NONZERO value
# rather than the slab IC's decay toward zero, and lambda_3D ~ 0.46 +/- 0.01.
# Report these with that fit and quote its stability across windows as the
# error; do not average the last 3 t_a as the 2D protocol does.
#
# The 2D campaign also found a discrepancy against the paper at high Ca that is
# NOT a run-length effect (see run_fig5_2d.sh) -- our morphology stays finely
# fragmented where theirs coarsens. Nothing rules that out here, and 3D has no
# published lambda(Ca) curve to catch it. Treat these numbers as provisional
# until the 2D morphology question is settled.
#
# 15 t_a was the OLD sizing, from a protocol that extrapolated lambda_inf + A/t
# to undo the slab IC's decay toward zero; see docs/HANDOFF_stretching.md
# section 0. That extrapolation is back in use for 3D, for a different reason.
#
# As in Linga et al. Table S3, whose Ca column is "resulting from different
# driving forces". The accelerations below come from one measured point
# (accel 3.0e-3 -> Ca 0.18) extrapolated by Darcy linearity. That linearity WILL
# break at low Ca once capillary trapping cuts the mobility, so read the realised
# value back from each run and relabel:
#
#     Ca = <u_z> * nu / gamma        <u_z> = column 5 of out/series.txt
#     with nu = (tau-0.5)/3 = 0.13333 and gamma = surface_tension = 0.004444
#
# Early-time agreement means nothing: at Ca = 1e-3 one advective time is 6e5
# steps, so the first few thousand steps are the viscous transient, before any
# trapping. Read Ca at the END.
set -euo pipefail

GEOM=${1:?usage: run_ca_sweep.sh GEOM_DIR OUT_DIR [BIN]}
OUT=${2:?usage: run_ca_sweep.sh GEOM_DIR OUT_DIR [BIN]}
BIN=${3:-$PWD/build/felbm_gpu}
TPL="$(cd "$(dirname "$0")" && pwd)/ca_sweep_templates"
DRY_RUN=${DRY_RUN:-0}
NTRACER=${NTRACER:-20000}
ONLY=${ONLY:-}      # comma-separated whitelist of point labels; empty = all
GPU_BASE=${GPU_BASE:-0}   # first GPU to use; lanes take GPU_BASE, +1, +2
CPU_BASE=${CPU_BASE:-0}   # index into the physical-core list where lane A starts

# --- physical cores, one CPU id per (socket,core); ignores SMT siblings -------
# Order SOCKET-major so the first lane occupies a whole socket: on 2x24 that
# puts lane A entirely on socket 0. Sorting by CPU id instead would only do so
# by accident of the kernel's numbering.
mapfile -t PCPU < <(lscpu -p=CPU,Core,Socket 2>/dev/null | grep -v '^#' \
                    | awk -F, '!seen[$3","$2]++ {print $3","$2","$1}' \
                    | sort -t, -k1,1n -k2,2n | cut -d, -f3)
NCORE=${CORES:-${#PCPU[@]}}
[ "$NCORE" -gt 0 ] || { echo "could not detect physical cores; set CORES=" >&2; exit 1; }
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); [ "$NGPU" -gt 0 ] || NGPU=1

# --- lanes: label:accel:steps:fskip, space-separated points run SEQUENTIALLY --
# Cores are split 1/2 : 1/4 : 1/4, which on a 2x24 box is 24/12/12 -- lane A
# takes one whole socket. Giving lane A more (32/8/8) is faster on paper, 43 h
# against 51 h, but makes it span both sockets, and the NUMA penalty on a random
# scatter can exceed that. Edit cA/cB/cC below to change it.
LANE_A="ca1e-3:1.667e-05:6000000:600000"
LANE_B="ca3e-3:5.000e-05:2000000:200000"
LANE_C="ca1e-2:1.667e-04:600000:60000 ca3e-2:5.000e-04:200000:20000 ca1e-1:1.667e-03:60000:6000"

cA=$(( NCORE / 2 )); cB=$(( NCORE / 4 )); cC=$(( NCORE - cA - cB ))
[ $(( CPU_BASE + NCORE )) -le ${#PCPU[@]} ] || {
  echo "CPU_BASE=$CPU_BASE + CORES=$NCORE exceeds the ${#PCPU[@]} physical cores available" >&2; exit 1; }
echo "run_ca_sweep: $NCORE physical cores, $NGPU GPUs, $NTRACER tracers/point"
echo "  lane A (GPU $(( (GPU_BASE+0) % NGPU )), $cA cores): $LANE_A"
echo "  lane B (GPU $(( (GPU_BASE+1) % NGPU )), $cB cores): $LANE_B"
echo "  lane C (GPU $(( (GPU_BASE+2) % NGPU )), $cC cores): $LANE_C"
echo "  binary: $BIN"

cpus_for() {  # $1 = start index, $2 = count -> comma list of physical CPU ids
  local s=$1 n=$2 out=() i
  for ((i=s;i<s+n;i++)); do out+=("${PCPU[$i]}"); done
  ( IFS=,; echo "${out[*]}" )
}

run_lane() {   # $1 = lane spec, $2 = gpu, $3 = cores, $4 = first core index
  local spec=$1 gpu=$2 ncore=$3 base=$4 cpus; cpus=$(cpus_for "$base" "$ncore")
  for pt in $spec; do
    IFS=: read -r label accel iters fskip <<<"$pt"
    if [ -n "$ONLY" ]; then
      case ",$ONLY," in *",$label,"*) ;; *) echo "  skip $label (not in ONLY)"; continue;; esac
    fi
    local R="$OUT/$label"
    # Skip only COMPLETED points. Testing for the mere existence of output
    # strands an interrupted run: it is skipped on the next invocation and never
    # finishes. felbm_gpu writes "felbm_gpu: done." as its last log line, so use
    # that as the completion marker and re-run anything partial.
    if grep -q "felbm_gpu: done" "$R/out/log.txt" 2>/dev/null; then
      echo "  SKIP $label (complete)"; continue
    fi
    if [ -e "$R/out/log.txt" ]; then
      echo "  REDO $label (partial output found -- restarting from scratch)"; rm -rf "$R/out"
    fi
    mkdir -p "$R/out"
    cp -r "$GEOM/image" "$R/image"; cp "$GEOM/domain.cfg" "$R/domain.cfg"
    sed -i 's|^image_dir *=.*|image_dir         = ./image/|' "$R/domain.cfg"
    cp "$TPL/params.cfg" "$R/params.cfg"; cp "$TPL/fluid.cfg" "$R/fluid.cfg"
    # fskip is one t_a, so checkpoint every 5 t_a. Enough granularity to extend a
    # run to a longer averaging window without re-running it from step 0.
    local ckpt=$((fskip*5))
    sed -e "s|@ACCEL@|$accel|" -e "s|@ITERS@|$iters|" -e "s|@FSKIP@|$fskip|" \
        -e "s|@CKPT@|$ckpt|" \
        -e "s|@THREADS@|$ncore|" -e "s|@NTRACER@|$NTRACER|" \
        "$TPL/settings.cfg" > "$R/settings.cfg"
    echo "  [$label] gpu=$gpu cpus=$cpus steps=$iters accel=$accel"
    [ "$DRY_RUN" = 1 ] && continue
    ( cd "$R" && CUDA_VISIBLE_DEVICES=$gpu OMP_PROC_BIND=close OMP_PLACES=cores \
        taskset -c "$cpus" "$BIN" settings.cfg > run.log 2>&1 )
  done
}

if [ "$DRY_RUN" = 1 ]; then echo "(dry run -- nothing launched)"; fi
run_lane "$LANE_A" $(( (GPU_BASE+0) % NGPU )) "$cA" $(( CPU_BASE ))            &
run_lane "$LANE_B" $(( (GPU_BASE+1) % NGPU )) "$cB" $(( CPU_BASE + cA ))      &
run_lane "$LANE_C" $(( (GPU_BASE+2) % NGPU )) "$cC" $(( CPU_BASE + cA + cB )) &
wait
echo "run_ca_sweep: done.  Now read the REALISED Ca from each out/series.txt"
echo "  and analyse with scripts/cluster_sizes.py on the last field dumps."
