#!/usr/bin/env bash
#
# run_fig5_2d.sh -- reproduce Fig. 5 of Linga et al. in 2D, on ONE GPU.
#
#   ./run_fig5_2d.sh OUT_DIR [BIN]
#   DRY_RUN=1 ./run_fig5_2d.sh OUT_DIR          # print the plan only
#
#   GPU=3 CORES=24 ./run_fig5_2d.sh OUT_DIR     # pick card and core budget
#   FROM=ca1e-2 ./run_fig5_2d.sh OUT_DIR        # start at this point, skip earlier ones
#   ONLY=ca1e-2,ca3e-3 ./run_fig5_2d.sh OUT_DIR # run just these
#
# This is the VALIDATION run for the 3D campaign: Fig. 5 is the only published
# lambda(Ca) curve, so reproducing it in 2D is the one place our estimator can be
# checked against a known answer. It needs no external geometry -- the cylinder
# pack is generated internally from domain.cfg.
#
# ---------------------------------------------------------------------------
# Fidelity: paper's box AND paper's interface
# ---------------------------------------------------------------------------
#   domain      60 d x 90 d = 1260 x 1890 at d = 21 lu   (Table S2)
#   porosity    0.62 from the RSA cylinder pack           (Table S2)
#   saturation  s_w = 0.5, slab IC with two smoothed interfaces
#   contact     theta_0 = 60 deg  ->  phi = 6 sigma cos(theta) = 0.012700
#   interface   interface_width = 3.0 lu.  felbm's W and the paper's eps are
#               NOT the same quantity: felbm is c = 0.5(1+tanh(2x/W)) and the
#               paper is phi = tanh(z/(sqrt(2) eps)), so W = 2 sqrt(2) eps.
#               W = 3 at d = 21 gives eps_equiv = 0.0505 d against the paper's
#               0.05 -- a 1% match, not the 3x error a naive comparison gives.
#   Oh = 0.447  surface_tension = nu^2/(Oh^2 d) = 0.004233, matching Table S2's
#               Oh so that Re = Ca/Oh^2 stays below 0.5 across the sweep.
#
# ---------------------------------------------------------------------------
# Cost, measured on an RTX 3090 with 64 cores (1.58M sites, 20k tracers)
# ---------------------------------------------------------------------------
#   9.02 ms/step.  The run is HOST-bound (scatter + pm_update ~ 7 ms of that),
#   so on the server expect ~13 ms at 24 cores, ~17 at 16, ~21 at 12.
#   Five points, 15 t_a each = 14.7e6 steps: 37 h at 64 cores, ~54 h at 24.
#
#   HOST-bound understates it. Re-measured 2026-08-08 on the same box at 32
#   threads: 20k steps in 246.4 s (12.3 ms/step, 128.6 MLUPS), of which the
#   tracking phases are d2h 24.2 + scatter 133.7 + pm_update 87.7 = 245.6 s.
#   That is the ENTIRE wall clock. `particles_overlap` is not hiding the tracer
#   work behind the GPU; the GPU is waiting on it. Tracers are not an overhead
#   on the run, they ARE the run.
#
#   Hence NTRACER defaults to 10000 since 2026-08-08 (was 20000), which should
#   nearly halve wall time everywhere. lambda is an unbiased ensemble mean, so
#   halving N does not shift it -- only its error bar, by sqrt(2). The paper
#   uses 1e4. Points already measured at 20k stay comparable in the MEAN.
#
#   For reference, and so nobody repeats the experiment: this workload needs the
#   GPU. camel (80 Xeon cores, no GPU) ran the same case at under 1.7 steps/s
#   against 81.2 here -- 45x slower. See the memory note on camel.
#
#   t_a = 1.5 d^2 / Ca = 661.5/Ca steps, so the LOWEST Ca is 68% of the total.
#   Points therefore run CHEAPEST FIRST: the high-Ca half of the curve lands in
#   under 4 h and can be sanity-checked before the 25 h point is committed. Kill
#   it after any point and what has completed is still usable.
#
# ---------------------------------------------------------------------------
# Ca is DIAGNOSED, not imposed
# ---------------------------------------------------------------------------
# The accelerations below are SEEDS, not targets: Ca is diagnosed and the curve
# only needs a good SPREAD. With the checkerboard IC the realised Ca came in at
# 1.10x / 0.87x / 0.73x these labels at Ca ~ 0.09 / 0.026 / 0.010 -- a fragmented
# state costs mobility, increasingly so as the forcing weakens. Step counts are
# sized for 10 t_a NOMINAL, giving ~7 t_a at 0.7x.
#
# ---------------------------------------------------------------------------
# WARNING: these step counts are NOT enough, and lengthening them is NOT a fix
# ---------------------------------------------------------------------------
# The paper averages lambda over t in [30 t_a, 60 t_a] (SI, section B.1) -- its
# measurement window STARTS at twice our longest run. An earlier version of this
# header claimed "lambda plateaus by 3-4 t_a with this IC, so that is ample".
# That was inferred from our own high-Ca run going flat at 0.28 from 4 to 15 t_a
# and it is WRONG: their Fig. S4 shows the high-Ca curves still descending at 30.
#
# But simply running longer does not close the gap. Digitising their Fig. S4 and
# overlaying our Ca = 0.085 run gives:
#
#   t/t_a      1      2      3      5      8     12    14.8
#   ours     0.677  0.415  0.314  0.278  0.273  0.283  0.274
#   theirs   0.648  0.429  0.312  0.204  0.138  0.103  0.087
#
# Identical to 3 t_a, then they keep decaying and we flatten. Agreement over the
# first 3 t_a says geometry, IC, Ca, Oh, contact angle and the estimator are all
# right; the divergence is in what the MORPHOLOGY does afterwards.
#
# interface_length.py settles what that difference is. Specific interface length
# s = l_int/A_f agrees with their Fig. 3B within ~10% at every Ca we have except
# the highest, where ours is 1.200 against their 0.333. Their sd COLLAPSES at
# the top two Ca as large clusters short-circuit the flow across the periodic
# boundary; at 15 t_a ours had not, sitting at ~400 clusters with the largest
# holding only 0.19 of its phase.
#
# THE FIX IS TO RUN LONGER, AND IT IS CHEAP. Over t > 5 t_a the largest cluster
# grows +0.0145 per t_a (nw) and +0.0121 (w) -- noisy, but consistently upward,
# and extrapolating from 0.19 at 10 t_a to the ~0.98 of a short-circuited state
# reaches it at 60-70 t_a, i.e. inside the paper's own [30 t_a, 60 t_a] window.
# The apparent plateau in lambda is a METASTABLE state, not a converged one.
#
# Cost runs the right way: t_a ~ 1/Ca, so the points needing the most advective
# times are the cheapest per advective time. At Ca = 0.085, t_a is 7818 steps,
# so 60 t_a is 1.7 h. The step counts below are sized accordingly -- 60 t_a at
# the top, tapering to what is affordable at the bottom, where sd shows the
# morphology has ALREADY converged (0.301 against their 0.312 at Ca ~ 1e-3,
# after only 3.4 t_a).
#
# The old slab IC reaches the short-circuited state immediately and gives
# sd = 0.271, lambda = 0.129 at Ca ~ 0.1 -- closer to the published 0.333/0.058
# than the checkerboard's 15 t_a answer. That is a useful cross-check, NOT a
# substitute: it is right at high Ca for the wrong reason and badly wrong at
# low and mid Ca, where it produces two co-flowing monolithic phases and drives
# lambda toward zero. Do not reintroduce it as the sweep's IC.
#
# Accelerations below come from a measured 2D calibration on this exact
# geometry: accel_y = 1.0e-4 -> <u_y> = 2.7145e-3 -> Ca = 0.0855, flat from
# step 4000. Scaled linearly from there, which holds while the flow is
# viscosity-dominated but WILL drift at low Ca once capillary trapping cuts the
# mobility. Read the realised value back from each run:
#
#     Ca = <u_y> * nu / gamma      <u_y> = column 4 of out/series.txt
#     nu = (tau-0.5)/3 = 0.13333   gamma = surface_tension = 0.004233
#
# and plot lambda against THAT, not against the label.
set -euo pipefail

OUT=${1:?usage: run_fig5_2d.sh OUT_DIR [BIN]}
BIN=${2:-$PWD/build/felbm_gpu}
TPL="$(cd "$(dirname "$0")" && pwd)/fig5_2d_templates"
DRY_RUN=${DRY_RUN:-0}
NTRACER=${NTRACER:-10000}
GPU=${GPU:-0}
FROM=${FROM:-}      # skip points before this label (points are ordered cheapest first)
ONLY=${ONLY:-}      # comma-separated whitelist; overrides FROM

# Physical cores only -- SMT siblings share L1/L2/TLB, which is what the
# scattered write is limited by; 2x oversubscription measured ~10% WORSE.
mapfile -t PCPU < <(lscpu -p=CPU,Core,Socket 2>/dev/null | grep -v '^#' \
                    | awk -F, '!seen[$3","$2]++ {print $3","$2","$1}' \
                    | sort -t, -k1,1n -k2,2n | cut -d, -f3)
NCORE=${CORES:-${#PCPU[@]}}
CPUSET=${CPUSET:-$( IFS=,; echo "${PCPU[*]:${CPU_BASE:-0}:$NCORE}" )}

# label : accel_y : steps : file_skip (= one MEASURED t_a)   -- CHEAPEST FIRST
#
# Sized on the t_a we MEASURED, not the nominal 661.5/Ca: realised Ca came in
# well under its label at the low end (0.72x, 0.55x, 0.29x), so a nominal t_a
# understates the real one by up to 3.4x and the old counts bought far fewer
# advective times than they claimed.
#
#   label      measured t_a   target      steps    ~hours @ 12.7 ms/step
#   ca8.6e-2       7100        60 t_a      430k       1.5
#   ca3e-2        26357        60 t_a     1.58M       5.6
#   ca1e-2       103462        31 t_a     3.20M      11.3
#   ca3e-3       680295        11 t_a     7.50M      26.5
#   ca1e-3      1780245       8.4 t_a    15.00M      53.0
#
# The bottom two t_a are now MEASURED, not extrapolated: ca3e-3 came in at
# 680295 (Ca 9.7e-4) and ca1e-3 at 1780245 (Ca 3.7e-4, i.e. 0.32x its label and
# almost exactly the paper's lowest point at 4.3e-4). The earlier 2.6M guess for
# ca1e-3 was 46% high, which would have spaced its checkpoints 5.2M steps apart.
#
# 11 and 8.4 t_a are the weakest points in this sweep. The completed old ca3e-3
# reached only 3.24 t_a and gave lambda = 0.407 against the paper's 0.233, still
# descending -- and its MORPHOLOGY was already converged there (sd 0.301 against
# their 0.312), so what is still relaxing is the estimator's own transient from
# random initial tracer orientations, not the flow. Their Fig. S4 shows low-Ca
# curves settling by 5-10 t_a, so 11 should be adequate and 8.4 is marginal.
#
# The taper is deliberate. 60 t_a everywhere would cost 132 h at ca3e-3 alone,
# and it is not needed there: sd already matches the paper at the low end after
# 3.4 t_a, so the morphology is converged even when lambda is still settling.
# The high-Ca points are the ones that must reach 60, and they are the cheap
# ones. Run cheapest first and the decisive answer lands in the first 1.5 h.
POINTS=(
  "ca8.6e-2:1.170e-04:430000:7100"
  "ca3e-2:3.509e-05:1580000:26400"
  "ca1e-2:1.170e-05:3200000:103500"
  "ca3e-3:3.509e-06:7500000:680000"
  "ca1e-3:1.170e-06:15000000:1780000"
)

echo "run_fig5_2d: GPU $GPU, $NCORE cores [$CPUSET], $NTRACER tracers"
echo "  60d x 90d at d=21 (1260x1890), W=3.0 (eps_equiv 0.0505 d), theta=60 deg"
echo "  binary: $BIN"
tot=0
for p in "${POINTS[@]}"; do IFS=: read -r l a n f <<<"$p"; tot=$((tot+n)); done
echo "  $((${#POINTS[@]})) points, $((tot/1000000))M steps total, cheapest first"

started=0
for p in "${POINTS[@]}"; do
  IFS=: read -r label accel iters fskip <<<"$p"

  # Selection. FROM/ONLY exist so a partly-done sweep can be continued without
  # inventing completion markers: the resume guard below only knows about output
  # that actually exists, and faking it would leave silent holes in the curve.
  if [ -n "$ONLY" ]; then
    case ",$ONLY," in *",$label,"*) ;; *) echo "  skip $label (not in ONLY)"; continue;; esac
  elif [ -n "$FROM" ]; then
    [ "$label" = "$FROM" ] && started=1
    if [ "$started" != 1 ]; then echo "  skip $label (before FROM=$FROM)"; continue; fi
  fi

  R="$OUT/$label"
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
  cp "$TPL/domain.cfg" "$TPL/params.cfg" "$TPL/fluid.cfg" "$R/"
  # fskip is one t_a, so checkpoint every 2 t_a. Extending a run needs only the
  # LAST checkpoint, so 2x rather than 5x is bought for CRASH recovery: it caps
  # a lost crash at 2 t_a of work, at 482 MB per file.
  ckpt=$((fskip*2))
  sed -e "s|@ACCEL@|$accel|" -e "s|@ITERS@|$iters|" -e "s|@FSKIP@|$fskip|" \
      -e "s|@CKPT@|$ckpt|" \
      -e "s|@THREADS@|$NCORE|" -e "s|@NTRACER@|$NTRACER|" \
      "$TPL/settings.cfg" > "$R/settings.cfg"
  echo "  [$label] accel_y=$accel steps=$iters dump/$fskip"
  if [ "$DRY_RUN" = 1 ]; then continue; fi
  ( cd "$R" && CUDA_VISIBLE_DEVICES=$GPU OMP_PROC_BIND=close OMP_PLACES=cores \
      taskset -c "$CPUSET" "$BIN" settings.cfg > run.log 2>&1 )
  echo "    done: $(grep -o 'in [0-9.]* s' "$R/run.log" | tail -1)"
done

if [ "$DRY_RUN" = 1 ]; then echo "(dry run -- nothing launched)"; fi
cat <<'EOT'

Next:
  # realised Ca and lambda for each point
  for d in */; do python3 - "$d" <<'PY'
import numpy as np,sys,glob
s=np.loadtxt(glob.glob(sys.argv[1]+"out/series.txt")[0])
st=np.loadtxt(glob.glob(sys.argv[1]+"out/stretching.txt")[0])
nu,gam,d_=0.13333,0.004233,21.0
U=abs(s[-1][3]); Ca=U*nu/gam; ta=d_/U
t=st[:,0]/ta; lam=st[:,1]*ta
m=t>=t[-1]*0.7
print(f"{sys.argv[1]:12s} Ca={Ca:.3e}  t_a={ta:8.0f}  run={t[-1]:5.1f} t_a  "
      f"lambda*t_a={lam[m].mean():.4f}  lambda_w={st[m,5].mean()*ta:.4f}  lambda_nw={st[m,6].mean()*ta:.4f}")
PY
  done
  # This averages the last 30% of the record. That is a PROVISIONAL number, not
  # a converged lambda: the paper's own window is t in [30 t_a, 60 t_a] and no
  # point here reaches it. Compare against the published curve with that in mind
  # -- and see the WARNING in this file's header before concluding that the
  # answer is simply to run longer. It is not.
  #
  # Digitised from the paper for comparison (Table S3 gives Ca exactly; lambda
  # read off Fig. 5 at 600 dpi, good to about +/-0.005):
  #   Ca      4.3e-4 1.1e-3 2.6e-3 4.3e-3 6.2e-3 8.6e-3 1.7e-2 2.3e-2 4.8e-2 9.9e-2
  #   lambda  0.185  0.233  0.259  0.286  0.320  0.329  0.314  0.270  0.087  0.058
  # and their fitted model, Eq. (6), is
  #   lambda = 3.25 sqrt(Ca) * (log 0.267 - 0.5 log Ca),  peaking at Ca ~ 1e-2.
EOT
