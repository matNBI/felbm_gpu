#!/usr/bin/env bash
#
# run_fig5_2d.sh -- reproduce Fig. 5 of Linga et al. in 2D, on ONE GPU.
#
#   ./run_fig5_2d.sh OUT_DIR [BIN]
#   DRY_RUN=1 ./run_fig5_2d.sh OUT_DIR          # print the plan only
#
#   GPU=3 CORES=24 ./run_fig5_2d.sh OUT_DIR     # pick card and core budget
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
# sized for 10 t_a NOMINAL, giving ~7 t_a at 0.7x; lambda plateaus by 3-4 t_a with
# this IC, so that is ample. (15 t_a was the old sizing for the obsolete
# lambda_inf + A/t extrapolation -- see docs/HANDOFF_stretching.md section 0.)
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
NTRACER=${NTRACER:-20000}
GPU=${GPU:-0}

# Physical cores only -- SMT siblings share L1/L2/TLB, which is what the
# scattered write is limited by; 2x oversubscription measured ~10% WORSE.
mapfile -t PCPU < <(lscpu -p=CPU,Core,Socket 2>/dev/null | grep -v '^#' \
                    | awk -F, '!seen[$3","$2]++ {print $3","$2","$1}' \
                    | sort -t, -k1,1n -k2,2n | cut -d, -f3)
NCORE=${CORES:-${#PCPU[@]}}
CPUSET=${CPUSET:-$( IFS=,; echo "${PCPU[*]:${CPU_BASE:-0}:$NCORE}" )}

# label : accel_y : steps (15 t_a) : file_skip (= t_a)   -- CHEAPEST FIRST
POINTS=(
  "ca8.6e-2:1.170e-04:66150:6615"
  "ca3e-2:3.509e-05:220500:22050"
  "ca1e-2:1.170e-05:661500:66150"
  "ca3e-3:3.509e-06:2205000:220500"
  "ca1e-3:1.170e-06:6615000:661500"
)

echo "run_fig5_2d: GPU $GPU, $NCORE cores [$CPUSET], $NTRACER tracers"
echo "  60d x 90d at d=21 (1260x1890), W=3.0 (eps_equiv 0.0505 d), theta=60 deg"
echo "  binary: $BIN"
tot=0
for p in "${POINTS[@]}"; do IFS=: read -r l a n f <<<"$p"; tot=$((tot+n)); done
echo "  $((${#POINTS[@]})) points, $((tot/1000000))M steps total, cheapest first"

for p in "${POINTS[@]}"; do
  IFS=: read -r label accel iters fskip <<<"$p"
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
  sed -e "s|@ACCEL@|$accel|" -e "s|@ITERS@|$iters|" -e "s|@FSKIP@|$fskip|" \
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
  # With the checkerboard IC lambda PLATEAUS -- just average the last ~3 t_a.
  # Do NOT use the lambda_inf + A/t fit from section 3.5b: that existed to undo
  # the old slab IC's decay toward zero and now biases the answer low.
EOT
