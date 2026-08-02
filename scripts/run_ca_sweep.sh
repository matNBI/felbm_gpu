#!/usr/bin/env bash
#
# run_ca_sweep.sh -- launch a Ca sweep, one point per GPU.
#
# Each Ca point is an independent felbm_gpu run, so the sweep is embarrassingly
# parallel: one process per card via CUDA_VISIBLE_DEVICES.  Wall time is set by
# the SLOWEST point, not the total -- t_a ~ 1/Ca, so the lowest Ca dominates and
# adding cards does not shorten it.
#
#   ./run_ca_sweep.sh GEOM_DIR OUT_DIR [BIN]
#
# GEOM_DIR must contain image/ and domain.cfg from voxelize_spheres.py.
# BIN defaults to ./build/felbm_gpu.
#
# Ca is DIAGNOSED, not imposed -- as in Linga et al. Table S3, which lists
# "capillary numbers resulting from different driving forces".  The accelerations
# below are calibrated from a measured point (accel 3.0e-3 -> Ca 0.18 on an
# 8x8x12 d, d=20 RCP pack) assuming Darcy linearity.  That linearity WILL break
# at low Ca, where capillary trapping cuts the mobility, so the realised Ca will
# drift above the label.  Read the true value from the run:
#
#     Ca = <u_z> * nu / gamma      with <u_z> = column 5 of series.txt
#
# and relabel the point accordingly rather than trusting the name.
set -euo pipefail

GEOM=${1:?usage: run_ca_sweep.sh GEOM_DIR OUT_DIR [BIN]}
OUT=${2:?usage: run_ca_sweep.sh GEOM_DIR OUT_DIR [BIN]}
BIN=${3:-$PWD/build/felbm_gpu}

# label   accel_z    max_iterations (=15 t_a)   file_skip (=t_a)
POINTS=(
  "ca1e-1  1.667e-03      90000      6000"
  "ca3e-2  5.000e-04     300000     20000"
  "ca1e-2  1.667e-04     900000     60000"
  "ca3e-3  5.000e-05    3000000    200000"
  "ca1e-3  1.667e-05    9000000    600000"
)

NGPU=$(nvidia-smi -L | wc -l)
echo "run_ca_sweep: $NGPU GPUs, ${#POINTS[@]} points, geometry $GEOM"
[ "${#POINTS[@]}" -gt "$NGPU" ] && echo "WARNING: more points than GPUs; the extras will share cards" >&2

i=0
for p in "${POINTS[@]}"; do
  set -- $p; label=$1; accel=$2; iters=$3; fskip=$4
  dev=$(( i % NGPU )); i=$(( i + 1 ))
  R="$OUT/$label"; mkdir -p "$R/out"
  cp -r "$GEOM/image" "$R/image"; cp "$GEOM/domain.cfg" "$R/domain.cfg"
  sed -i 's|^image_dir *=.*|image_dir         = ./image/|' "$R/domain.cfg"
  cp "$(dirname "$0")/ca_sweep_templates/params.cfg" "$R/params.cfg"
  cp "$(dirname "$0")/ca_sweep_templates/fluid.cfg"  "$R/fluid.cfg"
  sed -e "s|@ACCEL@|$accel|" -e "s|@ITERS@|$iters|" -e "s|@FSKIP@|$fskip|" \
      "$(dirname "$0")/ca_sweep_templates/settings.cfg" > "$R/settings.cfg"

  ( cd "$R" && CUDA_VISIBLE_DEVICES=$dev nohup "$BIN" settings.cfg > run.log 2>&1 ) &
  echo "  $label -> GPU $dev, $iters steps, accel_z=$accel, dump/$fskip   ($R)"
done

echo "launched; wait for all with: wait"
wait
echo "run_ca_sweep: all points finished"
