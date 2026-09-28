#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python3 main.py \
  --mode train \
  --output_path experiments/improved \
  --auto_resume_training_from_lastest_step False \
  --seed 1 \
  --lambda_inv 0.30 \
  --lambda_center 0.05 \
  --kl_temperature 2.0 \
  --label_smoothing 0.10 \
  "$@"
