#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python3 main.py \
  --mode train \
  --output_path experiments/v2_circle \
  --auto_resume_training_from_lastest_step False \
  --seed 1 \
  --lambda_circle 0.10 \
  --circle_margin 0.25 \
  --circle_gamma 128 \
  "$@"
