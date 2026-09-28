#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python3 main.py \
  --mode train \
  --output_path experiments/v1_hc \
  --auto_resume_training_from_lastest_step False \
  --seed 1 \
  --lambda_hc 0.50 \
  --hc_margin 0.30 \
  "$@"
