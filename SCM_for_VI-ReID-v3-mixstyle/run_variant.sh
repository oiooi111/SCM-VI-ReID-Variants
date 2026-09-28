#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python3 main.py \
  --mode train \
  --output_path experiments/v3_mixstyle \
  --auto_resume_training_from_lastest_step False \
  --seed 1 \
  --mixstyle_p 0.50 \
  --mixstyle_alpha 0.10 \
  "$@"
