#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python3 main.py \
  --mode train \
  --output_path experiments/v4_adamw_cosine \
  --auto_resume_training_from_lastest_step False \
  --seed 1 \
  --stage3_lr_min 1e-6 \
  --stage3_warmup_lr_init 1e-5 \
  --stage3_warmup_epochs 5 \
  "$@"
