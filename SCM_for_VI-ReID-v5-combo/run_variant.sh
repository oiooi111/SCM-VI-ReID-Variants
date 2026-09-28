#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python3 main.py \
  --mode train \
  --output_path experiments/v5_combo \
  --auto_resume_training_from_lastest_step False \
  --seed 1 \
  --lambda_hc 0.50 \
  --hc_margin 0.30 \
  --mixstyle_p 0.50 \
  --mixstyle_alpha 0.10 \
  --stage3_lr_min 1e-6 \
  --stage3_warmup_lr_init 1e-5 \
  --stage3_warmup_epochs 5 \
  "$@"
