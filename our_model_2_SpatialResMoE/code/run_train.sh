#!/usr/bin/env bash
# Train SpatialResMoE_1. All outputs under our_model_2_SpatialResMoE/train-jobs.
set -euo pipefail

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1

PROJECT=/root/autodl-tmp/URFusion-main
PY=/root/autodl-tmp/conda/envs/urfusion/bin/python
ROOT=$PROJECT/our_model_2_SpatialResMoE
CODE=$ROOT/code
LOGDIR=$ROOT/train-jobs/console_logs
mkdir -p "$LOGDIR" \
  "$ROOT/train-jobs/ckpt" \
  "$ROOT/train-jobs/metrics_logs" \
  "$ROOT/train-jobs/fixed_examples"

echo "[$(date '+%F %T')] START SpatialResMoE_3"
"$PY" "$CODE/train_fusion_gray.py" \
  --device 0 \
  --experiment SpatialResMoE_3 \
  --baseDir "$PROJECT/datasets/training_noleak" \
  --testDir "$PROJECT/datasets/M3FD/test" \
  --vis_content_ckpt Vis_Content_noleak \
  --ir_content_ckpt ir_Content_noleak \
  --content_ckpt_root "$PROJECT/our_model_1_DualMoE/train-jobs/ckpt" \
  --numEpoch 80 \
  --batchsize 12 \
  --patchsize 160 \
  --ckpt_interval 10 \
  --save_fixed_every 10 \
  --n_fixed 8 \
  --num_workers 12 \
  --lambda_int 0.1 \
  --lambda_grad 1.0 \
  --lambda_branch 0.01 \
  --lambda_route 0.05 \
  --cudnn_benchmark \
  "$@" \
  2>&1 | tee "$LOGDIR/SpatialResMoE_3.log"

echo "[$(date '+%F %T')] DONE SpatialResMoE_3"
