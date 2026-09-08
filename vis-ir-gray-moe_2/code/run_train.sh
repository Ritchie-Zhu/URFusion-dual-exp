#!/usr/bin/env bash
# Train SpatialResMoE_2 (leak-free: E_base on f_cap only).
# All outputs under vis-ir-gray-moe_2/train-jobs. Does not start unless invoked.
set -euo pipefail

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1

PROJECT=/root/autodl-tmp/URFusion-main
PY=/root/autodl-tmp/conda/envs/urfusion/bin/python
ROOT=$PROJECT/vis-ir-gray-moe_2
CODE=$ROOT/code
LOGDIR=$ROOT/train-jobs/console_logs
mkdir -p "$LOGDIR" \
  "$ROOT/train-jobs/ckpt" \
  "$ROOT/train-jobs/metrics_logs" \
  "$ROOT/train-jobs/fixed_examples"

echo "[$(date '+%F %T')] START SpatialResMoE_2 leak-free E_base(f_cap)"
"$PY" "$CODE/train_fusion_gray.py" \
  --device 0 \
  --experiment SpatialResMoE_2 \
  --baseDir "$PROJECT/datasets/training_noleak" \
  --testDir "$PROJECT/datasets/M3FD/test" \
  --vis_content_ckpt Vis_Content_noleak \
  --ir_content_ckpt ir_Content_noleak \
  --content_ckpt_root "$PROJECT/vis-ir-gray/train-jobs/ckpt" \
  --ckptRoot "$ROOT/train-jobs/ckpt" \
  --logRoot "$ROOT/train-jobs/metrics_logs" \
  --fixedRoot "$ROOT/train-jobs/fixed_examples" \
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
  2>&1 | tee "$LOGDIR/SpatialResMoE_2.log"

echo "[$(date '+%F %T')] DONE SpatialResMoE_2"
