#!/usr/bin/env bash
# Train Abl_SpatialResMoE_wo_Res. All outputs under ablation/Abl_SpatialResMoE_wo_Res/train-jobs.
set -euo pipefail

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1

PROJECT=/root/autodl-tmp/URFusion-main
PY=/root/autodl-tmp/conda/envs/urfusion/bin/python
ROOT=$PROJECT/ablation/Abl_SpatialResMoE_wo_Res
CODE=$ROOT
LOGDIR=$ROOT/train-jobs/console_logs
mkdir -p "$LOGDIR" \
  "$ROOT/train-jobs/ckpt" \
  "$ROOT/train-jobs/metrics_logs"

echo "[$(date '+%F %T')] START Abl_SpatialResMoE_wo_Res"
"$PY" "$CODE/train_fusion_gray.py" \
  --device 0 \
  --experiment Abl_SpatialResMoE_wo_Res \
  --baseDir "$PROJECT/datasets/training_noleak" \
  --testDir "$PROJECT/datasets/M3FD/test" \
  --vis_content_ckpt Vis_Content_noleak \
  --ir_content_ckpt ir_Content_noleak \
  --content_ckpt_root "$PROJECT/our_model_1_DualMoE/train-jobs/ckpt" \
  --ckptRoot "$ROOT/train-jobs/ckpt" \
  --logRoot "$ROOT/train-jobs/metrics_logs" \
  --fixedRoot "$ROOT/train-jobs/fixed_examples" \
  --numEpoch 80 \
  --batchsize 12 \
  --patchsize 160 \
  --ckpt_interval 10 \
  --save_fixed_every 0 \
  --num_workers 12 \
  --lambda_int 0.1 \
  --lambda_grad 1.0 \
  --lambda_branch 0.01 \
  --cudnn_benchmark \
  "$@" \
  2>&1 | tee "$LOGDIR/Abl_SpatialResMoE_wo_Res.log"

echo "[$(date '+%F %T')] DONE Abl_SpatialResMoE_wo_Res"
