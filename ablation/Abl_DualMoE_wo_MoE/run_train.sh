#!/usr/bin/env bash
# Train A2: Abl_DualMoE_wo_MoE (dual PreNet + single expert). lambda_aux=0.
set -euo pipefail

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1

PROJECT=/root/autodl-tmp/URFusion-main
PY=/root/autodl-tmp/conda/envs/urfusion/bin/python
ABL=$PROJECT/ablation/Abl_DualMoE_wo_MoE
DATA=$PROJECT/datasets/training_noleak
TEST=$PROJECT/datasets/M3FD/test
LOGDIR=$PROJECT/our_model_1_DualMoE/train-jobs/console_logs
mkdir -p "$LOGDIR"

echo "[$(date '+%F %T')] START Abl_DualMoE_wo_MoE"
"$PY" "$ABL/train_fusion_gray.py" \
  --device 0 \
  --experiment Abl_DualMoE_wo_MoE \
  --fusion_model dual_conv \
  --baseDir "$DATA" \
  --testDir "$TEST" \
  --vis_content_ckpt Vis_Content_noleak \
  --ir_content_ckpt ir_Content_noleak \
  --numEpoch 80 \
  --batchsize 12 \
  --patchsize 160 \
  --ckpt_interval 10 \
  --save_fixed_every 20 \
  --num_workers 12 \
  --lambda_int 0.1 \
  --lambda_grad 1.0 \
  --lambda_branch 0.01 \
  --lambda_aux 0 \
  --top_k 2 \
  --cudnn_benchmark \
  "$@" \
  2>&1 | tee "$LOGDIR/Abl_DualMoE_wo_MoE.log"

echo "[$(date '+%F %T')] DONE Abl_DualMoE_wo_MoE"
