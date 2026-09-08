#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/URFusion-main/vis-ir-gray/code

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1

PY=/root/autodl-tmp/conda/envs/urfusion/bin/python
DATA=/root/autodl-tmp/URFusion-main/datasets/training_noleak
TEST=/root/autodl-tmp/URFusion-main/datasets/M3FD/test
LOG=/root/autodl-tmp/URFusion-main/vis-ir-gray/train-jobs/console_logs/Abl_wo_Lbranch.log

echo "[$(date '+%F %T')] RESUME Abl_wo_Lbranch from epoch 58" | tee -a "$LOG"

"$PY" train_fusion_gray.py \
  --device 0 \
  --experiment Abl_wo_Lbranch \
  --fusion_model dual_moe_gray \
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
  --lambda_branch 0 \
  --lambda_aux 0.03 \
  --top_k 2 \
  --cudnn_benchmark \
  >> "$LOG" 2>&1

echo "[$(date '+%F %T')] DONE Abl_wo_Lbranch resume"
echo '[SHUTDOWN] AutoDL instance stopping'
/usr/bin/shutdown
