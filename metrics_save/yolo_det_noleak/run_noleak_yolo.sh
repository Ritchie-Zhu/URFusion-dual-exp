#!/usr/bin/env bash
set -euo pipefail
export PROJECT=/root/autodl-tmp/URFusion-main
export PYTHON=/root/autodl-tmp/conda/envs/urfusion/bin/python
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export TEST_DIR=$PROJECT/datasets/yolo/test_iv_2000
export OUT=$PROJECT/datasets/yolo/fusion_results/Fusion_noleak_1
export INFER=$PROJECT/metrics_save/llvip_sample1000/fusion_gray_infer.py
export COLOR=$PROJECT/metrics_save/llvip_sample1000/colorize_gray_infer.py
export SAVE=$PROJECT/metrics_save/yolo_det_noleak
mkdir -p $OUT/Y_fused $OUT/RGB_fused $SAVE

$PYTHON $INFER --experiment Fusion_noleak_1 --dataset yolo_test_2000 \
  --test_dir $TEST_DIR --y_out_dir $OUT/Y_fused \
  2>&1 | tee $OUT/infer.log

$PYTHON $COLOR --method Fusion_noleak_1 --dataset yolo_test_2000 \
  --test_dir $TEST_DIR --y_dir $OUT/Y_fused --rgb_out_dir $OUT/RGB_fused \
  2>&1 | tee $OUT/colorize.log

N=$(find $OUT/RGB_fused -type f | wc -l)
echo "RGB_fused=$N"
[ "$N" -eq 2000 ]

$PYTHON $PROJECT/metrics_save/yolo_det/prepare_eval_staging.py \
  --staging_root $PROJECT/datasets/yolo/eval_staging \
  --methods Fusion_noleak_1 \
  2>&1 | tee $SAVE/prepare_staging.log

$PYTHON $PROJECT/metrics_save/yolo_det/eval_yolo_detection.py \
  --weights $PROJECT/datasets/yolo/runs/yolo11s_iv_det/weights/best.pt \
  --staging_root $PROJECT/datasets/yolo/eval_staging \
  --save_dir $SAVE \
  --inputs Fusion_noleak_1 \
  --device 0 --batch 16 --imgsz 640 \
  2>&1 | tee $SAVE/eval.log

echo '===== DONE ====='
cat $SAVE/Fusion_noleak_1_det_metrics.json
echo '[SHUTDOWN] AutoDL instance stopping'
/usr/bin/shutdown
