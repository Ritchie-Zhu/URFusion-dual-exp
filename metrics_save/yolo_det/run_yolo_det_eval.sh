#!/usr/bin/env bash
# YOLO detection eval: Visible baseline + 8 fusion methods on test 2000.
# Results -> metrics_save/yolo_det_yolo_test_2000/
set -euo pipefail

PROJECT=/root/autodl-tmp/URFusion-main
PYTHON=${PYTHON:-/root/autodl-tmp/conda/envs/urfusion/bin/python}
ORCH="$PROJECT/metrics_save/yolo_det"
SAVE_DIR="$PROJECT/metrics_save/yolo_det_yolo_test_2000"
WEIGHTS="$PROJECT/datasets/yolo/runs/yolo11s_iv_det/weights/best.pt"
STAGING="$PROJECT/datasets/yolo/eval_staging"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "========== YOLO Detection Eval (test 2000) =========="
echo "weights: $WEIGHTS"
echo "save:    $SAVE_DIR"

# Step 1: build staging (visible + fusion images aligned to yolo labels)
"$PYTHON" "$ORCH/prepare_eval_staging.py" --staging_root "$STAGING"

# Step 2: run val for all inputs
"$PYTHON" "$ORCH/eval_yolo_detection.py" \
	--weights "$WEIGHTS" \
	--staging_root "$STAGING" \
	--save_dir "$SAVE_DIR" \
	--device 0 \
	--batch 16 \
	--imgsz 640

echo ""
echo "Results:"
echo "  $SAVE_DIR/summary.json"
echo "  $SAVE_DIR/detection_comparison.csv"
echo "  $SAVE_DIR/detection_comparison.xlsx"
ls -la "$SAVE_DIR"/*.json 2>/dev/null | head -15
