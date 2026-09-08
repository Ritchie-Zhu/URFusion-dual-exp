#!/usr/bin/env bash
# Re-infer URFusion on YOLO test 2000 with official ckpt/vis.mat, then re-run
# detection eval for all inputs (same protocol as run_yolo_det_eval.sh).
set -euo pipefail

PROJECT=/root/autodl-tmp/URFusion-main
PYTHON=/root/autodl-tmp/conda/envs/urfusion/bin/python
URFUSION_CODE="$PROJECT/URFusion/vis-ir/code"
CKPT="$PROJECT/URFusion/vis-ir/train-jobs/ckpt/content-fusion_ckpt.pth"
VIS_MAT="$PROJECT/URFusion/vis-ir/train-jobs/vis.mat"
A2V="$PROJECT/URFusion/vis-ir/train-jobs/ckpt/A2V_ckpt.pth"
TEST_DIR="$PROJECT/datasets/yolo/test_iv_2000"
OUT_DIR="$PROJECT/datasets/yolo/fusion_results/URFusion/RGB_fused"
STAGING="$PROJECT/datasets/yolo/fusion_results/URFusion_staging"
ORCH="$PROJECT/metrics_save/yolo_det"
SAVE_DIR="$PROJECT/metrics_save/yolo_det_yolo_test_2000"
WEIGHTS="$PROJECT/datasets/yolo/runs/yolo11s_iv_det/weights/best.pt"
EVAL_STAGING="$PROJECT/datasets/yolo/eval_staging"
LOG_DIR="$ORCH/urfusion_official_yolo2000"
LOG="$LOG_DIR/pipeline.log"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

count_out() {
	find "$1" -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.bmp' \) 2>/dev/null | wc -l
}

log "========== URFusion official @ YOLO test 2000 START =========="
for f in "$CKPT" "$VIS_MAT" "$A2V" "$WEIGHTS"; do
	[[ -f "$f" ]] || { log "ERROR: missing $f"; exit 1; }
done
log "ckpt: $CKPT"
log "vis.mat: $VIS_MAT"
log "A2V: $A2V"
log "weights: $WEIGHTS"

log "===== rebuild test_iv_2000 ====="
$PYTHON "$PROJECT/datasets/yolo/build_test_iv_2000.py" 2>&1 | tee -a "$LOG_DIR/build_test.log"
N_VIS=$(count_out "$TEST_DIR/vis")
N_IR=$(count_out "$TEST_DIR/ir")
# count_out misses symlinks; use find -type l too
N_VIS=$(find "$TEST_DIR/vis" \( -type f -o -type l \) | wc -l)
N_IR=$(find "$TEST_DIR/ir" \( -type f -o -type l \) | wc -l)
log "test pairs: vis=$N_VIS ir=$N_IR"
if [[ "$N_VIS" -ne 2000 ]] || [[ "$N_IR" -ne 2000 ]]; then
	log "ERROR: expected 2000 pairs"
	exit 1
fi

log "===== URFusion inference (official test.py) ====="
rm -rf "$STAGING" "$OUT_DIR"
mkdir -p "$STAGING" "$OUT_DIR"
# clean leftover wrong-ckpt staging if present
rm -rf "$PROJECT/vis-ir/results/URFusion_yolo_test_2000_staging"

cd "$URFUSION_CODE"
$PYTHON test.py \
	--device 0 \
	--testDir "$TEST_DIR" \
	--outputDir "$STAGING/" \
	--fusion_ckpt content-fusion \
	--A2V_ckpt A2V \
	2>&1 | tee -a "$LOG_DIR/URFusion_infer.log"

rsync -a "$STAGING/" "$OUT_DIR/"
N_OUT=$(count_out "$OUT_DIR")
log "URFusion fused: $N_OUT/2000 -> $OUT_DIR"
if [[ "$N_OUT" -lt 2000 ]]; then
	log "ERROR: incomplete inference ($N_OUT/2000)"
	exit 1
fi
rm -rf "$STAGING"

log "===== rebuild eval_staging (all methods, fair) ====="
$PYTHON "$ORCH/prepare_eval_staging.py" --staging_root "$EVAL_STAGING" \
	2>&1 | tee -a "$LOG_DIR/prepare_staging.log"

log "===== YOLO detection eval (Visible + 8 methods, same protocol) ====="
# Clear prior URFusion metrics; full table regenerated for all inputs
$PYTHON "$ORCH/eval_yolo_detection.py" \
	--weights "$WEIGHTS" \
	--staging_root "$EVAL_STAGING" \
	--save_dir "$SAVE_DIR" \
	--device 0 \
	--batch 16 \
	--imgsz 640 \
	2>&1 | tee -a "$LOG_DIR/det_eval.log"

log "========== COMPLETE =========="
log "fused: $OUT_DIR ($N_OUT)"
log "det:   $SAVE_DIR/detection_comparison.csv"
log "       $SAVE_DIR/URFusion_det_metrics.json"
