#!/usr/bin/env bash
# SpatialResMoE_3 YOLO on test_iv_2000. Same weights / staging as Full and A2.
# Does not overwrite Fusion_noleak_1 rows. Does not shutdown.
set -euo pipefail

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PROJECT=/root/autodl-tmp/URFusion-main
PY=/root/autodl-tmp/conda/envs/urfusion/bin/python
ROOT=$PROJECT/vis-ir-gray-moe_update
EXP="${EXP:-SpatialResMoE_3}"
TEST=$PROJECT/datasets/yolo/test_iv_2000
OUT=$PROJECT/datasets/yolo/fusion_results/$EXP
SAVE=$ROOT/eval_save/yolo_det
OFFICIAL=$PROJECT/metrics_save/yolo_det_yolo_test_2000
INFER=$ROOT/code/fusion_gray_infer.py
COLOR=$PROJECT/metrics_save/llvip_sample1000/colorize_gray_infer.py
CKPT_ROOT=$ROOT/train-jobs/ckpt
CKPT=$CKPT_ROOT/$EXP/${EXP}_ckpt.pth

mkdir -p "$SAVE" "$OUT/Y_fused" "$OUT/RGB_fused"
LOG=$SAVE/pipeline.log
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

count_images() {
	local dir=$1
	[[ -d "$dir" ]] || { echo 0; return; }
	find "$dir" \( -type f -o -type l \) \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.bmp' \) | wc -l
}

[[ -f "$CKPT" ]] || { log "ERROR: missing ckpt $CKPT"; exit 1; }
n_test=$(count_images "$TEST/vis")
[[ "$n_test" -eq 2000 ]] || { log "ERROR: test_iv_2000 vis=$n_test"; exit 1; }

avail_kb=$(df -k /root/autodl-tmp | awk 'NR==2 {print $4}')
log "========== YOLO START exp=$EXP vis=$n_test disk=$((avail_kb/1024/1024))G =========="
log "ckpt=$CKPT"

n_y=$(count_images "$OUT/Y_fused")
if [[ "$n_y" -lt 2000 ]]; then
	log "===== infer Y ($n_y/2000) ====="
	"$PY" "$INFER" \
		--experiment "$EXP" \
		--dataset yolo_test_2000 \
		--test_dir "$TEST" \
		--ckpt_root "$CKPT_ROOT" \
		--y_out_dir "$OUT/Y_fused" \
		2>&1 | tee -a "$LOG"
	n_y=$(count_images "$OUT/Y_fused")
else
	log "===== skip infer (Y=$n_y) ====="
fi
[[ "$n_y" -eq 2000 ]] || { log "ERROR: Y_fused $n_y/2000"; exit 1; }

n_rgb=$(count_images "$OUT/RGB_fused")
if [[ "$n_rgb" -lt 2000 ]]; then
	log "===== colorize VIS Cb/Cr ($n_rgb/2000) ====="
	"$PY" "$COLOR" \
		--method "$EXP" \
		--dataset yolo_test_2000 \
		--test_dir "$TEST" \
		--y_dir "$OUT/Y_fused" \
		--rgb_out_dir "$OUT/RGB_fused" \
		2>&1 | tee -a "$LOG"
	n_rgb=$(count_images "$OUT/RGB_fused")
else
	log "===== skip colorize (RGB=$n_rgb) ====="
fi
[[ "$n_rgb" -eq 2000 ]] || { log "ERROR: RGB_fused $n_rgb/2000"; exit 1; }

log "===== prepare staging ====="
"$PY" "$PROJECT/metrics_save/yolo_det/prepare_eval_staging.py" \
	--staging_root "$PROJECT/datasets/yolo/eval_staging" \
	--methods "$EXP" \
	2>&1 | tee -a "$LOG"

log "===== YOLO val ====="
"$PY" "$PROJECT/metrics_save/yolo_det/eval_yolo_detection.py" \
	--weights "$PROJECT/datasets/yolo/runs/yolo11s_iv_det/weights/best.pt" \
	--staging_root "$PROJECT/datasets/yolo/eval_staging" \
	--save_dir "$SAVE" \
	--inputs "$EXP" \
	--device 0 --batch 16 --imgsz 640 \
	2>&1 | tee -a "$LOG"

log "===== append official detection table ====="
"$PY" - "$SAVE" "$OFFICIAL" "$EXP" <<'PY'
import csv
import sys
from pathlib import Path

from openpyxl import load_workbook

src, official, exp = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
src_csv = src / 'detection_comparison.csv'
dst_csv = official / 'detection_comparison.csv'
dst_xlsx = official / 'detection_comparison.xlsx'
new_rows = list(csv.DictReader(src_csv.open()))
old_rows = list(csv.DictReader(dst_csv.open()))
fieldnames = list(old_rows[0].keys())
keep = [r for r in old_rows if r['input'] != exp]
merged = keep + new_rows
with dst_csv.open('w', newline='') as f:
	w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
	w.writeheader()
	w.writerows(merged)
wb = load_workbook(dst_xlsx)
ws = wb.active
headers = [c.value for c in ws[1]]
for row_idx in range(ws.max_row, 1, -1):
	if ws.cell(row_idx, 1).value == exp:
		ws.delete_rows(row_idx)
for row in new_rows:
	ws.append([row.get(h, '') for h in headers])
wb.save(dst_xlsx)
print('official rows:', [r['input'] for r in merged])
for r in new_rows:
	print(r['input'], 'mAP50=', r['mAP50'], 'mAP50-95=', r['mAP50-95'])
PY

date '+%F %T' > "$SAVE/COMPLETE"
log "========== YOLO COMPLETE ====="
log "metrics=$SAVE/${EXP}_det_metrics.json"
