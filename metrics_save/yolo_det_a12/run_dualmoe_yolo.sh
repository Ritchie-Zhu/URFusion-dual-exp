#!/usr/bin/env bash
# A1 Abl_DualMoE_wo_Dual + A2 Abl_DualMoE_wo_MoE YOLO eval. Shutdown only on full success.
set -euo pipefail
export PROJECT=/root/autodl-tmp/URFusion-main
export PYTHON=/root/autodl-tmp/conda/envs/urfusion/bin/python
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TEST_DIR=$PROJECT/datasets/yolo/test_iv_2000
export COLOR=$PROJECT/metrics_save/llvip_sample1000/colorize_gray_infer.py
export SAVE=$PROJECT/metrics_save/yolo_det_a12
export OFFICIAL=$PROJECT/metrics_save/yolo_det_yolo_test_2000
mkdir -p "$SAVE"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$SAVE/pipeline.log"; }

count_images() {
  find "$1" \( -type f -o -type l \) \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.bmp' \) 2>/dev/null | wc -l
}

avail_kb=$(df -k /root/autodl-tmp | awk 'NR==2 {print $4}')
avail_g=$((avail_kb / 1024 / 1024))
log "disk_avail=${avail_g}G"
if [[ "$avail_kb" -lt $((4 * 1024 * 1024)) ]]; then
  log "ERROR: need >=4G free"
  exit 1
fi

N_TEST=$(count_images "$TEST_DIR/vis")
log "test_iv_2000 vis=$N_TEST"
[[ "$N_TEST" -eq 2000 ]]

infer_one() {
  local method=$1 infer_py=$2
  local out=$PROJECT/datasets/yolo/fusion_results/$method
  mkdir -p "$out/Y_fused" "$out/RGB_fused"
  local n_y n_rgb
  n_y=$(count_images "$out/Y_fused")
  if [[ "$n_y" -lt 2000 ]]; then
    log "===== infer $method ($n_y/2000) ====="
    "$PYTHON" "$infer_py" \
      --experiment "$method" \
      --dataset yolo_test_2000 \
      --test_dir "$TEST_DIR" \
      --y_out_dir "$out/Y_fused" \
      2>&1 | tee "$out/infer.log"
    n_y=$(count_images "$out/Y_fused")
  else
    log "===== skip infer $method (Y=$n_y) ====="
  fi
  [[ "$n_y" -eq 2000 ]]

  n_rgb=$(count_images "$out/RGB_fused")
  if [[ "$n_rgb" -lt 2000 ]]; then
    log "===== colorize $method ($n_rgb/2000) ====="
    "$PYTHON" "$COLOR" \
      --method "$method" \
      --dataset yolo_test_2000 \
      --test_dir "$TEST_DIR" \
      --y_dir "$out/Y_fused" \
      --rgb_out_dir "$out/RGB_fused" \
      2>&1 | tee "$out/colorize.log"
    n_rgb=$(count_images "$out/RGB_fused")
  else
    log "===== skip colorize $method (RGB=$n_rgb) ====="
  fi
  [[ "$n_rgb" -eq 2000 ]]
  log "  verify $method Y=$n_y RGB=$n_rgb"
}

log "========== A1/A2 YOLO START =========="
infer_one Abl_DualMoE_wo_Dual "$PROJECT/ablation/Abl_DualMoE_wo_Dual/fusion_gray_infer.py"
infer_one Abl_DualMoE_wo_MoE  "$PROJECT/ablation/Abl_DualMoE_wo_MoE/fusion_gray_infer.py"

log "===== prepare staging ====="
"$PYTHON" "$PROJECT/metrics_save/yolo_det/prepare_eval_staging.py" \
  --staging_root "$PROJECT/datasets/yolo/eval_staging" \
  --methods Abl_DualMoE_wo_Dual Abl_DualMoE_wo_MoE \
  2>&1 | tee "$SAVE/prepare_staging.log"

log "===== YOLO val ====="
"$PYTHON" "$PROJECT/metrics_save/yolo_det/eval_yolo_detection.py" \
  --weights "$PROJECT/datasets/yolo/runs/yolo11s_iv_det/weights/best.pt" \
  --staging_root "$PROJECT/datasets/yolo/eval_staging" \
  --save_dir "$SAVE" \
  --inputs Abl_DualMoE_wo_Dual Abl_DualMoE_wo_MoE \
  --device 0 --batch 16 --imgsz 640 \
  2>&1 | tee "$SAVE/eval.log"

log "===== append official detection table ====="
"$PYTHON" - "$SAVE" "$OFFICIAL" << 'PY'
import csv, sys
from pathlib import Path
from openpyxl import load_workbook, Workbook

src, official = Path(sys.argv[1]), Path(sys.argv[2])
src_csv = src / 'detection_comparison.csv'
dst_csv = official / 'detection_comparison.csv'
dst_xlsx = official / 'detection_comparison.xlsx'
new_rows = list(csv.DictReader(src_csv.open()))
old_rows = list(csv.DictReader(dst_csv.open()))
fieldnames = list(old_rows[0].keys())
keep = [r for r in old_rows if r['input'] not in {x['input'] for x in new_rows}]
merged = keep + new_rows
with dst_csv.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
    w.writeheader()
    w.writerows(merged)
wb = load_workbook(dst_xlsx)
ws = wb.active
headers = [c.value for c in ws[1]]
existing = {ws.cell(r, 1).value for r in range(2, ws.max_row + 1)}
for row in new_rows:
    if row['input'] in existing:
        continue
    ws.append([row.get(h, '') for h in headers])
wb.save(dst_xlsx)
print('official rows:', [r['input'] for r in merged])
for r in new_rows:
    print(r['input'], 'mAP50=', r['mAP50'], 'mAP50-95=', r['mAP50-95'])
PY

date '+%F %T' > "$SAVE/COMPLETE"
log "========== A1/A2 YOLO COMPLETE =========="
sync
log "[SHUTDOWN] success; calling /usr/bin/shutdown"
sleep 3
/usr/bin/shutdown
