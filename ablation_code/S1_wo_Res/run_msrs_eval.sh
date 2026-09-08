#!/usr/bin/env bash
# S1 Abl_SR_wo_Res: MSRS infer -> VIS chroma colorize -> official 25-col (run_eval_mp, 8 workers).
# Outputs stay under ablation_code/S1_wo_Res. Does not write official metrics_save/*.xlsx.
# Shuts down the instance only after a fully successful run (SHUTDOWN=0 to skip).
set -euo pipefail

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PROJECT=/root/autodl-tmp/URFusion-main
PY=/root/autodl-tmp/conda/envs/urfusion/bin/python
ROOT=$PROJECT/ablation_code/S1_wo_Res
EXP="${EXP:-Abl_SR_wo_Res}"
SHUTDOWN="${SHUTDOWN:-1}"
RESULTS=$ROOT/results
EVAL_SAVE=$ROOT/eval_save
LOGDIR=$ROOT/train-jobs/console_logs
INFER=$ROOT/fusion_gray_infer.py
COLOR=$PROJECT/metrics_save/llvip_sample1000/colorize_gray_infer.py
EVAL=$PROJECT/metrics_save/llvip_sample1000/run_eval_mp.py
CKPT_ROOT=$ROOT/train-jobs/ckpt
CKPT=$CKPT_ROOT/$EXP/${EXP}_ckpt.pth
FULL_JSON=$PROJECT/vis-ir-gray-moe_update/eval_save/SpatialResMoE_3_MSRS/metrics_summary.json

DATASET=MSRS
TEST_DIR=$PROJECT/datasets/MSRS/test

mkdir -p "$LOGDIR" "$RESULTS" "$EVAL_SAVE"
LOG=$LOGDIR/${EXP}_MSRS_eval.log
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

count_images() {
	local dir=$1
	[[ -d "$dir" ]] || { echo 0; return; }
	find "$dir" \( -type f -o -type l \) \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.bmp' \) | wc -l
}

[[ -f "$CKPT" ]] || { log "ERROR: missing ckpt $CKPT"; exit 1; }
expected=$(count_images "$TEST_DIR/vis")
[[ "$expected" -gt 0 ]] || expected=$(count_images "$TEST_DIR/VIS")
[[ "$expected" -gt 0 ]] || { log "ERROR: no vis in $TEST_DIR"; exit 1; }

log "========== MSRS EVAL START exp=$EXP =========="
log "ckpt=$CKPT"
log "dataset=MSRS expected=$expected"
log "eval=run_eval_mp official evaluation_one 8 workers"
log "SHUTDOWN=$SHUTDOWN"

y_dir=$RESULTS/${EXP}_${DATASET}/Y_fused
rgb_dir=$RESULTS/${EXP}_${DATASET}/RGB_fused

n_y=$(count_images "$y_dir")
if [[ "$n_y" -lt "$expected" ]]; then
	log "===== infer Y ${DATASET} ($n_y/$expected) ====="
	"$PY" "$INFER" \
		--experiment "$EXP" \
		--dataset "$DATASET" \
		--test_dir "$TEST_DIR" \
		--ckpt_root "$CKPT_ROOT" \
		--output_root "$RESULTS" \
		--y_out_dir "$y_dir" \
		2>&1 | tee -a "$LOG"
	n_y=$(count_images "$y_dir")
else
	log "===== skip infer ${DATASET} (Y=$n_y/$expected) ====="
fi
[[ "$n_y" -ge "$expected" ]] || { log "ERROR: ${DATASET} Y_fused $n_y/$expected"; exit 1; }

n_rgb=$(count_images "$rgb_dir")
if [[ "$n_rgb" -lt "$expected" ]]; then
	log "===== colorize VIS Cb/Cr ${DATASET} ($n_rgb/$expected) ====="
	"$PY" "$COLOR" \
		--method "$EXP" \
		--dataset "$DATASET" \
		--test_dir "$TEST_DIR" \
		--results_root "$RESULTS" \
		--y_dir "$y_dir" \
		--rgb_out_dir "$rgb_dir" \
		2>&1 | tee -a "$LOG"
	n_rgb=$(count_images "$rgb_dir")
else
	log "===== skip colorize ${DATASET} (RGB=$n_rgb/$expected) ====="
fi
[[ "$n_rgb" -ge "$expected" ]] || { log "ERROR: ${DATASET} RGB_fused $n_rgb/$expected"; exit 1; }
log "  verify ${DATASET}: Y=$n_y RGB=$n_rgb"

log "===== official evaluation_one (8 workers) ====="
"$PY" "$EVAL" \
	--methods "$EXP" \
	--datasets "$DATASET" \
	--dataset_roots "$TEST_DIR" \
	--results_root "$RESULTS" \
	--metrics_save "$EVAL_SAVE" \
	--project_root "$PROJECT" \
	--num_workers 8 \
	2>&1 | tee -a "$LOG"

summary=$EVAL_SAVE/${EXP}_${DATASET}/metrics_summary.json
[[ -f "$summary" ]] || { log "ERROR: missing $summary"; exit 1; }

log "===== 6-col + 25-col ${DATASET} vs SpatialResMoE_3 ====="
"$PY" - "$summary" "$FULL_JSON" "$EVAL_SAVE/${EXP}_${DATASET}" <<'PY' | tee -a "$LOG"
import json, os, sys

SIX = ['NMI', 'Qy', 'MI', 'VIF', 'Qabf', 'VIFF']
NAMES = [
	'CE', 'NMI', 'QNCIE', 'TE', 'EI', 'Qy', 'Qcb', 'EN', 'MI', 'SF', 'AG', 'SD',
	'CC', 'SCD', 'VIF', 'MSE', 'PSNR', 'Qabf', 'Nabf', 'SSIM', 'MS_SSIM', 'VIFF',
	'NIQE', 'BRISQUE', 'MUSIQ',
]
LOWER_BETTER = {'CE', 'MSE', 'Nabf', 'NIQE', 'BRISQUE'}

ours = json.load(open(sys.argv[1], encoding='utf-8'))
full = json.load(open(sys.argv[2], encoding='utf-8'))
out_dir = sys.argv[3]
n = int(ours.get('n_images', 0))
lines = [f'MSRS n={n}  Abl_SR_wo_Res vs SpatialResMoE_3 (Full)']
lines.append('--- 6-col (higher better) ---')
lines.append(f"{'metric':>12}  {'S1':>10}  {'Full':>10}  {'delta':>10}")
six_lines = list(lines)
for m in SIX:
	a = float(ours[f'{m}_mean'])
	b = float(full[f'{m}_mean'])
	row = f"{m:>12}  {a:10.4f}  {b:10.4f}  {a-b:+10.4f}"
	lines.append(row)
	six_lines.append(row)
lines.append('--- 25-col ---')
lines.append(f"{'metric':>12}  {'S1':>10}  {'Full':>10}  {'delta':>10}  note")
s1_better = 0
for m in NAMES:
	a = float(ours[f'{m}_mean'])
	b = float(full[f'{m}_mean'])
	d = a - b
	if m in LOWER_BETTER:
		win = d < 0
		note = 'lower better'
	else:
		win = d > 0
		note = 'higher better'
	if win:
		s1_better += 1
	mark = 'S1' if win else 'Full'
	lines.append(f"{m:>12}  {a:10.4f}  {b:10.4f}  {d:+10.4f}  {note}  {mark}")
lines.append(f'S1 better on {s1_better}/25 cells')
text = '\n'.join(lines)
print(text)
open(os.path.join(out_dir, 'six_col.txt'), 'w', encoding='utf-8').write('\n'.join(six_lines) + '\n')
open(os.path.join(out_dir, 'all25.txt'), 'w', encoding='utf-8').write(text + '\n')
PY

date '+%F %T' > "$EVAL_SAVE/${EXP}_MSRS_COMPLETE"
log "========== MSRS EVAL COMPLETE ====="
log "results=$RESULTS/${EXP}_${DATASET}"
log "eval_save=$EVAL_SAVE/${EXP}_${DATASET}"

sync
if [[ "$SHUTDOWN" == "1" ]]; then
	log "[SHUTDOWN] success; calling /usr/bin/shutdown"
	sleep 3
	/usr/bin/shutdown
else
	log "[SHUTDOWN] skipped SHUTDOWN=$SHUTDOWN"
fi
