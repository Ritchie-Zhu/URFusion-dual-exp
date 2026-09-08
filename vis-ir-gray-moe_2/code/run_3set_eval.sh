#!/usr/bin/env bash
# SpatialResMoE_2: infer -> VIS chroma colorize -> official 25-col.
# Order: MSRS, M3FD, LLVIP. Outputs under vis-ir-gray-moe_2.
# Does not write official metrics_save/*.xlsx.
# Shuts down only after a fully successful run (SHUTDOWN=0 to skip).
set -euo pipefail

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PROJECT=/root/autodl-tmp/URFusion-main
PY=/root/autodl-tmp/conda/envs/urfusion/bin/python
ROOT=$PROJECT/vis-ir-gray-moe_2
CODE=$ROOT/code
EXP="${EXP:-SpatialResMoE_2}"
SHUTDOWN="${SHUTDOWN:-1}"
RESULTS=$ROOT/results
EVAL_SAVE=$ROOT/eval_save
LOGDIR=$ROOT/train-jobs/console_logs
INFER=$CODE/fusion_gray_infer.py
COLOR=$PROJECT/metrics_save/llvip_sample1000/colorize_gray_infer.py
EVAL=$PROJECT/metrics_save/llvip_sample1000/run_eval_mp.py
COMPARE=$CODE/compare_eval.py
CKPT_ROOT=$ROOT/train-jobs/ckpt
CKPT=$CKPT_ROOT/$EXP/${EXP}_ckpt.pth
S1_ROOT=$PROJECT/ablation_code/S1_wo_Res/eval_save
V3_ROOT=$PROJECT/vis-ir-gray-moe_update/eval_save

DATASET_NAMES=(MSRS M3FD LLVIP)
TEST_DIRS=(
	"$PROJECT/datasets/MSRS/test"
	"$PROJECT/datasets/M3FD/test"
	"$PROJECT/datasets/LLVIP/test_sample1000"
)

mkdir -p "$LOGDIR" "$RESULTS" "$EVAL_SAVE"
LOG=$LOGDIR/${EXP}_eval.log
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

count_images() {
	local dir=$1
	[[ -d "$dir" ]] || { echo 0; return; }
	find "$dir" \( -type f -o -type l \) \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.bmp' \) | wc -l
}

expected_count() {
	local test_dir=$1
	local vis="$test_dir/vis"
	[[ -d "$vis" ]] || vis="$test_dir/VIS"
	count_images "$vis"
}

[[ -f "$CKPT" ]] || { log "ERROR: missing ckpt $CKPT"; exit 1; }

declare -a EXPECTED=()
for i in "${!DATASET_NAMES[@]}"; do
	n=$(expected_count "${TEST_DIRS[$i]}")
	[[ "$n" -gt 0 ]] || { log "ERROR: no vis in ${TEST_DIRS[$i]}"; exit 1; }
	EXPECTED+=("$n")
done

log "========== 3-SET EVAL START exp=$EXP =========="
log "ckpt=$CKPT"
log "order=MSRS=${EXPECTED[0]} M3FD=${EXPECTED[1]} LLVIP=${EXPECTED[2]}"
log "eval=run_eval_mp official evaluation_one 8 workers"
log "SHUTDOWN=$SHUTDOWN"

infer_color_one() {
	local dataset=$1 test_dir=$2 expected=$3
	local y_dir=$RESULTS/${EXP}_${dataset}/Y_fused
	local rgb_dir=$RESULTS/${EXP}_${dataset}/RGB_fused
	local n_y n_rgb

	n_y=$(count_images "$y_dir")
	if [[ "$n_y" -lt "$expected" ]]; then
		log "===== infer Y ${dataset} ($n_y/$expected) ====="
		"$PY" "$INFER" \
			--experiment "$EXP" \
			--dataset "$dataset" \
			--test_dir "$test_dir" \
			--ckpt_root "$CKPT_ROOT" \
			--output_root "$RESULTS" \
			--y_out_dir "$y_dir" \
			2>&1 | tee -a "$LOG"
		n_y=$(count_images "$y_dir")
	else
		log "===== skip infer ${dataset} (Y=$n_y/$expected) ====="
	fi
	[[ "$n_y" -ge "$expected" ]] || { log "ERROR: ${dataset} Y_fused $n_y/$expected"; exit 1; }

	n_rgb=$(count_images "$rgb_dir")
	if [[ "$n_rgb" -lt "$expected" ]]; then
		log "===== colorize VIS Cb/Cr ${dataset} ($n_rgb/$expected) ====="
		"$PY" "$COLOR" \
			--method "$EXP" \
			--dataset "$dataset" \
			--test_dir "$test_dir" \
			--results_root "$RESULTS" \
			--y_dir "$y_dir" \
			--rgb_out_dir "$rgb_dir" \
			2>&1 | tee -a "$LOG"
		n_rgb=$(count_images "$rgb_dir")
	else
		log "===== skip colorize ${dataset} (RGB=$n_rgb/$expected) ====="
	fi
	[[ "$n_rgb" -ge "$expected" ]] || { log "ERROR: ${dataset} RGB_fused $n_rgb/$expected"; exit 1; }
	log "  verify ${dataset}: Y=$n_y RGB=$n_rgb"
}

eval_one() {
	local dataset=$1 test_dir=$2
	log "===== official evaluation_one ${dataset} (8 workers) ====="
	"$PY" "$EVAL" \
		--methods "$EXP" \
		--datasets "$dataset" \
		--dataset_roots "$test_dir" \
		--results_root "$RESULTS" \
		--metrics_save "$EVAL_SAVE" \
		--project_root "$PROJECT" \
		--num_workers 8 \
		2>&1 | tee -a "$LOG"

	local summary=$EVAL_SAVE/${EXP}_${dataset}/metrics_summary.json
	[[ -f "$summary" ]] || { log "ERROR: missing $summary"; exit 1; }

	local s1=$S1_ROOT/Abl_SR_wo_Res_${dataset}/metrics_summary.json
	if [[ -f "$s1" ]]; then
		log "===== 6-col + 25-col ${dataset} vs S1 ====="
		"$PY" "$COMPARE" \
			--ours "$summary" \
			--ref "$s1" \
			--ours_name SpatialResMoE_2 \
			--ref_name S1 \
			--dataset "$dataset" \
			--out_dir "$EVAL_SAVE/${EXP}_${dataset}" \
			--tag vs_S1 \
			| tee -a "$LOG"
	else
		log "WARN: no S1 json $s1"
	fi

	local v3=$V3_ROOT/SpatialResMoE_3_${dataset}/metrics_summary.json
	if [[ -f "$v3" ]]; then
		log "===== 6-col + 25-col ${dataset} vs SpatialResMoE_3 ====="
		"$PY" "$COMPARE" \
			--ours "$summary" \
			--ref "$v3" \
			--ours_name SpatialResMoE_2 \
			--ref_name v3 \
			--dataset "$dataset" \
			--out_dir "$EVAL_SAVE/${EXP}_${dataset}" \
			--tag vs_v3 \
			| tee -a "$LOG"
	fi
}

for i in "${!DATASET_NAMES[@]}"; do
	infer_color_one "${DATASET_NAMES[$i]}" "${TEST_DIRS[$i]}" "${EXPECTED[$i]}"
	eval_one "${DATASET_NAMES[$i]}" "${TEST_DIRS[$i]}"
done

date '+%F %T' > "$EVAL_SAVE/${EXP}_COMPLETE"
log "========== 3-SET EVAL COMPLETE ====="
log "eval_save=$EVAL_SAVE"

sync
if [[ "$SHUTDOWN" == "1" ]]; then
	log "[SHUTDOWN] success; calling /usr/bin/shutdown"
	sleep 3
	/usr/bin/shutdown
else
	log "[SHUTDOWN] skipped SHUTDOWN=$SHUTDOWN"
fi
