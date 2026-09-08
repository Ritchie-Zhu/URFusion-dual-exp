#!/usr/bin/env bash
# A1/A2 fusion eval: method-specific Y infer -> VIS chroma colorize -> 25 official metrics (8-proc).
# Appends to existing LLVIP/MSRS/M3FD.xlsx. Does not delete Full/B rows.
# Shuts down the AutoDL instance only after a fully successful run.
set -euo pipefail

PROJECT=/root/autodl-tmp/URFusion-main
PYTHON=/root/autodl-tmp/conda/envs/urfusion/bin/python
INFER_A1="$PROJECT/ablation_code/A1_wo_Dual/fusion_gray_infer.py"
INFER_A2="$PROJECT/ablation_code/A2_wo_MoE/fusion_gray_infer.py"
COLOR="$PROJECT/metrics_save/llvip_sample1000/colorize_gray_infer.py"
EVAL="$PROJECT/metrics_save/llvip_sample1000/run_eval_mp.py"
COLOR_XLSX="$PROJECT/metrics_save/new_methods_gifnet_lrrnet_sage/run_color_xlsx.py"
ORCH="$PROJECT/metrics_save/abl_a12_eval"
LOG="$ORCH/pipeline.log"
RESULTS="$PROJECT/vis-ir-gray/results"
SHUTDOWN="${SHUTDOWN:-1}"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

METHODS=(Abl_wo_Dual Abl_wo_MoE)
DATASET_NAMES=(LLVIP MSRS M3FD)
TEST_DIRS=(
	"$PROJECT/datasets/LLVIP/test_sample1000"
	"$PROJECT/datasets/MSRS/test"
	"$PROJECT/datasets/M3FD/test"
)

mkdir -p "$ORCH"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

infer_script_for() {
	case "$1" in
		Abl_wo_Dual) echo "$INFER_A1" ;;
		Abl_wo_MoE) echo "$INFER_A2" ;;
		*) log "ERROR: unknown method $1"; exit 1 ;;
	esac
}

count_images() {
	local dir=$1
	if [[ ! -d "$dir" ]]; then
		echo 0
		return
	fi
	find "$dir" \( -type f -o -type l \) \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.bmp' \) | wc -l
}

expected_count() {
	local test_dir=$1
	local vis="$test_dir/vis"
	[[ -d "$vis" ]] || vis="$test_dir/VIS"
	count_images "$vis"
}

verify_ckpt() {
	local exp=$1
	local ckpt="$PROJECT/vis-ir-gray/train-jobs/ckpt/${exp}/${exp}_ckpt.pth"
	if [[ ! -f "$ckpt" ]]; then
		log "ERROR: missing ckpt $ckpt"
		exit 1
	fi
	log "  ckpt $exp -> $ckpt"
}

infer_color_one() {
	local method=$1 dataset=$2 test_dir=$3 expected=$4
	local infer_py
	infer_py=$(infer_script_for "$method")
	local y_dir="$RESULTS/${method}_${dataset}/Y_fused"
	local rgb_dir="$RESULTS/${method}_${dataset}/RGB_fused"
	local n_y n_rgb

	n_y=$(count_images "$y_dir")
	if [[ "$n_y" -lt "$expected" ]]; then
		log "===== infer ${method} @ ${dataset} (${n_y}/${expected}) ====="
		"$PYTHON" "$infer_py" \
			--experiment "$method" \
			--dataset "$dataset" \
			--test_dir "$test_dir" \
			2>&1 | tee -a "$ORCH/infer_${method}_${dataset}.log"
		n_y=$(count_images "$y_dir")
	else
		log "===== skip infer ${method} @ ${dataset} (Y=${n_y}/${expected}) ====="
	fi
	if [[ "$n_y" -lt "$expected" ]]; then
		log "ERROR: ${method}_${dataset} Y_fused ${n_y}/${expected}"
		exit 1
	fi

	n_rgb=$(count_images "$rgb_dir")
	if [[ "$n_rgb" -lt "$expected" ]]; then
		log "===== colorize ${method} @ ${dataset} (${n_rgb}/${expected}) ====="
		"$PYTHON" "$COLOR" \
			--method "$method" \
			--dataset "$dataset" \
			--test_dir "$test_dir" \
			2>&1 | tee -a "$ORCH/color_${method}_${dataset}.log"
		n_rgb=$(count_images "$rgb_dir")
	else
		log "===== skip colorize ${method} @ ${dataset} (RGB=${n_rgb}/${expected}) ====="
	fi
	if [[ "$n_rgb" -lt "$expected" ]]; then
		log "ERROR: ${method}_${dataset} RGB_fused ${n_rgb}/${expected}"
		exit 1
	fi
	log "  verify ${method}_${dataset}: Y=${n_y} RGB=${n_rgb}"
}

verify_xlsx() {
	"$PYTHON" - "$PROJECT/metrics_save" <<'PY'
import sys
from pathlib import Path
import openpyxl

base = Path(sys.argv[1])
need = ['Abl_wo_Dual', 'Abl_wo_MoE']
keep = ['Fusion_noleak_1', 'Abl_wo_Lint', 'Abl_wo_Lgrad', 'Abl_wo_Lbranch']
for ds in ['LLVIP', 'MSRS', 'M3FD']:
	p = base / f'{ds}.xlsx'
	wb = openpyxl.load_workbook(p, data_only=True)
	ws = wb['MeanMetrics']
	methods = [ws.cell(r, 1).value for r in range(2, ws.max_row + 1)]
	missing = [m for m in need if m not in methods]
	lost = [m for m in keep if m not in methods]
	if lost:
		raise SystemExit(f'{ds}.xlsx lost {lost}')
	if missing:
		raise SystemExit(f'{ds}.xlsx missing {missing}')
	print(f'  {ds}: ok n={len(methods)} last={methods[-2:]}')
PY
}

log "========== Abl A1-A2 EVAL START =========="
avail_kb=$(df -k /root/autodl-tmp | awk 'NR==2 {print $4}')
avail_g=$((avail_kb / 1024 / 1024))
log "disk_avail=${avail_g}G"
if [[ "$avail_kb" -lt $((4 * 1024 * 1024)) ]]; then
	log "ERROR: need >=4G free on /root/autodl-tmp, got ${avail_g}G"
	exit 1
fi

for method in "${METHODS[@]}"; do
	verify_ckpt "$method"
done

declare -a EXPECTED=()
for i in "${!DATASET_NAMES[@]}"; do
	exp=$(expected_count "${TEST_DIRS[$i]}")
	EXPECTED+=("$exp")
	log "dataset ${DATASET_NAMES[$i]} expected=${exp}"
done

for method in "${METHODS[@]}"; do
	for i in "${!DATASET_NAMES[@]}"; do
		infer_color_one "$method" "${DATASET_NAMES[$i]}" "${TEST_DIRS[$i]}" "${EXPECTED[$i]}"
	done
done

log "===== 25-metric eval (8 workers) ====="
"$PYTHON" "$EVAL" \
	--methods "${METHODS[@]}" \
	--datasets "${DATASET_NAMES[@]}" \
	--dataset_roots "${TEST_DIRS[@]}" \
	--results_root "$RESULTS" \
	--metrics_save "$PROJECT/metrics_save" \
	--project_root "$PROJECT" \
	--num_workers 8 \
	2>&1 | tee -a "$ORCH/eval_mp.log"

log "===== color Excel ====="
for ds in "${DATASET_NAMES[@]}"; do
	"$PYTHON" "$COLOR_XLSX" --dataset "$ds" \
		2>&1 | tee -a "$ORCH/color_xlsx_${ds}.log"
done

log "===== verify Excel rows ====="
verify_xlsx | tee -a "$LOG"

date '+%F %T' > "$ORCH/COMPLETE"
log "========== Abl A1-A2 EVAL COMPLETE =========="
sync

if [[ "$SHUTDOWN" == "1" ]]; then
	log "[SHUTDOWN] success; calling /usr/bin/shutdown"
	sleep 3
	/usr/bin/shutdown
else
	log "[SHUTDOWN] skipped SHUTDOWN=$SHUTDOWN"
fi
