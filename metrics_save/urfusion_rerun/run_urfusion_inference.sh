#!/usr/bin/env bash
# URFusion inference on LLVIP / MSRS / M3FD using official ckpt + vis.mat.
set -euo pipefail

PROJECT=/root/autodl-tmp/URFusion-main
PYTHON=/root/autodl-tmp/conda/envs/urfusion/bin/python
URFUSION_CODE="$PROJECT/URFusion/vis-ir/code"
CKPT="$PROJECT/URFusion/vis-ir/train-jobs/ckpt/content-fusion_ckpt.pth"
VIS_MAT="$PROJECT/URFusion/vis-ir/train-jobs/vis.mat"
RESULTS="$PROJECT/our_model_1_DualMoE/results"
LOG_DIR="$PROJECT/metrics_save/urfusion_rerun"
LOG="$LOG_DIR/urfusion_inference.log"

export CUDA_VISIBLE_DEVICES=0

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

verify_assets() {
	for f in "$CKPT" "$VIS_MAT" "$PROJECT/URFusion/vis-ir/train-jobs/ckpt/A2V_ckpt.pth"; do
		[[ -f "$f" ]] || { log "ERROR: missing $f"; exit 1; }
	done
	log "ckpt: $CKPT"
	log "vis.mat: $VIS_MAT"
}

count_out() {
	local d=$1
	find "$d" -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.bmp' \) 2>/dev/null | wc -l
}

run_dataset() {
	local dataset=$1
	local test_dir=$2
	local expected=$3
	local staging="$RESULTS/URFusion_${dataset}_staging"
	local out_dir="$RESULTS/URFusion_${dataset}/RGB_fused"

	log "===== URFusion @ ${dataset} ====="
	rm -rf "$staging"
	mkdir -p "$staging" "$out_dir"

	cd "$URFUSION_CODE"
	$PYTHON test.py \
		--device 0 \
		--testDir "$test_dir" \
		--outputDir "$staging/" \
		--fusion_ckpt content-fusion \
		--A2V_ckpt A2V \
		2>&1 | tee -a "$LOG_DIR/URFusion_${dataset}.log"

	rsync -a "$staging/" "$out_dir/"
	local n
	n=$(count_out "$out_dir")
	log "  ${dataset}: ${n}/${expected} -> $out_dir"
	if [[ "$n" -lt "$expected" ]]; then
		log "ERROR: ${dataset} incomplete (${n}/${expected})"
		exit 1
	fi
}

log "========== URFusion INFERENCE (official ckpt) START =========="
verify_assets

run_dataset LLVIP "$PROJECT/datasets/LLVIP/test_sample1000" 1000
run_dataset MSRS "$PROJECT/datasets/MSRS/test" 361
run_dataset M3FD "$PROJECT/datasets/M3FD/test" 300

log "========== URFusion INFERENCE COMPLETE =========="
for ds in LLVIP MSRS M3FD; do
	log "  ${ds}: $(count_out "$RESULTS/URFusion_${ds}/RGB_fused")"
done
