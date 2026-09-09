#!/usr/bin/env bash
# Resume M3FD from MetaFusion failure. Skip completed inference/metrics.
set -euo pipefail

PROJECT=/root/autodl-tmp/URFusion-main
PYTHON=/root/autodl-tmp/conda/envs/urfusion/bin/python
ORCH="$PROJECT/metrics_save/m3fd_ten_methods"
LLVIP_ORCH="$PROJECT/metrics_save/llvip_sample1000"
NEW_ORCH="$PROJECT/metrics_save/new_methods_gifnet_lrrnet_sage"
EVAL="$LLVIP_ORCH/run_eval_from_cache.py"
COLOR="$NEW_ORCH/run_color_xlsx.py"
TEST_DIR="$PROJECT/datasets/M3FD/test"
RESULTS="$PROJECT/our_model_1_DualMoE/results"
LOG="$ORCH/m3fd_resume.log"

export CUDA_VISIBLE_DEVICES=0

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

count_rgb() {
	local method=$1
	find "$RESULTS/${method}_M3FD/RGB_fused" -type f 2>/dev/null | wc -l
}

prepare_out() {
	local method=$1
	rm -rf "$RESULTS/${method}_M3FD/RGB_fused"
	mkdir -p "$RESULTS/${method}_M3FD/RGB_fused"
}

run_if_needed() {
	local method=$1
	local expected=$2
	local n
	n=$(count_rgb "$method")
	if [[ "$n" -ge "$expected" ]]; then
		log "SKIP inference ${method}: already ${n}/${expected}"
		return 0
	fi
	log "RUN inference ${method}: ${n}/${expected}"
	return 1
}

N=$(find "$TEST_DIR/vis" -type f | wc -l)
log "========== M3FD RESUME (expect ${N} pairs) =========="
log "Existing metrics (skip): U2Fusion URFusion Fusion_training_1"

# --- Inference: MetaFusion -> SAGE only ---
if ! run_if_needed MetaFusion "$N"; then
	prepare_out MetaFusion
	cd "$PROJECT/MetaFusion"
	$PYTHON test.py \
		--test_ir_root "$TEST_DIR/ir" \
		--test_vis_root "$TEST_DIR/vis" \
		--save_path "$RESULTS/MetaFusion_M3FD/RGB_fused" \
		2>&1 | tee -a "$ORCH/MetaFusion.log"
	log "  MetaFusion: $(count_rgb MetaFusion) files"
fi

if ! run_if_needed EMMA "$N"; then
	prepare_out EMMA
	$PYTHON "$LLVIP_ORCH/run_emma_infer.py" \
		--dataset M3FD --test_dir "$TEST_DIR" \
		--output_dir "$RESULTS/EMMA_M3FD/RGB_fused" \
		2>&1 | tee -a "$ORCH/EMMA.log"
	log "  EMMA: $(count_rgb EMMA) files"
fi

if ! run_if_needed Text-IF "$N"; then
	prepare_out Text-IF
	cd "$PROJECT/Text-IF"
	$PYTHON test_m3fd.py \
		--save_path "$RESULTS/Text-IF_M3FD/RGB_fused" \
		2>&1 | tee -a "$ORCH/Text-IF.log"
	log "  Text-IF: $(count_rgb Text-IF) files"
fi

if ! run_if_needed GIFNet "$N"; then
	prepare_out GIFNet
	$PYTHON "$NEW_ORCH/run_gifnet_infer.py" \
		--dataset M3FD --test_dir "$TEST_DIR" \
		--output_dir "$RESULTS/GIFNet_M3FD/RGB_fused" \
		2>&1 | tee -a "$ORCH/GIFNet.log"
	log "  GIFNet: $(count_rgb GIFNet) files"
fi

if ! run_if_needed LRRNet "$N"; then
	prepare_out LRRNet
	$PYTHON "$NEW_ORCH/run_lrrnet_infer.py" \
		--dataset M3FD --test_dir "$TEST_DIR" \
		--output_dir "$RESULTS/LRRNet_M3FD/RGB_fused" \
		--force \
		2>&1 | tee -a "$ORCH/LRRNet.log"
	log "  LRRNet: $(count_rgb LRRNet) files"
fi

if ! run_if_needed SAGE "$N"; then
	prepare_out SAGE
	$PYTHON "$NEW_ORCH/run_sage_infer.py" \
		--dataset M3FD --test_dir "$TEST_DIR" \
		--output_dir "$RESULTS/SAGE_M3FD/RGB_fused" \
		2>&1 | tee -a "$ORCH/SAGE.log"
	log "  SAGE: $(count_rgb SAGE) files"
fi

log "Inference summary:"
for m in MUFusion U2Fusion URFusion Fusion_training_1 MetaFusion EMMA Text-IF GIFNet LRRNet SAGE; do
	log "  $m: $(count_rgb "$m")"
done

# --- Metrics: only methods without metrics_summary.json ---
PENDING=()
for m in MUFusion MetaFusion EMMA Text-IF GIFNet LRRNet SAGE; do
	if [[ -f "$PROJECT/metrics_save/${m}_M3FD/metrics_summary.json" ]]; then
		log "SKIP metrics ${m}: summary exists"
	else
		PENDING+=("$m")
	fi
done

if [[ ${#PENDING[@]} -eq 0 ]]; then
	log "All pending metrics already done."
else
	log "===== eval_multi_time @ M3FD (${#PENDING[@]} methods) ====="
	for m in "${PENDING[@]}"; do
		n=$(count_rgb "$m")
		if [[ "$n" -lt "$N" ]]; then
			log "ERROR: ${m}_M3FD has ${n}/${N} fused images"
			exit 1
		fi
		log "  verify ${m}: ${n}/${N}"
	done

	$PYTHON "$EVAL" \
		--methods "${PENDING[@]}" \
		--datasets M3FD \
		--dataset_roots "$TEST_DIR" \
		--results_root "$RESULTS" \
		--metrics_save "$PROJECT/metrics_save" \
		--project_root "$PROJECT" \
		2>&1 | tee -a "$ORCH/eval_M3FD_resume.log"
fi

log "===== color_methods_metrics @ M3FD ====="
$PYTHON "$COLOR" --dataset M3FD \
	2>&1 | tee -a "$ORCH/color_M3FD.log"

log "  Excel: $PROJECT/metrics_save/M3FD.xlsx"
log "  Colored: $PROJECT/metrics_save/M3FD_colored.xlsx"
log "========== M3FD RESUME COMPLETE =========="
