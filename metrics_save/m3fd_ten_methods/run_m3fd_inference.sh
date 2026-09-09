#!/usr/bin/env bash
# M3FD inference for 10 methods (same wrappers as LLVIP/MSRS/DroneVehicle).
set -euo pipefail

PROJECT=/root/autodl-tmp/URFusion-main
PYTHON=/root/autodl-tmp/conda/envs/urfusion/bin/python
ORCH="$PROJECT/metrics_save/m3fd_ten_methods"
LLVIP_ORCH="$PROJECT/metrics_save/llvip_sample1000"
NEW_ORCH="$PROJECT/metrics_save/new_methods_gifnet_lrrnet_sage"
TEST_DIR="$PROJECT/datasets/M3FD/test"
RESULTS="$PROJECT/our_model_1_DualMoE/results"
LOG="$ORCH/m3fd_inference.log"

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

log "========== M3FD INFERENCE (10 methods) START =========="

log "===== MUFusion @ M3FD ====="
prepare_out MUFusion
$PYTHON "$LLVIP_ORCH/run_pyc_inference.py" \
	--work_dir "$PROJECT/MUFusion/ir_vis" \
	-- --dataset M3FD --test_dir "$TEST_DIR" \
	--output_dir "$RESULTS/MUFusion_M3FD/RGB_fused" \
	2>&1 | tee -a "$ORCH/MUFusion.log"
log "  MUFusion: $(count_rgb MUFusion) files"

log "===== U2Fusion @ M3FD ====="
prepare_out U2Fusion
$PYTHON "$LLVIP_ORCH/run_pyc_inference.py" \
	--work_dir "$PROJECT/U2Fusion" \
	-- --dataset M3FD --test_dir "$TEST_DIR" \
	--output_dir "$RESULTS/U2Fusion_M3FD/RGB_fused" \
	2>&1 | tee -a "$ORCH/U2Fusion.log"
log "  U2Fusion: $(count_rgb U2Fusion) files"

log "===== URFusion @ M3FD ====="
prepare_out URFusion
STAGING="$PROJECT/vis-ir/results/URFusion_M3FD_staging"
rm -rf "$STAGING"
cd "$PROJECT/vis-ir/code"
$PYTHON test_M3FD.py \
	--fusion_model fusionnet \
	--fusion_pth ../train-jobs/ckpt/content-fusion-msrs_ckpt.pth \
	--vis_mat ../train-jobs/vis.mat \
	--testDir "$TEST_DIR/" \
	--out_name URFusion_M3FD_staging \
	2>&1 | tee -a "$ORCH/URFusion.log"
rsync -a "$STAGING/" "$RESULTS/URFusion_M3FD/RGB_fused/"
log "  URFusion: $(count_rgb URFusion) files"

log "===== Fusion_training_1 @ M3FD ====="
rm -rf "$RESULTS/Fusion_training_1_M3FD/Y_fused" "$RESULTS/Fusion_training_1_M3FD/RGB_fused"
$PYTHON "$LLVIP_ORCH/fusion_gray_infer.py" \
	--dataset M3FD --test_dir "$TEST_DIR" \
	2>&1 | tee -a "$ORCH/Fusion_training_1_infer.log"
$PYTHON "$LLVIP_ORCH/colorize_gray_infer.py" \
	--dataset M3FD --test_dir "$TEST_DIR" \
	2>&1 | tee -a "$ORCH/Fusion_training_1_colorize.log"
log "  Fusion_training_1: $(count_rgb Fusion_training_1) files"

log "===== MetaFusion @ M3FD ====="
prepare_out MetaFusion
$PYTHON "$LLVIP_ORCH/run_pyc_inference.py" \
	--work_dir "$PROJECT/MetaFusion" \
	-- --test_ir_root "$TEST_DIR/ir" \
	--test_vis_root "$TEST_DIR/vis" \
	--save_path "$RESULTS/MetaFusion_M3FD/RGB_fused" \
	2>&1 | tee -a "$ORCH/MetaFusion.log"
log "  MetaFusion: $(count_rgb MetaFusion) files"

log "===== EMMA @ M3FD ====="
prepare_out EMMA
$PYTHON "$LLVIP_ORCH/run_emma_infer.py" \
	--dataset M3FD --test_dir "$TEST_DIR" \
	--output_dir "$RESULTS/EMMA_M3FD/RGB_fused" \
	2>&1 | tee -a "$ORCH/EMMA.log"
log "  EMMA: $(count_rgb EMMA) files"

log "===== Text-IF @ M3FD ====="
prepare_out Text-IF
cd "$PROJECT/Text-IF"
$PYTHON test_m3fd.py \
	--save_path "$RESULTS/Text-IF_M3FD/RGB_fused" \
	2>&1 | tee -a "$ORCH/Text-IF.log"
log "  Text-IF: $(count_rgb Text-IF) files"

log "===== GIFNet @ M3FD ====="
prepare_out GIFNet
$PYTHON "$NEW_ORCH/run_gifnet_infer.py" \
	--dataset M3FD --test_dir "$TEST_DIR" \
	--output_dir "$RESULTS/GIFNet_M3FD/RGB_fused" \
	2>&1 | tee -a "$ORCH/GIFNet.log"
log "  GIFNet: $(count_rgb GIFNet) files"

log "===== LRRNet @ M3FD ====="
prepare_out LRRNet
$PYTHON "$NEW_ORCH/run_lrrnet_infer.py" \
	--dataset M3FD --test_dir "$TEST_DIR" \
	--output_dir "$RESULTS/LRRNet_M3FD/RGB_fused" \
	--force \
	2>&1 | tee -a "$ORCH/LRRNet.log"
log "  LRRNet: $(count_rgb LRRNet) files"

log "===== SAGE @ M3FD ====="
prepare_out SAGE
$PYTHON "$NEW_ORCH/run_sage_infer.py" \
	--dataset M3FD --test_dir "$TEST_DIR" \
	--output_dir "$RESULTS/SAGE_M3FD/RGB_fused" \
	2>&1 | tee -a "$ORCH/SAGE.log"
log "  SAGE: $(count_rgb SAGE) files"

log "========== M3FD INFERENCE COMPLETE =========="
for m in MUFusion U2Fusion URFusion Fusion_training_1 MetaFusion EMMA Text-IF GIFNet LRRNet SAGE; do
	log "  $m: $(count_rgb "$m")"
done
