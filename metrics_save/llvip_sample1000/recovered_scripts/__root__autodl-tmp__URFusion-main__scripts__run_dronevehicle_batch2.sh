#!/usr/bin/env bash
set -euo pipefail

PROJECT=/root/autodl-tmp/URFusion-main
PYTHON=/root/autodl-tmp/conda/envs/urfusion/bin/python
TEST_DIR="$PROJECT/datasets/DroneVehicle/test"
RESULTS="$PROJECT/vis-ir-gray/results"
LOG_DIR="$PROJECT/metrics_save/dronevehicle_inference"
mkdir -p "$LOG_DIR" "$RESULTS"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_DIR/batch2_pipeline.log"; }

run_urfusion() {
	log "===== URFusion @ DroneVehicle ====="
	cd "$PROJECT/vis-ir/code"
	staging="$PROJECT/vis-ir/results/URFusion_DroneVehicle_staging"
	rm -rf "$staging"
	mkdir -p "$staging"
	$PYTHON test_M3FD.py \
		--fusion_model fusionnet \
		--fusion_pth ../train-jobs/ckpt/content-fusion-msrs_ckpt.pth \
		--vis_mat ../train-jobs/vis.mat \
		--testDir "$TEST_DIR/" \
		--out_name URFusion_DroneVehicle_staging \
		2>&1 | tee "$LOG_DIR/URFusion.log"
	mkdir -p "$RESULTS/URFusion_DroneVehicle/RGB_fused"
	rsync -a "$staging/" "$RESULTS/URFusion_DroneVehicle/RGB_fused/"
}

run_textif() {
	log "===== Text-IF @ DroneVehicle ====="
	cd "$PROJECT/Text-IF"
	$PYTHON test_dronevehicle.py 2>&1 | tee "$LOG_DIR/Text-IF.log"
}

run_metafusion() {
	log "===== MetaFusion @ DroneVehicle ====="
	cd "$PROJECT/MetaFusion"
	$PYTHON test_m3fd.py \
		--test_ir_root "$TEST_DIR/ir" \
		--test_vis_root "$TEST_DIR/vis" \
		--save_path "$RESULTS/MetaFusion_DroneVehicle/RGB_fused" \
		2>&1 | tee "$LOG_DIR/MetaFusion.log"
}

run_fusion_training_1() {
	log "===== Fusion_training_1 @ DroneVehicle ====="
	cd "$PROJECT/vis-ir-gray/code"
	$PYTHON test_fusion_gray.py \
		--dataset DroneVehicle \
		--test_dir "$TEST_DIR" \
		--output_root "$RESULTS" \
		2>&1 | tee "$LOG_DIR/Fusion_training_1_infer.log"
	$PYTHON colorize_gray.py \
		--method Fusion_training_1 \
		--dataset DroneVehicle \
		--test_dir "$TEST_DIR" \
		--results_root "$RESULTS" \
		2>&1 | tee "$LOG_DIR/Fusion_training_1_colorize.log"
}

log "DroneVehicle batch2 start: URFusion, Text-IF, MetaFusion, Fusion_training_1 (8980 pairs)"
run_urfusion
run_textif
run_metafusion
run_fusion_training_1
log "DroneVehicle batch2 finished"
