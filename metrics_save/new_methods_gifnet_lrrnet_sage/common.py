"""Shared helpers for GIFNet / LRRNet / SAGE orchestration (metrics_save only)."""

import os

PROJECT = '/root/autodl-tmp/URFusion-main'
PYTHON = '/root/autodl-tmp/conda/envs/urfusion/bin/python'
RESULTS_ROOT = os.path.join(PROJECT, 'vis-ir-gray/results')
METRICS_SAVE = os.path.join(PROJECT, 'metrics_save')
ORCH_DIR = os.path.join(METRICS_SAVE, 'new_methods_gifnet_lrrnet_sage')

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
NEW_METHODS = ('GIFNet', 'LRRNet', 'SAGE')

DATASET_TEST_DIRS = {
	'LLVIP': os.path.join(PROJECT, 'datasets/LLVIP/test_sample1000'),
	'MSRS': os.path.join(PROJECT, 'datasets/MSRS/test'),
	'DroneVehicle': os.path.join(PROJECT, 'datasets/DroneVehicle/test_sample1000'),
}


def resolve_vis_ir_dirs(test_dir):
	test_dir = os.path.abspath(test_dir)
	for vis_name, ir_name in (('vis', 'ir'), ('VIS', 'IR'), ('Visible', 'Infrared')):
		vis_dir = os.path.join(test_dir, vis_name)
		ir_dir = os.path.join(test_dir, ir_name)
		if os.path.isdir(vis_dir) and os.path.isdir(ir_dir):
			return vis_dir, ir_dir
	raise FileNotFoundError(f'Cannot find vis/ir pair under {test_dir}')


def list_paired_names(test_dir):
	vis_dir, ir_dir = resolve_vis_ir_dirs(test_dir)
	return sorted(
		fn
		for fn in os.listdir(vis_dir)
		if os.path.splitext(fn)[1].lower() in IMAGE_EXTS
		and os.path.isfile(os.path.join(ir_dir, fn))
	)


def rgb_fused_dir(method, dataset):
	return os.path.join(RESULTS_ROOT, f'{method}_{dataset}', 'RGB_fused')


def sage_staging_dir(dataset):
	return os.path.join(ORCH_DIR, 'sage_staging', dataset)
