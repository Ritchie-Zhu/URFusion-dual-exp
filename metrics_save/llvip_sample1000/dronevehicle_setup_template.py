"""
DroneVehicle 1000-image fair subsample experiment setup.

1. Delete prior DroneVehicle metrics artifacts
2. Random sample 1000 paired names (fixed seed) from full test set
3. Trim 6 completed methods' fused outputs to sample only
4. Create test_sample1000/ vis+ir symlinks for U2Fusion inference

Usage:
  python scripts/dronevehicle_sample1000_setup.py
"""

import json
import os
import random
import shutil
from datetime import datetime
from pathlib import Path

PROJECT = Path('/root/autodl-tmp/URFusion-main')
TEST_DIR = PROJECT / 'datasets/DroneVehicle/test'
VIS_DIR = TEST_DIR / 'vis'
IR_DIR = TEST_DIR / 'ir'
RESULTS = PROJECT / 'vis-ir-gray/results'
METRICS_SAVE = PROJECT / 'metrics_save'
LOG_DIR = METRICS_SAVE / 'dronevehicle_sample1000'

SEED = 42
N_SAMPLE = 1000
IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

METHODS_JPG = [
	'MUFusion',
	'URFusion',
	'Fusion_training_1',
	'MetaFusion',
	'EMMA',
]
METHODS_PNG = ['Text-IF']
ALL_METHODS = METHODS_JPG + METHODS_PNG + ['U2Fusion']


def log(msg: str, lines: list):
	line = f'[{datetime.now().strftime("%F %T")}] {msg}'
	print(line)
	lines.append(line)


def list_paired_source_names():
	names = []
	for fn in sorted(os.listdir(VIS_DIR)):
		if os.path.splitext(fn)[1].lower() not in IMAGE_EXTS:
			continue
		if os.path.isfile(IR_DIR / fn):
			names.append(fn)
	return names


def delete_metrics_artifacts(lines: list):
	log('=== Step 1: Delete prior DroneVehicle metrics ===', lines)
	removed = []
	for path in [
		METRICS_SAVE / 'DroneVehicle.xlsx',
		METRICS_SAVE / 'DroneVehicle_colored.xlsx',
	]:
		if path.is_file():
			path.unlink()
			removed.append(str(path))
	for d in METRICS_SAVE.glob('*_DroneVehicle'):
		if d.is_dir():
			shutil.rmtree(d)
			removed.append(str(d))
	log(f'Removed {len(removed)} metrics paths', lines)
	for p in removed:
		log(f'  deleted: {p}', lines)


def sample_names(lines: list):
	log('=== Step 2: Random sample 1000 pairs (seed=42) ===', lines)
	all_names = list_paired_source_names()
	log(f'Full test set: {len(all_names)} paired images', lines)
	if len(all_names) < N_SAMPLE:
		raise RuntimeError(f'Need at least {N_SAMPLE} pairs, found {len(all_names)}')

	rng = random.Random(SEED)
	chosen = sorted(rng.sample(all_names, N_SAMPLE))
	stems = {os.path.splitext(n)[0] for n in chosen}

	manifest = {
		'dataset': 'DroneVehicle',
		'seed': SEED,
		'n_sample': N_SAMPLE,
		'n_total': len(all_names),
		'created_at': datetime.now().isoformat(),
		'sampling': 'random.sample without replacement, sorted for reproducibility',
		'names': chosen,
	}
	LOG_DIR.mkdir(parents=True, exist_ok=True)
	manifest_path = LOG_DIR / 'sample_1000_seed42.json'
	with open(manifest_path, 'w', encoding='utf-8') as f:
		json.dump(manifest, f, indent=2)
	log(f'Manifest saved: {manifest_path}', lines)
	log(f'First 5: {chosen[:5]}', lines)
	log(f'Last 5: {chosen[-5:]}', lines)
	return chosen, stems, manifest_path


def trim_fused_dir(method: str, keep_names: set, keep_stems: set, lines: list):
	rgb_dir = RESULTS / f'{method}_DroneVehicle' / 'RGB_fused'
	if not rgb_dir.is_dir():
		log(f'  {method}: RGB_fused missing, skip trim', lines)
		return 0, 0

	kept = 0
	deleted = 0
	for fn in os.listdir(rgb_dir):
		fp = rgb_dir / fn
		if not fp.is_file():
			continue
		stem = os.path.splitext(fn)[0]
		keep = fn in keep_names or stem in keep_stems
		if keep:
			kept += 1
		else:
			fp.unlink()
			deleted += 1
	log(f'  {method} RGB_fused: kept {kept}, deleted {deleted}', lines)
	return kept, deleted


def trim_y_fused(keep_names: set, lines: list):
	y_dir = RESULTS / 'Fusion_training_1_DroneVehicle' / 'Y_fused'
	if not y_dir.is_dir():
		log('  Fusion_training_1 Y_fused: missing, skip', lines)
		return
	kept, deleted = 0, 0
	for fn in os.listdir(y_dir):
		fp = y_dir / fn
		if not fp.is_file():
			continue
		if fn in keep_names:
			kept += 1
		else:
			fp.unlink()
			deleted += 1
	log(f'  Fusion_training_1 Y_fused: kept {kept}, deleted {deleted}', lines)


def clear_u2fusion(lines: list):
	u2_dir = RESULTS / 'U2Fusion_DroneVehicle' / 'RGB_fused'
	if u2_dir.is_dir():
		n = len(list(u2_dir.iterdir()))
		shutil.rmtree(u2_dir)
		log(f'  U2Fusion: removed entire RGB_fused ({n} files) for fresh 1000-run', lines)
	u2_dir.mkdir(parents=True, exist_ok=True)


def create_sample_test_dir(chosen: list, lines: list):
	log('=== Step 4: Create test_sample1000 symlink subset ===', lines)
	sample_root = PROJECT / 'datasets/DroneVehicle/test_sample1000'
	vis_out = sample_root / 'vis'
	ir_out = sample_root / 'ir'
	if sample_root.exists():
		shutil.rmtree(sample_root)
	vis_out.mkdir(parents=True)
	ir_out.mkdir(parents=True)

	for name in chosen:
		os.symlink(VIS_DIR / name, vis_out / name)
		os.symlink(IR_DIR / name, ir_out / name)

	log(f'Created {sample_root} with {len(chosen)} vis/ir symlinks', lines)
	return sample_root


def verify_trim(chosen: list, lines: list):
	log('=== Step 5: Verify counts ===', lines)
	keep_names = set(chosen)
	for m in ALL_METHODS:
		d = RESULTS / f'{m}_DroneVehicle' / 'RGB_fused'
		n = len(os.listdir(d)) if d.is_dir() else 0
		ok = 'OK' if m == 'U2Fusion' and n == 0 else ('OK' if n == N_SAMPLE else 'MISMATCH')
		log(f'  {m}: {n} files [{ok}]', lines)


def main():
	lines = []
	log('DroneVehicle sample-1000 setup start', lines)

	delete_metrics_artifacts(lines)
	chosen, stems, manifest_path = sample_names(lines)
	keep_names = set(chosen)

	log('=== Step 3: Trim fused outputs to sample only ===', lines)
	for m in METHODS_JPG:
		trim_fused_dir(m, keep_names, stems, lines)
	trim_fused_dir('Text-IF', set(), stems, lines)  # .png by stem
	trim_y_fused(keep_names, lines)
	clear_u2fusion(lines)

	sample_root = create_sample_test_dir(chosen, lines)
	verify_trim(chosen, lines)

	log_path = LOG_DIR / 'setup.log'
	with open(log_path, 'w', encoding='utf-8') as f:
		f.write('\n'.join(lines) + '\n')
	log(f'Setup log: {log_path}', lines)
	log('Setup complete.', lines)
	return manifest_path, sample_root


if __name__ == '__main__':
	main()
