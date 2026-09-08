#!/usr/bin/env python3
"""Evaluate YOLO11s detector on visible baseline + fusion RGB_fused (test 2000).

Metrics (Ultralytics val, COCO-style):
  - mAP50      : AP @ IoU=0.5
  - mAP50-95   : AP @ IoU=0.5:0.95 (primary COCO metric)
  - precision  : box precision (P)
  - recall     : box recall (R)
  - f1         : 2PR/(P+R)
  - per-class AP50 / AP50-95 for 6 classes

Outputs under metrics_save/yolo_det_yolo_test_2000/
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from openpyxl import Workbook

PROJECT = Path(__file__).resolve().parents[2]
YOLO_ROOT = PROJECT / 'datasets/yolo'
STAGING_ROOT = YOLO_ROOT / 'eval_staging'
DEFAULT_WEIGHTS = YOLO_ROOT / 'runs/yolo11s_iv_det/weights/best.pt'
DEFAULT_SAVE = PROJECT / 'metrics_save/yolo_det_yolo_test_2000'

CLASS_NAMES = ['person', 'car', 'bus', 'lamp', 'motorcycle', 'truck']

METHODS = [
	'Fusion_training_1',
	'MUFusion',
	'U2Fusion',
	'URFusion',
	'MetaFusion',
	'GIFNet',
	'LRRNet',
	'SAGE',
]

INPUTS = ['Visible'] + METHODS

# Human-readable metric descriptions for JSON sidecar
METRIC_INFO = {
	'mAP50': 'Mean AP @ IoU=0.50. Standard detection metric; higher is better.',
	'mAP50-95': 'Mean AP @ IoU=0.50:0.95 (COCO primary). Stricter; higher is better.',
	'precision': 'Precision = TP/(TP+FP) at default val conf. Higher is better.',
	'recall': 'Recall = TP/(TP+FN). Higher is better.',
	'f1': 'Harmonic mean of precision and recall.',
	'AP50_person': 'Per-class AP50 for person (class 0).',
	'AP50_car': 'Per-class AP50 for car (class 1).',
	'AP50_bus': 'Per-class AP50 for bus (class 2).',
	'AP50_lamp': 'Per-class AP50 for lamp (class 3).',
	'AP50_motorcycle': 'Per-class AP50 for motorcycle (class 4).',
	'AP50_truck': 'Per-class AP50 for truck (class 5).',
}


def parse_args():
	parser = argparse.ArgumentParser(description='YOLO detection eval on fusion test set')
	parser.add_argument('--weights', type=Path, default=DEFAULT_WEIGHTS)
	parser.add_argument('--staging_root', type=Path, default=STAGING_ROOT)
	parser.add_argument('--save_dir', type=Path, default=DEFAULT_SAVE)
	parser.add_argument('--inputs', nargs='+', default=INPUTS)
	parser.add_argument('--device', default='0')
	parser.add_argument('--imgsz', type=int, default=640)
	parser.add_argument('--batch', type=int, default=16)
	parser.add_argument('--conf', type=float, default=0.001, help='Val confidence threshold')
	parser.add_argument('--iou', type=float, default=0.7, help='Val NMS IoU')
	parser.add_argument('--prepare', action='store_true', help='Run prepare_eval_staging first')
	parser.add_argument('--subsets', action='store_true', help='Also eval LLVIP-only and M3FD-only staging')
	return parser.parse_args()


def run_prepare(staging_root: Path, subsets: bool) -> None:
	import subprocess
	py = '/root/autodl-tmp/conda/envs/urfusion/bin/python'
	script = PROJECT / 'metrics_save/yolo_det/prepare_eval_staging.py'
	cmd = [py, str(script), '--staging_root', str(staging_root)]
	if subsets:
		cmd.append('--subsets')
	subprocess.check_call(cmd)


def extract_metrics(metrics_obj, n_images: int, input_name: str) -> dict:
	box = metrics_obj.box
	p = float(box.mp)
	r = float(box.mr)
	f1 = 2 * p * r / (p + r + 1e-9)
	out = {
		'input': input_name,
		'n_images': n_images,
		'mAP50': round(float(box.map50), 6),
		'mAP50-95': round(float(box.map), 6),
		'precision': round(p, 6),
		'recall': round(r, 6),
		'f1': round(f1, 6),
	}
	# per-class AP50-95 in box.maps; per-class AP50 may be in results dict
	maps = list(box.maps) if hasattr(box, 'maps') and box.maps is not None else []
	for i, cname in enumerate(CLASS_NAMES):
		key = f'AP50-95_{cname}'
		out[key] = round(float(maps[i]), 6) if i < len(maps) else None
	# ap50 per class from metrics.results_dict if available
	rd = getattr(metrics_obj, 'results_dict', {}) or {}
	for i, cname in enumerate(CLASS_NAMES):
		k = f'metrics/AP50({cname})'
		if k in rd:
			out[f'AP50_{cname}'] = round(float(rd[k]), 6)
	return out


def eval_one(model, yaml_path: Path, args) -> dict:
	staging_dir = yaml_path.parent / yaml_path.stem
	n_images = len(list((staging_dir / 'images' / 'test').glob('*')))
	t0 = time.time()
	metrics = model.val(
		data=str(yaml_path),
		split='val',
		imgsz=args.imgsz,
		batch=args.batch,
		device=args.device,
		conf=args.conf,
		iou=args.iou,
		verbose=False,
		plots=False,
	)
	elapsed = time.time() - t0
	row = extract_metrics(metrics, n_images, yaml_path.stem)
	row['eval_time_s'] = round(elapsed, 2)
	return row


def save_excel(rows: list[dict], path: Path) -> None:
	if not rows:
		return
	keys = [
		'input', 'n_images', 'mAP50', 'mAP50-95', 'precision', 'recall', 'f1',
		'AP50_person', 'AP50_car', 'AP50_bus', 'AP50_lamp', 'AP50_motorcycle', 'AP50_truck',
		'AP50-95_person', 'AP50-95_car', 'AP50-95_bus', 'AP50-95_lamp',
		'AP50-95_motorcycle', 'AP50-95_truck', 'eval_time_s',
	]
	wb = Workbook()
	ws = wb.active
	ws.title = 'DetectionMetrics'
	ws.append([k for k in keys if any(k in r for r in rows)])
	header = ws[1]
	col_keys = [c.value for c in header]
	for row in rows:
		ws.append([row.get(k, '') for k in col_keys])
	path.parent.mkdir(parents=True, exist_ok=True)
	wb.save(path)


def main() -> None:
	args = parse_args()
	if args.prepare:
		run_prepare(args.staging_root, args.subsets)

	if not args.weights.exists():
		raise FileNotFoundError(f'missing weights: {args.weights}')

	from ultralytics import YOLO

	args.save_dir.mkdir(parents=True, exist_ok=True)
	model = YOLO(str(args.weights))

	all_inputs = list(args.inputs)
	if args.subsets:
		for base in list(args.inputs):
			for src in ('LLVIP', 'M3FD'):
				all_inputs.append(f'{base}_{src}')

	rows: list[dict] = []
	for name in all_inputs:
		yaml_path = args.staging_root / f'{name}.yaml'
		if not yaml_path.exists():
			print(f'[SKIP] missing staging yaml: {yaml_path}')
			continue
		print(f'[EVAL] {name} ...')
		row = eval_one(model, yaml_path, args)
		rows.append(row)
		out_json = args.save_dir / f'{name}_det_metrics.json'
		out_json.write_text(json.dumps(row, indent=2) + '\n')
		print(f'  mAP50={row["mAP50"]}  mAP50-95={row["mAP50-95"]}  P={row["precision"]}  R={row["recall"]}')

	summary = {
		'weights': str(args.weights.resolve()),
		'n_inputs': len(rows),
		'metric_descriptions': METRIC_INFO,
		'results': rows,
	}
	(args.save_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

	csv_path = args.save_dir / 'detection_comparison.csv'
	if rows:
		fieldnames = list(rows[0].keys())
		with csv_path.open('w', newline='') as f:
			w = csv.DictWriter(f, fieldnames=fieldnames)
			w.writeheader()
			w.writerows(rows)

	save_excel(rows, args.save_dir / 'detection_comparison.xlsx')
	print(f'[DONE] saved to {args.save_dir}')


if __name__ == '__main__':
	main()
