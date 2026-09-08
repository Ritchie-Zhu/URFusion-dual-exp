#!/usr/bin/env python3
"""Draw YOLO detections on visible / fusion test images for manual inspection.

Examples:
  # Pick 4 test images with many GT boxes from M3FD
  python visualize_yolo_det_samples.py --pick 4 --source M3FD --min_boxes 8

  # Specific yolo names, compare visible + 3 fusion methods
  python visualize_yolo_det_samples.py \\
    --names M3FD_03299 LLVIP_010156 \\
    --methods Visible Fusion_training_1 SAGE U2Fusion

  # Side-by-side grid per image
  python visualize_yolo_det_samples.py --names M3FD_03299 --grid
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
YOLO_ROOT = PROJECT / 'datasets/yolo'
MANIFEST = YOLO_ROOT / 'split_manifest.csv'
FUSION_ROOT = YOLO_ROOT / 'fusion_results'
STAGING_ROOT = YOLO_ROOT / 'eval_staging'
DEFAULT_WEIGHTS = YOLO_ROOT / 'runs/yolo11s_iv_det/weights/best.pt'
DEFAULT_OUT = PROJECT / 'metrics_save/yolo_det_yolo_test_2000/vis_samples'

CLASS_NAMES = ['person', 'car', 'bus', 'lamp', 'motorcycle', 'truck']
CLASS_COLORS = [
	(0, 255, 0),
	(255, 128, 0),
	(0, 128, 255),
	(255, 0, 255),
	(0, 255, 255),
	(128, 0, 255),
]
PRED_COLOR = (0, 0, 255)
GT_COLOR = (0, 255, 0)

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


def parse_args():
	parser = argparse.ArgumentParser(description='Visualize YOLO detections on fusion test images')
	parser.add_argument('--weights', type=Path, default=DEFAULT_WEIGHTS)
	parser.add_argument('--out_dir', type=Path, default=DEFAULT_OUT)
	parser.add_argument('--names', nargs='+', help='Explicit yolo_name list, e.g. M3FD_03299 LLVIP_010156')
	parser.add_argument('--pick', type=int, default=0, help='Randomly pick N test images (use with --source / --min_boxes)')
	parser.add_argument('--source', choices=['LLVIP', 'M3FD'], help='Filter test images by dataset source')
	parser.add_argument('--min_boxes', type=int, default=1, help='Minimum GT box count when picking samples')
	parser.add_argument('--seed', type=int, default=42)
	parser.add_argument('--methods', nargs='+', default=['Visible', 'Fusion_training_1', 'SAGE', 'U2Fusion'])
	parser.add_argument('--conf', type=float, default=0.25, help='Predict confidence threshold for visualization')
	parser.add_argument('--device', default='0')
	parser.add_argument('--draw_gt', action='store_true', default=True, help='Draw GT boxes in green')
	parser.add_argument('--no_draw_gt', action='store_false', dest='draw_gt')
	parser.add_argument('--grid', action='store_true', help='Also save one horizontal grid per image')
	parser.add_argument('--staging_root', type=Path, default=STAGING_ROOT)
	return parser.parse_args()


def load_test_rows(source: str | None = None) -> list[dict]:
	rows = []
	with MANIFEST.open() as f:
		for row in csv.DictReader(f):
			if row['split'] != 'test':
				continue
			if source and row['source'] != source:
				continue
			rows.append(row)
	return rows


def pick_rows(args) -> list[dict]:
	if args.names:
		by_name = {r['yolo_name']: r for r in load_test_rows()}
		missing = [n for n in args.names if n not in by_name]
		if missing:
			raise KeyError(f'unknown or non-test yolo_name: {missing}')
		return [by_name[n] for n in args.names]

	rows = load_test_rows(args.source)
	rows = [r for r in rows if int(r['num_boxes']) >= args.min_boxes]
	if not rows:
		raise RuntimeError('no test rows match filters')
	rng = random.Random(args.seed)
	if args.pick <= 0:
		raise ValueError('provide --names or --pick N')
	if args.pick >= len(rows):
		return rows
	return rng.sample(rows, args.pick)


def staging_image_path(method: str, row: dict, staging_root: Path) -> Path:
	ext = Path(row['image_src']).suffix.lower()
	return staging_root / method / 'images' / 'test' / f"{row['yolo_name']}{ext}"


def label_path(row: dict) -> Path:
	return YOLO_ROOT / 'labels' / 'test' / f"{row['yolo_name']}.txt"


def read_image(path: Path) -> np.ndarray:
	img = cv2.imread(str(path))
	if img is None:
		raise FileNotFoundError(f'cannot read image: {path}')
	return img


def load_yolo_labels(label_file: Path, w: int, h: int) -> list[tuple[int, int, int, int, int]]:
	boxes = []
	if not label_file.exists():
		return boxes
	for line in label_file.read_text().splitlines():
		parts = line.strip().split()
		if len(parts) != 5:
			continue
		cls_id = int(float(parts[0]))
		cx, cy, bw, bh = map(float, parts[1:])
		x1 = int((cx - bw / 2) * w)
		y1 = int((cy - bh / 2) * h)
		x2 = int((cx + bw / 2) * w)
		y2 = int((cy + bh / 2) * h)
		boxes.append((cls_id, x1, y1, x2, y2))
	return boxes


def draw_box(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, color, label: str, thickness: int = 2):
	x1, y1 = max(0, x1), max(0, y1)
	x2, y2 = min(img.shape[1] - 1, x2), min(img.shape[0] - 1, y2)
	cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
	if label:
		(text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
		y_text = max(y1, text_h + 4)
		cv2.rectangle(img, (x1, y_text - text_h - 4), (x1 + text_w + 4, y_text + baseline), color, -1)
		cv2.putText(img, label, (x1 + 2, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_gt(img: np.ndarray, row: dict) -> np.ndarray:
	out = img.copy()
	h, w = out.shape[:2]
	for cls_id, x1, y1, x2, y2 in load_yolo_labels(label_path(row), w, h):
		name = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else str(cls_id)
		draw_box(out, x1, y1, x2, y2, GT_COLOR, f'GT:{name}', thickness=2)
	return out


def draw_predictions(model, img: np.ndarray, conf: float, device: str) -> np.ndarray:
	out = img.copy()
	results = model.predict(source=out, conf=conf, device=device, verbose=False)[0]
	if results.boxes is None or len(results.boxes) == 0:
		return out
	for box in results.boxes:
		x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
		cls_id = int(box.cls[0])
		score = float(box.conf[0])
		name = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else str(cls_id)
		draw_box(out, x1, y1, x2, y2, PRED_COLOR, f'{name}:{score:.2f}', thickness=2)
	return out


def annotate_method(model, method: str, row: dict, staging_root: Path, args) -> np.ndarray:
	img_path = staging_image_path(method, row, staging_root)
	img = read_image(img_path)
	if args.draw_gt:
		img = draw_gt(img, row)
	return draw_predictions(model, img, args.conf, args.device)


def add_title(img: np.ndarray, title: str) -> np.ndarray:
	bar_h = 28
	out = np.zeros((img.shape[0] + bar_h, img.shape[1], 3), dtype=np.uint8)
	out[bar_h:, :] = img
	cv2.putText(out, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
	return out


def make_grid(tiles: list[np.ndarray]) -> np.ndarray:
	h = max(t.shape[0] for t in tiles)
	padded = []
	for tile in tiles:
		if tile.shape[0] != h:
			scale = h / tile.shape[0]
			tile = cv2.resize(tile, (int(tile.shape[1] * scale), h))
		padded.append(tile)
	return np.hstack(padded)


def ensure_staging(methods: list[str], staging_root: Path) -> None:
	missing = [m for m in methods if not (staging_root / m / 'images' / 'test').exists()]
	if not missing:
		return
	import subprocess
	py = '/root/autodl-tmp/conda/envs/urfusion/bin/python'
	script = PROJECT / 'metrics_save/yolo_det/prepare_eval_staging.py'
	subprocess.check_call([py, str(script), '--staging_root', str(staging_root)])


def main() -> None:
	args = parse_args()
	for m in args.methods:
		if m not in INPUTS:
			raise ValueError(f'unknown method/input: {m}')

	if not args.weights.exists():
		raise FileNotFoundError(f'missing weights: {args.weights}')

	ensure_staging(args.methods, args.staging_root)
	rows = pick_rows(args)
	args.out_dir.mkdir(parents=True, exist_ok=True)

	from ultralytics import YOLO

	model = YOLO(str(args.weights))
	manifest_out = args.out_dir / 'selected_samples.csv'
	with manifest_out.open('w', newline='') as f:
		writer = csv.DictWriter(f, fieldnames=['yolo_name', 'source', 'original_stem', 'num_boxes'])
		writer.writeheader()
		for row in rows:
			writer.writerow({k: row[k] for k in writer.fieldnames})

	print(f'[INFO] output -> {args.out_dir}')
	print(f'[INFO] samples: {[r["yolo_name"] for r in rows]}')

	for row in rows:
		sample_dir = args.out_dir / row['yolo_name']
		sample_dir.mkdir(parents=True, exist_ok=True)
		tiles = []
		for method in args.methods:
			annotated = annotate_method(model, method, row, args.staging_root, args)
			title = f'{method} | {row["yolo_name"]} ({row["source"]}, GT={row["num_boxes"]})'
			titled = add_title(annotated, title)
			out_path = sample_dir / f'{method}_det.jpg'
			cv2.imwrite(str(out_path), titled)
			print(f'  saved {out_path}')
			tiles.append(titled)

		if args.grid and len(tiles) > 1:
			grid = make_grid(tiles)
			grid_path = sample_dir / 'compare_grid.jpg'
			cv2.imwrite(str(grid_path), grid)
			print(f'  saved {grid_path}')

	print(f'[DONE] manifest: {manifest_out}')


if __name__ == '__main__':
	main()
