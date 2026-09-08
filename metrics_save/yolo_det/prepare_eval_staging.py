#!/usr/bin/env python3
"""Build YOLO val staging: fusion/visible images renamed to yolo_name + test labels."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
YOLO_ROOT = PROJECT / 'datasets/yolo'
MANIFEST = YOLO_ROOT / 'split_manifest.csv'
FUSION_ROOT = YOLO_ROOT / 'fusion_results'
STAGING_ROOT = YOLO_ROOT / 'eval_staging'

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


def load_test_rows(source_filter: str | None = None) -> list[dict]:
	rows = []
	with MANIFEST.open() as f:
		for row in csv.DictReader(f):
			if row['split'] != 'test':
				continue
			if source_filter and row['source'] != source_filter:
				continue
			rows.append(row)
	return rows


def image_ext(row: dict) -> str:
	return Path(row['image_src']).suffix.lower()


def fusion_image_path(method: str, row: dict) -> Path:
	ext = image_ext(row)
	fn = f"{row['original_stem']}{ext}"
	return FUSION_ROOT / method / 'RGB_fused' / fn


def visible_image_path(row: dict) -> Path:
	ext = image_ext(row)
	return YOLO_ROOT / 'images' / 'test' / f"{row['yolo_name']}{ext}"


def label_path(row: dict) -> Path:
	return YOLO_ROOT / 'labels' / 'test' / f"{row['yolo_name']}.txt"


def build_staging(name: str, image_resolver, rows: list[dict], out_root: Path) -> int:
	staging = out_root / name
	if staging.exists():
		shutil.rmtree(staging)
	img_dir = staging / 'images' / 'test'
	lbl_dir = staging / 'labels' / 'test'
	img_dir.mkdir(parents=True)
	lbl_dir.mkdir(parents=True)

	n = 0
	for row in rows:
		yolo_name = row['yolo_name']
		ext = image_ext(row)
		img_src = image_resolver(row)
		lbl_src = label_path(row)
		if not img_src.exists():
			raise FileNotFoundError(f'missing image: {img_src}')
		if not lbl_src.exists():
			raise FileNotFoundError(f'missing label: {lbl_src}')
		dst_img = img_dir / f'{yolo_name}{ext}'
		dst_lbl = lbl_dir / f'{yolo_name}.txt'
		dst_img.symlink_to(img_src.resolve())
		dst_lbl.symlink_to(lbl_src.resolve())
		n += 1
	return n


def write_yaml(staging: Path, yaml_path: Path) -> None:
	content = f"""# Auto-generated for detection eval: {staging.name}
path: {staging.resolve()}
train: images/test
val: images/test
test: images/test
nc: 6
names:
  0: person
  1: car
  2: bus
  3: lamp
  4: motorcycle
  5: truck
"""
	yaml_path.write_text(content)


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument('--staging_root', type=Path, default=STAGING_ROOT)
	parser.add_argument('--methods', nargs='+', default=METHODS)
	parser.add_argument('--include_visible', action='store_true', default=True)
	parser.add_argument('--subsets', action='store_true', help='Also build LLVIP-only and M3FD-only staging')
	args = parser.parse_args()

	args.staging_root.mkdir(parents=True, exist_ok=True)
	all_rows = load_test_rows()
	print(f'test rows: {len(all_rows)}')

	if args.include_visible:
		n = build_staging('Visible', visible_image_path, all_rows, args.staging_root)
		write_yaml(args.staging_root / 'Visible', args.staging_root / 'Visible.yaml')
		print(f'Visible: {n} pairs')

	for method in args.methods:
		n = build_staging(
			method,
			lambda row, m=method: fusion_image_path(m, row),
			all_rows,
			args.staging_root,
		)
		write_yaml(args.staging_root / method, args.staging_root / f'{method}.yaml')
		print(f'{method}: {n} pairs')

	if args.subsets:
		for source in ('LLVIP', 'M3FD'):
			sub_rows = load_test_rows(source_filter=source)
			for base in ['Visible'] + list(args.methods):
				if base == 'Visible':
					resolver = visible_image_path
				else:
					resolver = lambda row, m=base: fusion_image_path(m, row)
				name = f'{base}_{source}'
				n = build_staging(name, resolver, sub_rows, args.staging_root)
				write_yaml(args.staging_root / name, args.staging_root / f'{name}.yaml')
				print(f'{name}: {n} pairs')

	print(f'staging root: {args.staging_root}')


if __name__ == '__main__':
	main()
