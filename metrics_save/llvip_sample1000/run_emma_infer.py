"""EMMA inference wrapper."""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
from tqdm import tqdm

PROJECT = '/root/autodl-tmp/URFusion-main'
EMMA_DIR = os.path.join(PROJECT, 'EMMA')
sys.path.insert(0, EMMA_DIR)

from nets.Ufuser import Ufuser
from test_msrs import fuse_one

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', default='M3FD')
	parser.add_argument('--test_dir', required=True)
	parser.add_argument('--output_dir', required=True)
	parser.add_argument('--model_path', default=os.path.join(EMMA_DIR, 'model', 'EMMA.pth'))
	return parser.parse_args()


def list_names(test_dir):
	vis_dir = os.path.join(test_dir, 'vis')
	ir_dir = os.path.join(test_dir, 'ir')
	return sorted(
		fn
		for fn in os.listdir(vis_dir)
		if os.path.splitext(fn)[1].lower() in IMAGE_EXTS
		and os.path.isfile(os.path.join(ir_dir, fn))
	)


def main():
	args = parse_args()
	os.makedirs(args.output_dir, exist_ok=True)
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	model = Ufuser().to(device)
	model.load_state_dict(torch.load(args.model_path, map_location=device))
	model.eval()
	names = list_names(args.test_dir)
	print(f'EMMA {args.dataset}: {len(names)} pairs -> {args.output_dir}')
	t0 = time.time()
	for name in tqdm(names, desc=f'EMMA_{args.dataset}'):
		out_path = os.path.join(args.output_dir, name)
		fused = fuse_one(model, device, os.path.join(args.test_dir, 'ir', name), os.path.join(args.test_dir, 'vis', name))
		cv2.imwrite(out_path, np.clip(fused, 0, 255).astype(np.uint8))
	print(f'Done in {time.time()-t0:.1f}s')


if __name__ == '__main__':
	main()
