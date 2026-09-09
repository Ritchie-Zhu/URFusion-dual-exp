"""Colorize Fusion_training_1 Y_fused using VIS chroma (Cb/Cr)."""

import argparse
import os
import sys
import time

import torch
import torchvision.utils
from PIL import Image
from torchvision import transforms

PROJECT = '/root/autodl-tmp/URFusion-main'
CODE_DIR = os.path.join(PROJECT, 'our_model_1_DualMoE/code')
sys.path.insert(0, CODE_DIR)
os.chdir(CODE_DIR)

from ycbcr import rgb_to_ycbcr_tensor, ycbcr_to_rgb_tensor

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--method', type=str, default='Fusion_training_1')
	parser.add_argument('--dataset', type=str, default='M3FD')
	parser.add_argument('--test_dir', type=str, required=True)
	parser.add_argument('--results_root', type=str, default=os.path.join(PROJECT, 'our_model_1_DualMoE/results'))
	parser.add_argument('--y_dir', type=str, default=None, help='Override Y_fused input directory')
	parser.add_argument('--rgb_out_dir', type=str, default=None, help='Override RGB_fused output directory')
	parser.add_argument('--device', type=int, default=0)
	return parser.parse_args()


def list_pairs(test_dir):
	vis_dir = os.path.join(test_dir, 'vis')
	ir_dir = os.path.join(test_dir, 'ir')
	if not os.path.isdir(vis_dir):
		vis_dir = os.path.join(test_dir, 'VIS')
		ir_dir = os.path.join(test_dir, 'IR')
	names = sorted(
		fn
		for fn in os.listdir(vis_dir)
		if os.path.splitext(fn)[1].lower() in IMAGE_EXTS
		and os.path.isfile(os.path.join(ir_dir, fn))
	)
	return vis_dir, names


def main():
	args = parse_args()
	device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
	vis_dir, names = list_pairs(args.test_dir)
	tag = f'{args.method}_{args.dataset}'
	if args.y_dir:
		y_dir = args.y_dir
	else:
		y_dir = os.path.join(args.results_root, tag, 'Y_fused')
	if args.rgb_out_dir:
		rgb_dir = args.rgb_out_dir
	else:
		rgb_dir = os.path.join(args.results_root, tag, 'RGB_fused')
	os.makedirs(rgb_dir, exist_ok=True)

	to_tensor = transforms.ToTensor()
	t0 = time.time()
	with torch.no_grad():
		for name in names:
			stem, _ = os.path.splitext(name)
			y_path = os.path.join(y_dir, name)
			if not os.path.isfile(y_path):
				y_path = os.path.join(y_dir, f'{stem}.png')
			vis_rgb = to_tensor(Image.open(os.path.join(vis_dir, name)).convert('RGB')).unsqueeze(0).to(device)
			_, cb, cr = rgb_to_ycbcr_tensor(vis_rgb)
			y_f = to_tensor(Image.open(y_path).convert('L')).unsqueeze(0).to(device)
			rgb = ycbcr_to_rgb_tensor(y_f, cb, cr)
			torchvision.utils.save_image(rgb, os.path.join(rgb_dir, name))
	print(f'[DONE] {len(names)} RGB_fused in {time.time()-t0:.1f}s -> {rgb_dir}')


if __name__ == '__main__':
	main()
