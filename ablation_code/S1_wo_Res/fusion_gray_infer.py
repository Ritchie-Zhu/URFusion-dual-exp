"""Gray-Y inference for S1 Abl_SR_wo_Res. Does not load C."""

import argparse
import os
import sys
import time

import cv2
import torch
import torchvision.utils
from PIL import Image
from torchvision import transforms

ABLATION_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(ABLATION_DIR, '..', '..'))
BASE_CODE = os.path.join(PROJECT, 'vis-ir-gray', 'code')
sys.path.insert(0, BASE_CODE)
sys.path.insert(0, ABLATION_DIR)

from base_only_fusion import DualSpatialBaseOnlyNetGray
from utils import resolve_latest_ckpt_path
from ycbcr import rgb_to_ycbcr_tensor

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
DEFAULT_CKPT_ROOT = os.path.join(ABLATION_DIR, 'train-jobs', 'ckpt')
DEFAULT_OUTPUT_ROOT = os.path.join(ABLATION_DIR, 'results')


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str, default='M3FD')
	parser.add_argument('--test_dir', type=str, required=True)
	parser.add_argument('--output_root', type=str, default=DEFAULT_OUTPUT_ROOT)
	parser.add_argument('--y_out_dir', type=str, default=None)
	parser.add_argument('--experiment', type=str, default='Abl_SR_wo_Res')
	parser.add_argument('--ckpt_root', type=str, default=DEFAULT_CKPT_ROOT)
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
		if os.path.splitext(fn)[1].lower() in IMAGE_EXTS and os.path.isfile(os.path.join(ir_dir, fn))
	)
	return vis_dir, ir_dir, names


def main():
	args = parse_args()
	device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
	vis_dir, ir_dir, names = list_pairs(args.test_dir)
	out_dir = args.y_out_dir or os.path.join(
		args.output_root, f'{args.experiment}_{args.dataset}', 'Y_fused'
	)
	os.makedirs(out_dir, exist_ok=True)

	ckpt_path = resolve_latest_ckpt_path(args.ckpt_root, args.experiment)
	ckpt = torch.load(ckpt_path, map_location=device)
	model = DualSpatialBaseOnlyNetGray().to(device)
	model.load_state_dict(ckpt['model_F_state_dict'])
	model.eval()

	print(f'[infer] {args.experiment} @ {args.dataset}: {len(names)} pairs')
	print(f'  ckpt={ckpt_path}')
	to_tensor = transforms.ToTensor()
	t0 = time.time()
	with torch.no_grad():
		for name in names:
			vis_rgb = to_tensor(Image.open(os.path.join(vis_dir, name)).convert('RGB')).unsqueeze(0).to(device)
			ir_gray = to_tensor(cv2.imread(os.path.join(ir_dir, name), cv2.IMREAD_GRAYSCALE)).unsqueeze(0).to(device)
			y_vis, _, _ = rgb_to_ycbcr_tensor(vis_rgb)
			y_f, _ = model(y_vis, ir_gray)
			torchvision.utils.save_image(y_f, os.path.join(out_dir, name))
	print(f'[DONE] {len(names)} Y_fused in {time.time() - t0:.1f}s -> {out_dir}')


if __name__ == '__main__':
	main()
