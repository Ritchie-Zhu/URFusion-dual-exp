"""Run Text-IF general fusion inference on MSRS test set."""

import argparse
import os

import clip
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F

from model.Text_IF_model import Text_IF as create_model

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

DEFAULT_TEXT = 'This is the infrared and visible light image fusion task.'


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--dataset_path',
		type=str,
		default='/root/autodl-tmp/URFusion-main/Text-IF/dataset/MSRS/eval',
	)
	parser.add_argument(
		'--weights_path',
		type=str,
		default='/root/autodl-tmp/URFusion-main/Text-IF/pretrained_weights/simple_fusion.pth',
	)
	parser.add_argument(
		'--save_path',
		type=str,
		default='/root/autodl-tmp/URFusion-main/vis-ir-gray/results/Text-IF_MSRS/RGB_fused',
	)
	parser.add_argument('--input_text', type=str, default=DEFAULT_TEXT)
	parser.add_argument('--device', type=str, default='cuda')
	parser.add_argument('--limit', type=int, default=0, help='0 = all images')
	return parser.parse_args()


def tensor2numpy(img_tensor):
	img = img_tensor.squeeze(0).cpu().detach().numpy()
	return np.transpose(img, [1, 2, 0])


def save_pic(outputpic, path, name):
	outputpic = np.clip(outputpic, 0.0, 1.0)
	outputpic = cv2.normalize(outputpic, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F)
	outputpic = outputpic[:, :, ::-1]
	stem, _ = os.path.splitext(name)
	out_path = os.path.join(path, f'{stem}.png')
	cv2.imwrite(out_path, outputpic)


def main():
	args = parse_args()
	os.makedirs(args.save_path, exist_ok=True)
	if not os.path.isfile(args.weights_path) or os.path.getsize(args.weights_path) < 1024:
		raise FileNotFoundError(
			f'Missing Text-IF weights: {args.weights_path}. '
			'Download simple_fusion.pth into pretrained_weights/.'
		)

	device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
	supported = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

	visible_root = os.path.join(args.dataset_path, 'Visible')
	infrared_root = os.path.join(args.dataset_path, 'Infrared')
	visible_paths = sorted(
		os.path.join(visible_root, name)
		for name in os.listdir(visible_root)
		if os.path.splitext(name)[-1].lower() in supported
	)
	infrared_paths = sorted(
		os.path.join(infrared_root, name)
		for name in os.listdir(infrared_root)
		if os.path.splitext(name)[-1].lower() in supported
	)
	assert len(visible_paths) == len(infrared_paths), 'VIS/IR count mismatch'
	if args.limit > 0:
		visible_paths = visible_paths[: args.limit]
		infrared_paths = infrared_paths[: args.limit]
	print(f'Found {len(visible_paths)} pairs')

	with torch.no_grad():
		model_clip, _ = clip.load('ViT-B/32', device=device)
		model = create_model(model_clip).to(device)
		ckpt = torch.load(args.weights_path, map_location=device)
		model.load_state_dict(ckpt['model'])
		model.eval()
		text = clip.tokenize(args.input_text).to(device)

		for idx, (vi_path, ir_path) in enumerate(zip(visible_paths, infrared_paths), start=1):
			vi_name = os.path.basename(vi_path)
			ir_name = os.path.basename(ir_path)
			assert vi_name == ir_name, f'pair mismatch: {vi_name} vs {ir_name}'
			ir = Image.open(ir_path).convert('RGB')
			vi = Image.open(vi_path).convert('RGB')
			width, height = vi.size
			new_width = (width // 16) * 16
			new_height = (height // 16) * 16
			ir = ir.resize((new_width, new_height))
			vi = vi.resize((new_width, new_height))
			ir_t = F.to_tensor(ir).unsqueeze(0).to(device)
			vi_t = F.to_tensor(vi).unsqueeze(0).to(device)
			fused = model(vi_t, ir_t, text)
			save_pic(tensor2numpy(fused), args.save_path, vi_name)
			if idx % 20 == 0 or idx == len(visible_paths):
				print(f'[{idx}/{len(visible_paths)}] saved {vi_name}')

	print(f'Done. Results in {args.save_path}')


if __name__ == '__main__':
	main()
