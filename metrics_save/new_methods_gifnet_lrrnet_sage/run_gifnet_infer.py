"""GIFNet IVIF inference wrapper — does not modify GIFNet/test.py."""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

PROJECT = '/root/autodl-tmp/URFusion-main'
GIFNET_DIR = os.path.join(PROJECT, 'GIFNet')
ORCH_DIR = os.path.join(PROJECT, 'metrics_save/new_methods_gifnet_lrrnet_sage')
sys.path.insert(0, ORCH_DIR)
sys.path.insert(0, GIFNET_DIR)
os.chdir(GIFNET_DIR)

from GIFNet_model import TwoBranchesFusionNet  # noqa: E402
from args import Args as args  # noqa: E402
from common import list_paired_names, resolve_vis_ir_dirs  # noqa: E402


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', required=True)
	parser.add_argument('--test_dir', required=True)
	parser.add_argument('--output_dir', required=True)
	parser.add_argument('--checkpoint', default=os.path.join(GIFNET_DIR, 'model/Final.model'))
	parser.add_argument('--vis_is_rgb', action='store_true', default=True)
	return parser.parse_args()


def rgb_to_ycbcr(image):
	rgb_array = np.array(image)
	m = np.array([[0.299, 0.587, 0.114], [-0.169, -0.331, 0.5], [0.5, -0.419, -0.081]])
	ycbcr_array = np.dot(rgb_array, m.T)
	return np.clip(ycbcr_array[:, :, 0], 0, 255), ycbcr_array[:, :, 1], ycbcr_array[:, :, 2]


def ycbcr_to_rgb(y, cb, cr):
	ycbcr_array = np.stack((y, cb, cr), axis=-1)
	m = np.array([[1, 0, 1.402], [1, -0.344136, -0.714136], [1, 1.772, 0]])
	rgb_array = np.clip(np.dot(ycbcr_array, m.T), 0, 255)
	return Image.fromarray(np.round(rgb_array).astype(np.uint8), mode='RGB')


def load_model(model_path):
	model = TwoBranchesFusionNet(args.s, args.n, args.channel, args.stride)
	model.load_state_dict(torch.load(model_path, map_location='cpu'))
	model.eval()
	if args.cuda:
		model.cuda()
	return model


def fuse_one(model, ir_path, vis_path, vis_is_rgb):
	transform = transforms.Compose([transforms.ToTensor()])
	if vis_is_rgb:
		vis_img = Image.open(vis_path).convert('RGB')
		vis_y, cb, cr = rgb_to_ycbcr(vis_img)
		vis_tensor = transform(vis_y.astype(np.uint8))
	else:
		cb = cr = None
		vis_tensor = transform(cv2.imread(vis_path, cv2.IMREAD_GRAYSCALE))
	ir_tensor = transform(cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE))
	if args.cuda:
		ir_tensor, vis_tensor = ir_tensor.cuda(), vis_tensor.cuda()
	with torch.no_grad():
		fea_com = model.forward_encoder(ir_tensor.unsqueeze(0).float(), vis_tensor.unsqueeze(0).float())
		fea_fused = model.forward_MultiTask_branch(fea_com_ivif=fea_com, fea_com_mfif=fea_com)
		out_y = model.forward_mixed_decoder(fea_com, fea_fused)
		fused_y = out_y[0, 0].cpu().numpy() * 255.0
	if vis_is_rgb:
		return ycbcr_to_rgb(fused_y, cb, cr)
	return Image.fromarray(fused_y.astype(np.uint8), mode='L')


def main():
	cli = parse_args()
	os.makedirs(cli.output_dir, exist_ok=True)
	vis_dir, ir_dir = resolve_vis_ir_dirs(cli.test_dir)
	names = list_paired_names(cli.test_dir)
	print(f'[GIFNet] {cli.dataset}: {len(names)} pairs -> {cli.output_dir}')
	model = load_model(cli.checkpoint)
	t0 = time.time()
	for name in tqdm(names, desc=f'GIFNet_{cli.dataset}'):
		result = fuse_one(model, os.path.join(ir_dir, name), os.path.join(vis_dir, name), cli.vis_is_rgb)
		out_path = os.path.join(cli.output_dir, name)
		result.save(out_path) if isinstance(result, Image.Image) else cv2.imwrite(out_path, result)
	print(f'[GIFNet DONE] {len(names)} in {time.time()-t0:.1f}s')


if __name__ == '__main__':
	main()
