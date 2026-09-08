"""Run EMMA on MSRS and save fused images for metrics."""

import os
import sys
import time

import cv2
import numpy as np
import torch
from natsort import natsorted
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nets.Ufuser import Ufuser
from utils import image_read_cv2

VIS_ROOT = '/root/autodl-tmp/URFusion-main/datasets/MSRS/test/vis'
IR_ROOT = '/root/autodl-tmp/URFusion-main/datasets/MSRS/test/ir'
OUTPUT_DIR = '/root/autodl-tmp/URFusion-main/vis-ir-gray/results/EMMA_MSRS/RGB_fused'
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'EMMA.pth')


def fuse_one(model, device, ir_path, vi_path):
	IR = image_read_cv2(ir_path, 'GRAY')[np.newaxis, np.newaxis, ...] / 255
	VI = image_read_cv2(vi_path, 'GRAY')[np.newaxis, np.newaxis, ...] / 255

	h, w = IR.shape[2:]
	h1 = h - h % 32
	w1 = w - w % 32
	h2 = h % 32
	w2 = w % 32

	with torch.no_grad():
		if h1 == h and w1 == w:
			ir = torch.FloatTensor(IR).to(device)
			vi = torch.FloatTensor(VI).to(device)
			data_fuse = model(ir, vi)
			data_fuse = (data_fuse - torch.min(data_fuse)) / (torch.max(data_fuse) - torch.min(data_fuse))
			return np.squeeze((data_fuse * 255).cpu().numpy())

		fused_temp = np.zeros((h, w), dtype=np.float32)
		ir_t = torch.FloatTensor(IR)[:, :, :h1, :w1].to(device)
		vi_t = torch.FloatTensor(VI)[:, :, :h1, :w1].to(device)
		data_fuse = model(ir_t, vi_t)
		fused_temp[:h1, :w1] = np.squeeze((data_fuse * 255).cpu().numpy())

		if w1 != w:
			ir_t = torch.FloatTensor(IR)[:, :, :h1, -w1:].to(device)
			vi_t = torch.FloatTensor(VI)[:, :, :h1, -w1:].to(device)
			data_fuse = model(ir_t, vi_t)
			fused_image = np.squeeze((data_fuse * 255).cpu().numpy())
			fused_temp[:h1, -w2:] = fused_image[:, -w2:]

		if h1 != h:
			ir_t = torch.FloatTensor(IR)[:, :, -h1:, :w1].to(device)
			vi_t = torch.FloatTensor(VI)[:, :, -h1:, :w1].to(device)
			data_fuse = model(ir_t, vi_t)
			fused_image = np.squeeze((data_fuse * 255).cpu().numpy())
			fused_temp[-h2:, :w1] = fused_image[-h2:, :]

		if h1 != h and w1 != w:
			ir_t = torch.FloatTensor(IR)[:, :, -h1:, -w1:].to(device)
			vi_t = torch.FloatTensor(VI)[:, :, -h1:, -w1:].to(device)
			data_fuse = model(ir_t, vi_t)
			fused_image = np.squeeze((data_fuse * 255).cpu().numpy())
			fused_temp[-h2:, -w2:] = fused_image[-h2:, -w2:]

		fused_temp = (fused_temp - np.min(fused_temp)) / (np.max(fused_temp) - np.min(fused_temp))
		return fused_temp * 255


def main():
	os.makedirs(OUTPUT_DIR, exist_ok=True)
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	model = Ufuser().to(device)
	model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
	model.eval()

	names = natsorted(
		f for f in os.listdir(VIS_ROOT)
		if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
		and os.path.isfile(os.path.join(IR_ROOT, f))
	)
	print(f'EMMA MSRS: {len(names)} pairs -> {OUTPUT_DIR}')

	t0 = time.time()
	for name in tqdm(names, desc='EMMA_MSRS'):
		ir_path = os.path.join(IR_ROOT, name)
		vi_path = os.path.join(VIS_ROOT, name)
		fused = fuse_one(model, device, ir_path, vi_path)
		stem, _ = os.path.splitext(name)
		out_path = os.path.join(OUTPUT_DIR, f'{stem}.png')
		cv2.imwrite(out_path, np.clip(fused, 0, 255).astype(np.uint8))

	elapsed = time.time() - t0
	print(f'Done. {len(names)} images in {elapsed:.1f}s ({elapsed / len(names):.2f}s/img)')

if __name__ == '__main__':
	main()
