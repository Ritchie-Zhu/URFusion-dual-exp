"""Run EgeFusion on MSRS and save RGB_fused for metrics."""

import os
import time

import cv2
import numpy as np
from natsort import natsorted
from tqdm import tqdm

from egefusion_py import fuse_gray

VIS_ROOT = '/root/autodl-tmp/URFusion-main/datasets/MSRS/test/vis'
IR_ROOT = '/root/autodl-tmp/URFusion-main/datasets/MSRS/test/ir'
OUTPUT_DIR = '/root/autodl-tmp/URFusion-main/our_model_1_DualMoE/results/EgeFusion_MSRS/RGB_fused'


def rgb_to_ycbcr(img_bgr: np.ndarray):
	img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float64)
	t = np.array([
		[0.299, 0.587, 0.114],
		[-0.169, -0.331, 0.5],
		[0.5, -0.419, -0.081],
	])
	ycbcr = img @ t.T
	return ycbcr[:, :, 0] / 255.0, ycbcr[:, :, 1], ycbcr[:, :, 2]


def ycbcr_to_bgr(y01: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
	y = np.clip(y01 * 255.0, 0, 255)
	ycbcr = np.stack([y, cb, cr], axis=-1)
	t = np.array([
		[1, 0, 1.402],
		[1, -0.344136, -0.714136],
		[1, 1.772, 0],
	])
	rgb = ycbcr @ t.T
	rgb = np.clip(rgb, 0, 255).astype(np.uint8)
	return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def main():
	os.makedirs(OUTPUT_DIR, exist_ok=True)
	names = natsorted(
		f for f in os.listdir(VIS_ROOT)
		if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
		and os.path.isfile(os.path.join(IR_ROOT, f))
	)
	print(f'EgeFusion MSRS: {len(names)} pairs')

	t0 = time.time()
	for name in tqdm(names, desc='EgeFusion_MSRS'):
		vis = cv2.imread(os.path.join(VIS_ROOT, name), cv2.IMREAD_COLOR)
		ir = cv2.imread(os.path.join(IR_ROOT, name), cv2.IMREAD_GRAYSCALE)
		if vis is None or ir is None:
			continue
		if ir.shape[:2] != vis.shape[:2]:
			ir = cv2.resize(ir, (vis.shape[1], vis.shape[0]))

		_, cb, cr = rgb_to_ycbcr(vis)
		y_fused = fuse_gray(ir, vis)
		out = ycbcr_to_bgr(y_fused, cb, cr)
		stem, _ = os.path.splitext(name)
		cv2.imwrite(os.path.join(OUTPUT_DIR, f'{stem}.png'), out)

	print(f'Done. {len(names)} images in {time.time() - t0:.1f}s -> {OUTPUT_DIR}')


if __name__ == '__main__':
	main()
