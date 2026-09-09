"""Run EMMA on DroneVehicle and save fused images for metrics."""

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
from test_msrs import fuse_one

PROJECT_ROOT = '/root/autodl-tmp/URFusion-main'
VIS_ROOT = os.path.join(PROJECT_ROOT, 'datasets/DroneVehicle/test/vis')
IR_ROOT = os.path.join(PROJECT_ROOT, 'datasets/DroneVehicle/test/ir')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'our_model_1_DualMoE/results/EMMA_DroneVehicle/RGB_fused')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'EMMA.pth')


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
	print(f'EMMA DroneVehicle: {len(names)} pairs -> {OUTPUT_DIR}')

	t0 = time.time()
	for name in tqdm(names, desc='EMMA_DroneVehicle'):
		fused = fuse_one(model, device, os.path.join(IR_ROOT, name), os.path.join(VIS_ROOT, name))
		cv2.imwrite(
			os.path.join(OUTPUT_DIR, name),
			np.clip(fused, 0, 255).astype(np.uint8),
		)

	elapsed = time.time() - t0
	print(f'Done. {len(names)} images in {elapsed:.1f}s ({elapsed / len(names):.3f}s/img)')


if __name__ == '__main__':
	main()
