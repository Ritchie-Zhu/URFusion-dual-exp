"""M3FD datasets for gray-Y content extractor training."""

import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ycbcr import rgb_uint8_to_y_uint8


def apply_gamma_high(x):
	result = x / 255.0 ** random.uniform(0.75, 0.95) * 255.0
	return np.uint8(np.clip(result, 0, 255))


def apply_gamma_low(x):
	return np.round(x / 255.0 ** random.uniform(1.05, 1.3) * 255.0).astype(np.uint8)


def ori(x):
	return x


def gaussian_noise(x):
	x = np.float32(x)
	sigma = random.uniform(0.04, 0.2)
	noise = np.random.normal(0.0, sigma, x.shape) * 255.0
	return np.uint8(np.clip(x + noise, 0, 255))


def higher_contrast(image):
	alpha = random.uniform(1.2, 1.5)
	beta = np.mean(image) * (1 - alpha)
	return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def lower_contrast(image):
	alpha = random.uniform(0.5, 0.9)
	beta = np.mean(image) * (1 - alpha)
	return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def _apply_two_view_y_aug(base_y, distortions, dist_range):
	"""Same patch-replacement logic as vis-ir SICE_VIS_pairs / SICE_IR, on uint8 Y [H,W]."""
	img1_r = np.copy(base_y)
	img2_r = np.copy(base_y)
	noise_func = distortions[1]
	h, w = base_y.shape

	x = random.randint(0, h // 3 - 1)
	y = random.randint(0, w // 3 - 1)
	h2 = random.randint(h // 3 * 2, h - 1)
	w2 = random.randint(w // 3 * 2, w - 1)
	dist1 = random.randint(dist_range[0], dist_range[1])
	dist_func1 = distortions[dist1]
	img1 = noise_func(dist_func1(base_y)) if random.randint(0, 5) < 2 else dist_func1(base_y)
	img1_r[x:h2, y:w2] = img1[x:h2, y:w2]

	x = random.randint(0, h // 3 - 1)
	y = random.randint(0, w // 3 - 1)
	h2 = random.randint(h // 3 * 2, h - 1)
	w2 = random.randint(w // 3 * 2, w - 1)
	dist2 = random.randint(dist_range[0], dist_range[1])
	dist_func2 = distortions[dist2]
	img2 = noise_func(dist_func2(base_y)) if random.randint(0, 5) < 2 else dist_func2(base_y)
	img2_r[x:h2, y:w2] = img2[x:h2, y:w2]
	return img1_r, img2_r


class M3FDVisYPairs(Dataset):
	"""
	Train C_vis_gray: RGB -> Y first, then Y-domain degradations (no saturation).
	Outputs img1/img2 as [1, H, W] float tensors in [0, 1] after transform.
	"""

	def __init__(self, img_dir, transform=None):
		self.source_dir = os.path.join(img_dir, 'VIS')
		self.source = [
			im_name
			for im_name in os.listdir(self.source_dir)
			if im_name.split('.')[-1].lower() in ('jpg', 'png', 'bmp')
		]
		self.transform = transform
		# Same photometric set as old VIS extractor dist 2-6 (exclude saturation=7).
		self.distortions = {
			1: gaussian_noise,
			2: apply_gamma_low,
			3: apply_gamma_high,
			4: lower_contrast,
			5: higher_contrast,
			6: ori,
		}

	def __len__(self):
		return len(self.source)

	def __getitem__(self, idx):
		name = self.source[idx]
		rgb = cv2.imread(os.path.join(self.source_dir, name))
		rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
		base_y = rgb_uint8_to_y_uint8(rgb)
		img1_r, img2_r = _apply_two_view_y_aug(base_y, self.distortions, (2, 6))

		if self.transform:
			seed = torch.random.seed()
			torch.random.manual_seed(seed)
			img1_r = self.transform(img1_r)
			torch.random.manual_seed(seed)
			img2_r = self.transform(img2_r)

		return {'name': name, 'img1': img1_r, 'img2': img2_r}


class M3FDFusionTrain(Dataset):
	"""
	Train F_gray: paired VIS RGB + IR gray from M3FD train split.
	Augmentations exclude saturation; Y is extracted in the training loop from RGB.
	"""

	def __init__(self, img_dir, transform=None):
		self.vis_dir = os.path.join(img_dir, 'VIS')
		self.ir_dir = os.path.join(img_dir, 'IR')
		self.source = [
			im_name
			for im_name in os.listdir(self.vis_dir)
			if im_name.split('.')[-1].lower() in ('jpg', 'png', 'bmp')
		]
		self.transform = transform
		self.distortions = {
			1: apply_gamma_low,
			2: lower_contrast,
			3: higher_contrast,
			4: gaussian_noise,
			5: ori,
		}

	def __len__(self):
		return len(self.source)

	def __getitem__(self, idx):
		name = self.source[idx]
		vis_rgb = cv2.imread(os.path.join(self.vis_dir, name))
		vis_rgb = cv2.cvtColor(vis_rgb, cv2.COLOR_BGR2RGB)
		ir_gray = cv2.imread(os.path.join(self.ir_dir, name), cv2.IMREAD_GRAYSCALE)

		h, w = vis_rgb.shape[:2]
		if h < 280 or w < 280:
			vis_rgb = cv2.resize(vis_rgb, (280, 280))
			ir_gray = cv2.resize(ir_gray, (280, 280))

		dist1 = random.randint(1, 5)
		vis_rgb = self.distortions[dist1](vis_rgb)
		dist2 = random.randint(1, 5)
		ir_gray = self.distortions[dist2](ir_gray)

		if self.transform:
			seed = torch.random.seed()
			torch.random.manual_seed(seed)
			vis_rgb = self.transform(vis_rgb)
			torch.random.manual_seed(seed)
			ir_gray = self.transform(ir_gray)

		return {'name': name, 'img1': vis_rgb, 'img2': ir_gray}


class M3FDFusionTrain(Dataset):
	"""
	Train F_gray: paired VIS RGB + IR gray from M3FD train split.
	Augmentations exclude saturation; Y is extracted in the training loop from RGB.
	"""

	def __init__(self, img_dir, transform=None):
		self.vis_dir = os.path.join(img_dir, 'VIS')
		self.ir_dir = os.path.join(img_dir, 'IR')
		self.source = [
			im_name
			for im_name in os.listdir(self.vis_dir)
			if im_name.split('.')[-1].lower() in ('jpg', 'png', 'bmp')
		]
		self.transform = transform
		self.distortions = {
			1: apply_gamma_low,
			2: lower_contrast,
			3: higher_contrast,
			4: gaussian_noise,
			5: ori,
		}

	def __len__(self):
		return len(self.source)

	def __getitem__(self, idx):
		name = self.source[idx]
		vis_rgb = cv2.imread(os.path.join(self.vis_dir, name))
		vis_rgb = cv2.cvtColor(vis_rgb, cv2.COLOR_BGR2RGB)
		ir_gray = cv2.imread(os.path.join(self.ir_dir, name), cv2.IMREAD_GRAYSCALE)

		h, w = vis_rgb.shape[:2]
		if h < 280 or w < 280:
			vis_rgb = cv2.resize(vis_rgb, (280, 280))
			ir_gray = cv2.resize(ir_gray, (280, 280))

		dist1 = random.randint(1, 5)
		vis_rgb = self.distortions[dist1](vis_rgb)
		dist2 = random.randint(1, 5)
		ir_gray = self.distortions[dist2](ir_gray)

		if self.transform:
			seed = torch.random.seed()
			torch.random.manual_seed(seed)
			vis_rgb = self.transform(vis_rgb)
			torch.random.manual_seed(seed)
			ir_gray = self.transform(ir_gray)

		return {'name': name, 'img1': vis_rgb, 'img2': ir_gray}


class M3FDIRPairs(Dataset):
	"""Train C_ir_gray: IR grayscale with same aug as vis-ir SICE_IR."""

	def __init__(self, img_dir, transform=None):
		self.source_dir = os.path.join(img_dir, 'IR')
		self.source = [
			im_name
			for im_name in os.listdir(self.source_dir)
			if im_name.split('.')[-1].lower() in ('jpg', 'png', 'bmp')
		]
		self.transform = transform
		self.distortions = {
			1: gaussian_noise,
			2: apply_gamma_low,
			3: apply_gamma_high,
			4: lower_contrast,
			5: higher_contrast,
			6: ori,
		}

	def __len__(self):
		return len(self.source)

	def __getitem__(self, idx):
		name = self.source[idx]
		img = cv2.imread(os.path.join(self.source_dir, name), cv2.IMREAD_GRAYSCALE)
		img1_r, img2_r = _apply_two_view_y_aug(img, self.distortions, (2, 6))

		if self.transform:
			seed = torch.random.seed()
			torch.random.manual_seed(seed)
			img1_r = self.transform(img1_r)
			torch.random.manual_seed(seed)
			img2_r = self.transform(img2_r)

		return {'name': name, 'img1': img1_r, 'img2': img2_r}


class M3FDFusionTrain(Dataset):
	"""
	Train F_gray: paired VIS RGB + IR gray from M3FD train split.
	Augmentations exclude saturation; Y is extracted in the training loop from RGB.
	"""

	def __init__(self, img_dir, transform=None):
		self.vis_dir = os.path.join(img_dir, 'VIS')
		self.ir_dir = os.path.join(img_dir, 'IR')
		self.source = [
			im_name
			for im_name in os.listdir(self.vis_dir)
			if im_name.split('.')[-1].lower() in ('jpg', 'png', 'bmp')
		]
		self.transform = transform
		self.distortions = {
			1: apply_gamma_low,
			2: lower_contrast,
			3: higher_contrast,
			4: gaussian_noise,
			5: ori,
		}

	def __len__(self):
		return len(self.source)

	def __getitem__(self, idx):
		name = self.source[idx]
		vis_rgb = cv2.imread(os.path.join(self.vis_dir, name))
		vis_rgb = cv2.cvtColor(vis_rgb, cv2.COLOR_BGR2RGB)
		ir_gray = cv2.imread(os.path.join(self.ir_dir, name), cv2.IMREAD_GRAYSCALE)

		h, w = vis_rgb.shape[:2]
		if h < 280 or w < 280:
			vis_rgb = cv2.resize(vis_rgb, (280, 280))
			ir_gray = cv2.resize(ir_gray, (280, 280))

		dist1 = random.randint(1, 5)
		vis_rgb = self.distortions[dist1](vis_rgb)
		dist2 = random.randint(1, 5)
		ir_gray = self.distortions[dist2](ir_gray)

		if self.transform:
			seed = torch.random.seed()
			torch.random.manual_seed(seed)
			vis_rgb = self.transform(vis_rgb)
			torch.random.manual_seed(seed)
			ir_gray = self.transform(ir_gray)

		return {'name': name, 'img1': vis_rgb, 'img2': ir_gray}
