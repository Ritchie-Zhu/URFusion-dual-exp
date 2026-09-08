"""
BT.601 YCbCr utilities for the gray-Y pipeline.

Color conversion lives here (not inside C or F). All tensors are in [0, 1].
Coefficients match vis-ir/code/utils.py rgb2ycbcr / ycbcr2rgb.
"""

import numpy as np
import torch


def rgb_to_ycbcr_tensor(rgb):
	"""
	Args:
		rgb: [B, 3, H, W] in [0, 1]
	Returns:
		y, cb, cr: each [B, 1, H, W] in [0, 1]
	"""
	rgb = torch.clamp(rgb, min=0.0, max=1.0)
	r = rgb[:, 0:1, :, :] * 255.0
	g = rgb[:, 1:2, :, :] * 255.0
	b = rgb[:, 2:3, :, :] * 255.0

	y = 0.257 * r + 0.504 * g + 0.098 * b + 16.0
	cb = -0.148 * r - 0.291 * g + 0.439 * b + 128.0
	cr = 0.439 * r - 0.368 * g - 0.071 * b + 128.0

	y = torch.clamp(y / 255.0, 0.0, 1.0)
	cb = torch.clamp(cb / 255.0, 0.0, 1.0)
	cr = torch.clamp(cr / 255.0, 0.0, 1.0)
	return y, cb, cr


def ycbcr_to_rgb_tensor(y, cb, cr):
	"""
	Args:
		y, cb, cr: each [B, 1, H, W] in [0, 1]
	Returns:
		rgb: [B, 3, H, W] in [0, 1]
	"""
	y = torch.clamp(y, 0.0, 1.0) * 255.0
	cb = torch.clamp(cb, 0.0, 1.0) * 255.0
	cr = torch.clamp(cr, 0.0, 1.0) * 255.0

	r = 1.164 * (y - 16.0) + 1.596 * (cr - 128.0)
	g = 1.164 * (y - 16.0) - 0.813 * (cr - 128.0) - 0.392 * (cb - 128.0)
	b = 1.164 * (y - 16.0) + 2.017 * (cb - 128.0)

	rgb = torch.cat((r, g, b), dim=1) / 255.0
	return torch.clamp(rgb, 0.0, 1.0)


def gray_to_vgg3(gray):
	"""Repeat 1-channel Y/IR to 3ch for VGG auxiliary branches only."""
	if gray.size(1) == 3:
		return gray
	if gray.size(1) != 1:
		raise ValueError(f'gray_to_vgg3 expects 1 or 3 channels, got {gray.size(1)}')
	return gray.repeat(1, 3, 1, 1)


def rgb_uint8_to_y_uint8(rgb):
	"""Numpy RGB uint8 [H,W,3] -> Y uint8 [H,W] (BT.601). Used in dataset before Y-domain aug."""
	r = rgb[:, :, 0].astype(np.float32)
	g = rgb[:, :, 1].astype(np.float32)
	b = rgb[:, :, 2].astype(np.float32)
	y = 0.257 * r + 0.504 * g + 0.098 * b + 16.0
	return np.clip(y, 0, 255).astype(np.uint8)
