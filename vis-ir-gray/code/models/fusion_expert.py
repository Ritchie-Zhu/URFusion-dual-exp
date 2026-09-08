"""Cross-modal fusion expert: per-expert Conv1x1(3C->C) then conv + spatial attention."""

import torch.nn as nn
import torch.nn.functional as F

from models.spatial_attention import SpatialAttention64


class FusionExpertBlock(nn.Module):
	"""
	Expert for dual-modal MoE (not dehaze/derain specialists).
	Input x_moe = concat(f_vis, f_ir, |f_vis - f_ir|); each expert has its own 3C->C projection.
	"""

	def __init__(self, in_channels, out_channels=64):
		super().__init__()
		self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
		self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)
		self.attention = SpatialAttention64()

	def forward(self, x_moe):
		z = self.proj(x_moe)
		h = F.relu(self.conv1(z))
		h = self.attention(h)
		h = self.conv2(h)
		return z + h
