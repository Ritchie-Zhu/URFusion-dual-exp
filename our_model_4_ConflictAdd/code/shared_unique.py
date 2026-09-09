"""Shared / unique split in PreNet feature space (not DWT/FFT)."""

import torch
import torch.nn as nn


class SharedUniqueSplit(nn.Module):
	"""
	f_cap = 0.5 * (f_vis + f_ir) + delta(concat)
	f_vis_u = f_vis - f_cap
	f_ir_u  = f_ir  - f_cap

	delta is zero-init so the first forward is an exact average split.
	Unique is identity/log only in this experiment; it does not enter fusion.
	"""

	def __init__(self, channels=64):
		super().__init__()
		self.channels = channels
		self.delta = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True)
		nn.init.zeros_(self.delta.weight)
		nn.init.zeros_(self.delta.bias)

	def forward(self, f_vis, f_ir):
		f_cap = 0.5 * (f_vis + f_ir) + self.delta(torch.cat([f_vis, f_ir], dim=1))
		f_vis_u = f_vis - f_cap
		f_ir_u = f_ir - f_cap
		return f_cap, f_vis_u, f_ir_u
