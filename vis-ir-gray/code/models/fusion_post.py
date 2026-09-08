"""Post-MoE decoder for gray-Y fusion: 64ch -> 1ch Y_F. No A2V modulation."""

import torch.nn as nn
import torch.nn.functional as F


class FusionPostNetGray(nn.Module):
	def __init__(self, out_channels=1):
		super().__init__()
		self.conv8 = nn.Conv2d(64, 32, kernel_size=3, padding=0, bias=True)
		self.conv9 = nn.Conv2d(32, 16, kernel_size=3, padding=0, bias=True)
		self.conv10 = nn.Conv2d(16, 8, kernel_size=3, padding=0, bias=True)
		self.conv11 = nn.Conv2d(8, 4, kernel_size=3, padding=0, bias=True)
		self.conv12 = nn.Conv2d(4, out_channels, kernel_size=3, padding=0, bias=True)
		self.norm8 = nn.GroupNorm(num_groups=1, num_channels=8, eps=0.001, affine=False)
		self.norm4 = nn.GroupNorm(num_groups=1, num_channels=4, eps=0.01, affine=False)
		self.tanh = nn.Tanh()

	def forward(self, f_fused):
		out8 = F.pad(f_fused, (1, 1, 1, 1), mode='reflect')
		out8 = F.leaky_relu(self.conv8(out8))

		out9 = F.pad(out8, (1, 1, 1, 1), mode='reflect')
		out9 = F.leaky_relu(self.conv9(out9))

		out10 = F.pad(out9, (1, 1, 1, 1), mode='reflect')
		out10 = self.conv10(out10)
		out10 = self.norm8(out10)
		out10 = F.leaky_relu(out10)

		out11 = F.pad(out10, (1, 1, 1, 1), mode='reflect')
		out11 = self.conv11(out11)
		out11 = self.norm4(out11)
		out11 = F.leaky_relu(out11)

		out12 = F.pad(out11, (1, 1, 1, 1), mode='reflect')
		out12 = self.conv12(out12)
		y_f = self.tanh(out12) / 2 + 0.5
		return y_f
