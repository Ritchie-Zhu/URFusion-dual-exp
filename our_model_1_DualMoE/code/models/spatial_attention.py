"""64-channel spatial attention (sa_conv3-style from FusionNet)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention64(nn.Module):
	def __init__(self):
		super().__init__()
		self.sa_conv1 = nn.Conv2d(2, 8, 5, padding=0, bias=True)
		self.sa_conv2 = nn.Conv2d(8, 1, 3, padding=0, bias=True)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		avg_x = torch.mean(x, dim=1, keepdim=True)
		max_x, _ = torch.max(x, dim=1, keepdim=True)
		inp = torch.cat([avg_x, max_x], dim=1)
		inp = F.pad(inp, (2, 2, 2, 2), mode='reflect')
		att = F.leaky_relu(self.sa_conv1(inp))
		att = F.pad(att, (1, 1, 1, 1), mode='reflect')
		att = self.sa_conv2(att)
		return x * self.sigmoid(att)
