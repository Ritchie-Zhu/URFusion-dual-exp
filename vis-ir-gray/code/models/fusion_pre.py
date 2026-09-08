"""
Single-modality pre-MoE encoder: 1ch gray/Y or IR -> [B, 64, H, W].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionPreNet(nn.Module):
	def __init__(self, in_channels=1, out_channels=64):
		super().__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=0, bias=True)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=0, bias=True)
		self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=0, bias=True)
		self.conv4 = nn.Conv2d(48, 64, kernel_size=3, padding=0, bias=True)
		self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=0, bias=True)
		self.conv6 = nn.Conv2d(96, 64, kernel_size=3, padding=0, bias=True)
		self.conv7 = nn.Conv2d(128, out_channels, kernel_size=3, padding=0, bias=True)

		self.avg_pool_2 = nn.AvgPool2d(kernel_size=2, stride=2)

		self.sa_conv1_1 = nn.Conv2d(2, 8, 5, padding=0, bias=True)
		self.sa_conv1_2 = nn.Conv2d(8, 1, 3, padding=0, bias=True)
		self.sa_conv2_1 = nn.Conv2d(2, 8, 5, padding=0, bias=True)
		self.sa_conv2_2 = nn.Conv2d(8, 1, 3, padding=0, bias=True)
		self.sa_conv3_1 = nn.Conv2d(2, 8, 5, padding=0, bias=True)
		self.sa_conv3_2 = nn.Conv2d(8, 1, 3, padding=0, bias=True)

		self.sigmoid = nn.Sigmoid()
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.max_pool = nn.AdaptiveMaxPool2d(1)
		self.fc = nn.Sequential(
			nn.Linear(48, 48 // 2),
			nn.ReLU(inplace=True),
			nn.Linear(48 // 2, 48),
			nn.Sigmoid(),
		)

	def forward(self, x):
		x = F.pad(x, (1, 1, 1, 1), mode='reflect')
		out1 = F.leaky_relu(self.conv1(x))

		out2 = F.pad(out1, (1, 1, 1, 1), mode='reflect')
		out2 = F.leaky_relu(self.conv2(out2))

		out3 = F.pad(out2, (1, 1, 1, 1), mode='reflect')
		out3 = F.leaky_relu(self.conv3(out3))

		out13 = torch.cat((out3, out1), 1)
		avg_out13 = self.avg_pool(out13).squeeze(-1).squeeze(-1)
		max_out13 = self.max_pool(out13).squeeze(-1).squeeze(-1)
		avg_attention13 = self.fc(avg_out13)
		max_attention13 = self.fc(max_out13)
		attention13 = avg_attention13.unsqueeze(2).unsqueeze(3) + max_attention13.unsqueeze(2).unsqueeze(3)
		out13_atten = out13 * self.sigmoid(attention13)

		out4 = F.pad(out13_atten, (1, 1, 1, 1), mode='reflect')
		out4 = self.conv4(out4)
		avg_out4 = torch.mean(out4, dim=1, keepdim=True)
		max_out4, _ = torch.max(out4, dim=1, keepdim=True)
		sa_in = torch.cat([avg_out4, max_out4], dim=1)
		sa_in = F.pad(sa_in, (2, 2, 2, 2), mode='reflect')
		attention4 = F.leaky_relu(self.sa_conv1_1(sa_in))
		attention4 = F.pad(attention4, (1, 1, 1, 1), mode='reflect')
		attention4 = self.sa_conv1_2(attention4)
		out4 = F.leaky_relu(out4 * self.sigmoid(attention4))

		out4_ds = self.avg_pool_2(out4)
		out5 = F.pad(out4_ds, (1, 1, 1, 1), mode='reflect')
		out5 = F.leaky_relu(self.conv5(out5))

		out3_ds = self.avg_pool_2(out3)
		out35 = torch.cat((out5, out3_ds), 1)
		avg_out35 = torch.mean(out35, dim=1, keepdim=True)
		max_out35, _ = torch.max(out35, dim=1, keepdim=True)
		sa_in = torch.cat([avg_out35, max_out35], dim=1)
		sa_in = F.pad(sa_in, (2, 2, 2, 2), mode='reflect')
		attention35 = F.leaky_relu(self.sa_conv2_1(sa_in))
		attention35 = F.pad(attention35, (1, 1, 1, 1), mode='reflect')
		attention35 = self.sa_conv2_2(attention35)
		out35 = F.leaky_relu(out35 * self.sigmoid(attention35))
		out35_us = F.interpolate(out35, scale_factor=2, mode='bicubic', align_corners=True)

		out6 = F.pad(out35_us, (1, 1, 1, 1), mode='reflect')
		out6 = F.leaky_relu(self.conv6(out6))

		merged = torch.cat((out6, out4), dim=1)
		merged = F.pad(merged, (1, 1, 1, 1), mode='reflect')
		out7 = self.conv7(merged)
		avg_out7 = torch.mean(out7, dim=1, keepdim=True)
		max_out7, _ = torch.max(out7, dim=1, keepdim=True)
		sa_in = torch.cat([avg_out7, max_out7], dim=1)
		sa_in = F.pad(sa_in, (2, 2, 2, 2), mode='reflect')
		attention7 = F.leaky_relu(self.sa_conv3_1(sa_in))
		attention7 = F.pad(attention7, (1, 1, 1, 1), mode='reflect')
		attention7 = self.sa_conv3_2(attention7)
		out7 = F.leaky_relu(out7 * self.sigmoid(attention7))
		return out7
