"""
1-channel gray content encoder C: [B,1,H,W] -> [B,64,H,W].

Used only during training to supervise F; not used at inference.
"""

import torch
import torch.nn as nn


class StructureEncoderGray(nn.Module):
	def __init__(self, in_channels=1, out_channels=64):
		super().__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=True)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=True)
		self.conv3 = nn.Conv2d(48, 32, kernel_size=5, padding=2, bias=True)
		self.conv4 = nn.Conv2d(32, 32, kernel_size=5, padding=2, bias=True)
		self.conv5 = nn.Conv2d(32, 32, kernel_size=5, padding=2, bias=True)
		self.conv6 = nn.Conv2d(32, 32, kernel_size=5, padding=2, bias=True)
		self.conv7 = nn.Conv2d(32, 32, kernel_size=5, padding=2, bias=True)
		self.conv8 = nn.Conv2d(96, 64, kernel_size=3, padding=1, bias=True)
		self.conv9 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
		self.conv10 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1, bias=True)
		self.avg_pool_2 = nn.AvgPool2d(kernel_size=2, stride=2)

		self.sa_conv1_1 = nn.Conv2d(2, 8, 7, padding=3, bias=True)
		self.sa_conv1_2 = nn.Conv2d(8, 1, 3, padding=1, bias=True)
		self.sa_conv2_1 = nn.Conv2d(2, 8, 7, padding=3, bias=True)
		self.sa_conv2_2 = nn.Conv2d(8, 1, 3, padding=1, bias=True)
		self.sa_conv3_1 = nn.Conv2d(2, 8, 7, padding=3, bias=True)
		self.sa_conv3_2 = nn.Conv2d(8, 1, 3, padding=1, bias=True)
		self.sa_conv4_1 = nn.Conv2d(2, 8, 7, padding=3, bias=True)
		self.sa_conv4_2 = nn.Conv2d(8, 1, 3, padding=1, bias=True)

		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		out1 = nn.functional.leaky_relu(self.conv1(x))
		out2 = nn.functional.leaky_relu(self.conv2(out1))

		out3 = self.conv3(torch.cat((out1, out2), dim=1))
		avg_out3 = torch.mean(out3, dim=1, keepdim=True)
		max_out3, _ = torch.max(out3, dim=1, keepdim=True)
		attention3 = self.sa_conv1_1(torch.cat([avg_out3, max_out3], dim=1))
		attention3 = self.sa_conv1_2(nn.functional.leaky_relu(attention3))
		out3 = nn.functional.leaky_relu(out3 * self.sigmoid(attention3))

		out3_ds = self.avg_pool_2(out3)
		avg_out3d = torch.mean(out3_ds, dim=1, keepdim=True)
		max_out3d, _ = torch.max(out3_ds, dim=1, keepdim=True)
		attention3d = self.sa_conv2_1(torch.cat([avg_out3d, max_out3d], dim=1))
		attention3d = self.sa_conv2_2(nn.functional.leaky_relu(attention3d))
		out3_ds = nn.functional.leaky_relu(out3_ds * self.sigmoid(attention3d))

		out4 = nn.functional.relu(self.conv4(out3_ds))

		out4_ds = self.avg_pool_2(out4)
		avg_out4d = torch.mean(out4_ds, dim=1, keepdim=True)
		max_out4d, _ = torch.max(out4_ds, dim=1, keepdim=True)
		attention4d = self.sa_conv3_1(torch.cat([avg_out4d, max_out4d], dim=1))
		attention4d = self.sa_conv3_2(nn.functional.leaky_relu(attention4d))
		out4_ds = nn.functional.leaky_relu(out4_ds * self.sigmoid(attention4d))

		out5 = nn.functional.relu(self.conv5(out4_ds))

		out4_us = nn.functional.interpolate(out4, scale_factor=2, mode='bicubic', align_corners=True)
		out5_us = nn.functional.interpolate(out5, scale_factor=4, mode='bicubic', align_corners=True)

		out6 = nn.functional.leaky_relu(self.conv6(out4_us))
		out7 = nn.functional.leaky_relu(self.conv7(out5_us))

		out8 = self.conv8(torch.cat((out6, out7, out3), dim=1))
		avg_out8 = torch.mean(out8, dim=1, keepdim=True)
		max_out8, _ = torch.max(out8, dim=1, keepdim=True)
		attention8 = self.sa_conv4_1(torch.cat([avg_out8, max_out8], dim=1))
		attention8 = self.sa_conv4_2(nn.functional.leaky_relu(attention8))
		out8 = nn.functional.leaky_relu(out8 * self.sigmoid(attention8))

		out9 = nn.functional.leaky_relu(self.conv9(out8))
		out10 = self.conv10(out9)
		return nn.functional.leaky_relu(out10)
