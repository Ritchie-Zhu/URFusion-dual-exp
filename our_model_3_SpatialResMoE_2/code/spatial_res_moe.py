"""Leak-free spatial residual MoE.

v3 (our_model_2_SpatialResMoE) fed E_base concat(f_cap, |diff|). |diff| already
carries unique vis/ir contrast, so E_vis / E_ir were optional and S1 (base only)
won the 6-col table.

This block keeps the same split, residual experts, and detached C-mask router.
The only fusion-path change: E_base sees f_cap only. Unique information can
enter the fused feature solely through E_vis(f_vis_u) and E_ir(f_ir_u).

|diff| is still an input to the router (gating), not to E_base.
"""

import torch
import torch.nn as nn

from models.fusion_expert import FusionExpertBlock

from shared_unique import SharedUniqueSplit


class SpatialResidualMoE(nn.Module):
	def __init__(self, feat_channels=64):
		super().__init__()
		self.feat_channels = feat_channels
		self.split = SharedUniqueSplit(feat_channels)
		self.expert_base = FusionExpertBlock(feat_channels, feat_channels)
		self.expert_vis = FusionExpertBlock(feat_channels, feat_channels)
		self.expert_ir = FusionExpertBlock(feat_channels, feat_channels)
		self.router = nn.Sequential(
			nn.Conv2d(feat_channels * 3, feat_channels, kernel_size=3, padding=1, bias=True),
			nn.LeakyReLU(inplace=True),
			nn.Conv2d(feat_channels, 1, kernel_size=1, bias=True),
		)
		# Bias 0 => g~0.5 at init. Do not zero last-layer weight (blocks router grad).
		nn.init.zeros_(self.router[2].bias)
		self.gate_floor = 0.05

	def forward(self, f_vis, f_ir):
		diff = torch.abs(f_vis - f_ir)
		f_cap, f_vis_u, f_ir_u = self.split(f_vis, f_ir)
		e_base = self.expert_base(f_cap)
		e_vis = self.expert_vis(f_vis_u)
		e_ir = self.expert_ir(f_ir_u)
		logits = self.router(torch.cat([f_vis, f_ir, diff], dim=1)).clamp(-8.0, 8.0)
		g = torch.sigmoid(logits)
		# Router is taught only by L_route. Fusion losses must not drive g.
		g_mix = self.gate_floor + (1.0 - 2.0 * self.gate_floor) * g.detach()
		f_fused = e_base + g_mix * e_vis + (1.0 - g_mix) * e_ir
		aux = {
			'g': g,
			'gate_logits': logits,
			'f_cap': f_cap,
			'f_vis_u': f_vis_u,
			'f_ir_u': f_ir_u,
			'e_base': e_base,
			'e_vis': e_vis,
			'e_ir': e_ir,
			'diff': diff,
		}
		return f_fused, aux
