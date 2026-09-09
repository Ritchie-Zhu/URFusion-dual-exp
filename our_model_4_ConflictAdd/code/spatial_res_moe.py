"""ConflictAdd_1: leak-free base + one independent conflict residual.

Hypothesis (the only remaining confound after SpatialResMoE_2):
  MoE_2 lost because exclusive mix + g.detach() + C-mask routing threw unique
  away, not because E_base(f_cap) is wrong. If that is true, a single always-
  available A2-style mixer added on top of the leak-free base should matter.
  If it does not beat S1 and A2 on the 6-col table, MoE stops.

Fusion:
  F = B(f_cap) + a * R(concat(f_vis, f_ir, |diff|))

Locked:
  - B is FusionExpertBlock(64->64). No |diff|, no unique.
  - R is FusionExpertBlock(192->64), same block and same input as A2.
  - a is an independent spatial sigmoid. No softmax, top-2, exclusive mix, floor.
  - a is NOT detached. Fusion losses train the router.
  - No L_route / L_aux / load-balance / diversity. C is train-only fusion supervision.
  - Unique from the split is identity/log only.
  - No C or mask in this module.
  - No R_local / R_context. One residual, so a fail/win is interpretable.

Not SpatialResMoE_3 (that name is our_model_2_SpatialResMoE, leaky E_base).
"""

import torch
import torch.nn as nn

from models.fusion_expert import FusionExpertBlock

from shared_unique import SharedUniqueSplit


class ConflictAddMoE(nn.Module):
	def __init__(self, feat_channels=64):
		super().__init__()
		self.feat_channels = feat_channels
		self.split = SharedUniqueSplit(feat_channels)
		self.expert_base = FusionExpertBlock(feat_channels, feat_channels)
		self.expert_conflict = FusionExpertBlock(feat_channels * 3, feat_channels)
		self.router = nn.Sequential(
			nn.Conv2d(feat_channels * 3, feat_channels, kernel_size=3, padding=1, bias=True),
			nn.LeakyReLU(inplace=True),
			nn.Conv2d(feat_channels, 1, kernel_size=1, bias=True),
		)
		# Bias 0 => a~0.5 at init. Do not zero last-layer weight (blocks router grad).
		nn.init.zeros_(self.router[2].bias)

	def forward(self, f_vis, f_ir):
		diff = torch.abs(f_vis - f_ir)
		x_conflict = torch.cat([f_vis, f_ir, diff], dim=1)
		f_cap, f_vis_u, f_ir_u = self.split(f_vis, f_ir)
		e_base = self.expert_base(f_cap)
		e_conflict = self.expert_conflict(x_conflict)
		logits = self.router(x_conflict).clamp(-8.0, 8.0)
		a = torch.sigmoid(logits)
		f_fused = e_base + a * e_conflict
		aux = {
			'a': a,
			'gate_logits': logits,
			'f_cap': f_cap,
			'f_vis_u': f_vis_u,
			'f_ir_u': f_ir_u,
			'e_base': e_base,
			'e_conflict': e_conflict,
			'diff': diff,
			'x_conflict': x_conflict,
		}
		return f_fused, aux
