"""
A1 ablation: shared PreNet for VIS-Y and IR (wo dual encoders).
Keeps noisy top-2 MoE + post + branch projections; same aux layout as Full F.
"""

import os
import sys

_BASE_CODE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'vis-ir-gray', 'code'))
if _BASE_CODE not in sys.path:
	sys.path.insert(0, _BASE_CODE)

import torch.nn as nn

from models.dual_moe import DualModalMoEFusionBlockNoiseTop2
from models.fusion_post import FusionPostNetGray
from models.fusion_pre import FusionPreNet


class SharedPreDualBranchFusionNetGray(nn.Module):
	"""Full F with one shared FusionPreNet for both modalities."""

	def __init__(self, num_experts=4, top_k=2, feat_channels=64, noise_epsilon=1e-2):
		super().__init__()
		self.shared_pre = FusionPreNet(in_channels=1, out_channels=feat_channels)
		self.moe_block = DualModalMoEFusionBlockNoiseTop2(
			num_experts=num_experts,
			top_k=top_k,
			feat_channels=feat_channels,
			noise_epsilon=noise_epsilon,
		)
		self.post_net = FusionPostNetGray(out_channels=1)
		self.branch_proj_vis = nn.Conv2d(feat_channels, feat_channels, kernel_size=1, bias=True)
		self.branch_proj_ir = nn.Conv2d(feat_channels, feat_channels, kernel_size=1, bias=True)

	def forward(self, y_vis, ir_gray):
		f_vis = self.shared_pre(y_vis)
		f_ir = self.shared_pre(ir_gray)
		f_fused, moe_aux = self.moe_block(f_vis, f_ir)
		y_f = self.post_net(f_fused)

		e_vis = f_vis.abs().mean()
		e_ir = f_ir.abs().mean()
		eps = 1e-6
		p_vis = e_vis / (e_vis + e_ir + eps)
		p_ir = 1.0 - p_vis

		aux = dict(moe_aux)
		aux.update(
			{
				'f_vis': f_vis,
				'f_ir': f_ir,
				'f_fused': f_fused,
				'f_vis_proj': self.branch_proj_vis(f_vis),
				'f_ir_proj': self.branch_proj_ir(f_ir),
				'e_vis': e_vis,
				'e_ir': e_ir,
				'p_vis': p_vis,
				'p_ir': p_ir,
			}
		)
		return y_f, aux
