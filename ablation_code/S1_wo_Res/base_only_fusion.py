"""S1: Dual PreNet + SharedUniqueSplit + E_base only + PostNet.

Fair ablation of SpatialResMoE: remove E_vis, E_ir, and the router.
E_base input is still concat(f_cap, |f_vis-f_ir|), same FusionExpertBlock(128->64).
Unique residuals are computed for identity/logs only; they do not enter fusion.
"""

import torch
import torch.nn as nn

from models.fusion_expert import FusionExpertBlock
from models.fusion_post import FusionPostNetGray
from models.fusion_pre import FusionPreNet

from shared_unique import SharedUniqueSplit


class SpatialBaseOnly(nn.Module):
	def __init__(self, feat_channels=64):
		super().__init__()
		self.feat_channels = feat_channels
		self.split = SharedUniqueSplit(feat_channels)
		self.expert_base = FusionExpertBlock(feat_channels * 2, feat_channels)

	def forward(self, f_vis, f_ir):
		diff = torch.abs(f_vis - f_ir)
		f_cap, f_vis_u, f_ir_u = self.split(f_vis, f_ir)
		e_base = self.expert_base(torch.cat([f_cap, diff], dim=1))
		aux = {
			'f_cap': f_cap,
			'f_vis_u': f_vis_u,
			'f_ir_u': f_ir_u,
			'e_base': e_base,
		}
		return e_base, aux


class DualSpatialBaseOnlyNetGray(nn.Module):
	def __init__(self, feat_channels=64):
		super().__init__()
		self.feat_channels = feat_channels
		self.vis_pre = FusionPreNet(in_channels=1, out_channels=feat_channels)
		self.ir_pre = FusionPreNet(in_channels=1, out_channels=feat_channels)
		self.moe_block = SpatialBaseOnly(feat_channels=feat_channels)
		self.post_net = FusionPostNetGray(out_channels=1)
		self.branch_proj_vis = nn.Conv2d(feat_channels, feat_channels, kernel_size=1, bias=True)
		self.branch_proj_ir = nn.Conv2d(feat_channels, feat_channels, kernel_size=1, bias=True)

	def forward(self, y_vis, ir_gray):
		f_vis = self.vis_pre(y_vis)
		f_ir = self.ir_pre(ir_gray)
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
