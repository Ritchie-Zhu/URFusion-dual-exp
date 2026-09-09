"""
A2 ablation: dual PreNet + single FusionExpertBlock (wo MoE router/experts).
MoE aux loss is disabled at train time (lambda_aux=0); aux dict kept for loss API compatibility.
"""

import os
import sys

_BASE_CODE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'our_model_1_DualMoE', 'code'))
if _BASE_CODE not in sys.path:
	sys.path.insert(0, _BASE_CODE)

import torch
import torch.nn as nn

from models.fusion_expert import FusionExpertBlock
from models.fusion_post import FusionPostNetGray
from models.fusion_pre import FusionPreNet


def _dummy_moe_aux(batch_size, device, num_experts=4):
	gate = torch.zeros(batch_size, num_experts, device=device)
	gate[:, 0] = 1.0
	zeros = torch.zeros(batch_size, num_experts, device=device)
	return {
		'gate': gate,
		'avg_gate': gate.mean(dim=0),
		'clean_logits': zeros,
		'raw_noise_std': zeros,
		'noisy_logits': zeros,
		'topk_indices': torch.zeros(batch_size, 2, dtype=torch.long, device=device),
		'topk_values': zeros[:, :2],
		'routing_mode': 'single_expert',
	}


class DualConvFusionNetGray(nn.Module):
	"""Dual-branch pre-encoder + one cross-modal conv expert (no MoE)."""

	def __init__(self, feat_channels=64):
		super().__init__()
		self.feat_channels = feat_channels
		self.vis_pre = FusionPreNet(in_channels=1, out_channels=feat_channels)
		self.ir_pre = FusionPreNet(in_channels=1, out_channels=feat_channels)
		moe_in = feat_channels * 3
		self.fusion_expert = FusionExpertBlock(moe_in, feat_channels)
		self.post_net = FusionPostNetGray(out_channels=1)
		self.branch_proj_vis = nn.Conv2d(feat_channels, feat_channels, kernel_size=1, bias=True)
		self.branch_proj_ir = nn.Conv2d(feat_channels, feat_channels, kernel_size=1, bias=True)

	@staticmethod
	def _build_x_moe(f_vis, f_ir):
		return torch.cat([f_vis, f_ir, torch.abs(f_vis - f_ir)], dim=1)

	def forward(self, y_vis, ir_gray):
		f_vis = self.vis_pre(y_vis)
		f_ir = self.ir_pre(ir_gray)
		x_moe = self._build_x_moe(f_vis, f_ir)
		f_fused = self.fusion_expert(x_moe)
		y_f = self.post_net(f_fused)

		e_vis = f_vis.abs().mean()
		e_ir = f_ir.abs().mean()
		eps = 1e-6
		p_vis = e_vis / (e_vis + e_ir + eps)
		p_ir = 1.0 - p_vis

		moe_aux = _dummy_moe_aux(y_vis.size(0), y_vis.device)
		moe_aux['x_moe'] = x_moe
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
