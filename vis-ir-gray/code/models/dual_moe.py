"""
Dual-modal noisy top-2 MoE: first explicit cross-modal fusion point.
Router uses GAP(f_vis), GAP(f_ir), GAP(|diff|); experts consume spatial x_moe.
No ConditionPriorExtractor by default.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fusion_expert import FusionExpertBlock


class DualModalMoEFusionBlockNoiseTop2(nn.Module):
	def __init__(self, num_experts=4, top_k=2, feat_channels=64, noise_epsilon=1e-2):
		super().__init__()
		self.num_experts = num_experts
		self.top_k = top_k
		self.feat_channels = feat_channels
		self.noise_epsilon = float(noise_epsilon)
		self.deterministic_inference = False

		router_in = feat_channels * 3
		self.router = nn.Sequential(
			nn.Linear(router_in, 64),
			nn.LeakyReLU(inplace=True),
			nn.Linear(64, 2 * num_experts),
		)
		moe_in = feat_channels * 3
		self.experts = nn.ModuleList(
			[FusionExpertBlock(moe_in, feat_channels) for _ in range(num_experts)]
		)

	def _build_x_moe(self, f_vis, f_ir):
		return torch.cat([f_vis, f_ir, torch.abs(f_vis - f_ir)], dim=1)

	def _router_input(self, f_vis, f_ir):
		diff = torch.abs(f_vis - f_ir)
		g_vis = F.adaptive_avg_pool2d(f_vis, 1).flatten(1)
		g_ir = F.adaptive_avg_pool2d(f_ir, 1).flatten(1)
		g_diff = F.adaptive_avg_pool2d(diff, 1).flatten(1)
		return torch.cat([g_vis, g_ir, g_diff], dim=1)

	def forward(self, f_vis, f_ir):
		x_moe = self._build_x_moe(f_vis, f_ir)
		ri = self._router_input(f_vis, f_ir)
		h = self.router[0](ri)
		h = F.leaky_relu(h)
		z = self.router[2](h)
		n = self.num_experts
		clean_logits = z[:, :n] + z[:, n : 2 * n]
		raw_noise_std = z[:, n : 2 * n]
		std = F.softplus(raw_noise_std) + self.noise_epsilon
		use_noisy_route = self.training or not self.deterministic_inference
		if use_noisy_route:
			route_logits = clean_logits + torch.randn_like(clean_logits) * std
		else:
			route_logits = clean_logits

		top_v, top_i = torch.topk(route_logits, self.top_k, dim=1)
		masked = torch.full_like(clean_logits, float('-inf'))
		masked.scatter_(1, top_i, top_v)
		gate = F.softmax(masked, dim=1)

		out = 0
		for k in range(n):
			out = out + gate[:, k : k + 1, None, None] * self.experts[k](x_moe)

		aux = {
			'gate': gate,
			'avg_gate': gate.mean(dim=0),
			'clean_logits': clean_logits,
			'raw_noise_std': raw_noise_std,
			'noisy_logits': route_logits,
			'topk_indices': top_i,
			'topk_values': top_v,
			'routing_mode': 'noisy' if use_noisy_route else 'clean_deterministic',
			'x_moe': x_moe,
		}
		return out, aux
