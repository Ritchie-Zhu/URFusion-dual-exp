"""Gray-Y dual PreNet + ConflictAdd MoE + PostNet. C is not in forward."""

import torch.nn as nn

from models.fusion_post import FusionPostNetGray
from models.fusion_pre import FusionPreNet

from spatial_res_moe import ConflictAddMoE


class DualConflictAddNetGray(nn.Module):
	def __init__(self, feat_channels=64):
		super().__init__()
		self.feat_channels = feat_channels
		self.vis_pre = FusionPreNet(in_channels=1, out_channels=feat_channels)
		self.ir_pre = FusionPreNet(in_channels=1, out_channels=feat_channels)
		self.moe_block = ConflictAddMoE(feat_channels=feat_channels)
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
