"""Forward / identity / leak / grad checks before training. Does not train."""

import os
import sys

import torch

UPDATE_CODE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(UPDATE_CODE, '..', '..'))
BASE_CODE = os.path.join(PROJECT, 'our_model_1_DualMoE', 'code')
A2_CODE = os.path.join(PROJECT, 'ablation', 'Abl_DualMoE_wo_MoE')
S1_CODE = os.path.join(PROJECT, 'ablation', 'Abl_SpatialResMoE_wo_Res')
sys.path.insert(0, BASE_CODE)
sys.path.insert(0, UPDATE_CODE)

from dual_spatial_moe_fusion import DualConflictAddNetGray
from models.fusion_expert import FusionExpertBlock


def _n_params(m):
	return sum(p.numel() for p in m.parameters())


def main():
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	model = DualConflictAddNetGray().to(device)
	model.train()
	moe = model.moe_block
	base_in = int(moe.expert_base.proj.in_channels)
	conflict_in = int(moe.expert_conflict.proj.in_channels)
	assert base_in == 64, f'E_base must take f_cap only, got in_channels={base_in}'
	assert conflict_in == 192, f'E_conflict must take A2 concat, got in_channels={conflict_in}'
	assert not hasattr(moe, 'expert_vis')
	assert not hasattr(moe, 'expert_ir')
	print(f'E_base in_channels={base_in} (leak-free)')
	print(f'E_conflict in_channels={conflict_in} (A2 input)')

	y_vis = torch.rand(2, 1, 32, 32, device=device)
	ir_gray = torch.rand(2, 1, 32, 32, device=device)
	y_f, aux = model(y_vis, ir_gray)

	assert y_f.shape == (2, 1, 32, 32), y_f.shape
	assert y_f.min() >= 0 and y_f.max() <= 1
	ident_vis = (aux['f_cap'] + aux['f_vis_u'] - aux['f_vis']).abs().max().item()
	ident_ir = (aux['f_cap'] + aux['f_ir_u'] - aux['f_ir']).abs().max().item()
	assert ident_vis < 1e-6, ident_vis
	assert ident_ir < 1e-6, ident_ir

	fused_check = (aux['f_fused'] - (aux['e_base'] + aux['a'] * aux['e_conflict'])).abs().max().item()
	assert fused_check < 1e-6, fused_check

	a = aux['a']
	assert a.shape == (2, 1, 32, 32)
	assert (a > 0).all() and (a < 1).all()
	print(f'a_mean={a.mean().item():.4f} (expect ~0.5 at init)')
	assert aux['x_conflict'].shape[1] == 192
	assert aux['diff'].shape == aux['f_vis'].shape
	assert aux['f_vis_u'].abs().mean().item() > 1e-4
	assert aux['f_ir_u'].abs().mean().item() > 1e-4

	# Unique must not be an expert input. Zeroing unique tensors after split
	# cannot be hooked here; channel counts plus missing expert_vis/ir are the check.

	model.zero_grad()
	y_f.mean().backward()
	router_w = moe.router[0].weight.grad
	assert router_w is not None and router_w.abs().sum() > 0, 'fusion loss must train router (no detach)'
	assert moe.expert_base.proj.weight.grad is not None
	assert moe.expert_conflict.proj.weight.grad is not None
	print('router / E_base / E_conflict have grad through y_f (no detach)')

	old_base = FusionExpertBlock(128, 64)
	ok = False
	try:
		moe.expert_base.load_state_dict(old_base.state_dict())
	except RuntimeError:
		ok = True
	assert ok, 'v3 128-ch E_base should not load into 64-ch E_base'
	print('v3 E_base ckpt cannot load (expected)')

	sys.path.insert(0, A2_CODE)
	from dual_conv_fusion import DualConvFusionNetGray

	sys.path.insert(0, S1_CODE)
	from base_only_fusion import DualSpatialBaseOnlyNetGray

	n_ours = _n_params(model)
	n_a2 = _n_params(DualConvFusionNetGray())
	n_s1 = _n_params(DualSpatialBaseOnlyNetGray())
	print(f'params ours={n_ours} A2={n_a2} S1={n_s1} extra_vs_A2={n_ours - n_a2}')
	print('[SANITY OK]')


if __name__ == '__main__':
	main()
