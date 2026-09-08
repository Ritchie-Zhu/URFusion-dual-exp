"""Forward / identity / leak / grad checks before training."""

import os
import sys

import torch

UPDATE_CODE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(UPDATE_CODE, '..', '..'))
BASE_CODE = os.path.join(PROJECT, 'vis-ir-gray', 'code')
sys.path.insert(0, BASE_CODE)
sys.path.insert(0, UPDATE_CODE)

from dual_spatial_moe_fusion import DualSpatialResMoENetGray
from models.fusion_expert import FusionExpertBlock


def main():
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	model = DualSpatialResMoENetGray().to(device)
	model.train()
	base_in = int(model.moe_block.expert_base.proj.in_channels)
	assert base_in == 64, f'E_base must take f_cap only, got in_channels={base_in}'
	assert int(model.moe_block.expert_vis.proj.in_channels) == 64
	assert int(model.moe_block.expert_ir.proj.in_channels) == 64
	# v3 leaked unique via concat(f_cap, |diff|) = 128-ch E_base.
	assert not hasattr(model.moe_block.expert_base.proj, 'in_features')
	print(f'E_base in_channels={base_in} (leak-free)')

	y_vis = torch.rand(2, 1, 32, 32, device=device)
	ir_gray = torch.rand(2, 1, 32, 32, device=device)
	y_f, aux = model(y_vis, ir_gray)

	assert y_f.shape == (2, 1, 32, 32), y_f.shape
	assert y_f.min() >= 0 and y_f.max() <= 1
	ident_vis = (aux['f_cap'] + aux['f_vis_u'] - aux['f_vis']).abs().max().item()
	ident_ir = (aux['f_cap'] + aux['f_ir_u'] - aux['f_ir']).abs().max().item()
	assert ident_vis < 1e-6, ident_vis
	assert ident_ir < 1e-6, ident_ir
	g = aux['g']
	assert g.shape == (2, 1, 32, 32)
	assert (g > 0).all() and (g < 1).all()
	print(f'g_mean={g.mean().item():.4f} (expect ~0.5 at init)')
	assert aux['diff'].shape == aux['f_vis'].shape

	# Unique must be nonzero at init (zero-init delta => unique = ±0.5*(vis-ir)).
	assert aux['f_vis_u'].abs().mean().item() > 1e-4
	assert aux['f_ir_u'].abs().mean().item() > 1e-4

	g_sum = g.mean()
	g_sum.backward()
	router_w = model.moe_block.router[0].weight.grad
	assert router_w is not None and router_w.abs().sum() > 0
	print('router has grad through g')

	model.zero_grad()
	y_f2, _ = model(y_vis, ir_gray)
	y_f2.mean().backward()
	assert model.moe_block.expert_base.proj.weight.grad is not None
	assert model.moe_block.expert_vis.proj.weight.grad is not None
	assert model.moe_block.expert_ir.proj.weight.grad is not None
	print('E_base / E_vis / E_ir have grad through y_f')

	old_base = FusionExpertBlock(128, 64)
	ok = False
	try:
		model.moe_block.expert_base.load_state_dict(old_base.state_dict())
	except RuntimeError:
		ok = True
	assert ok, 'v3 128-ch E_base should not load into leak-free 64-ch E_base'
	print('v3 E_base ckpt cannot load (expected)')
	print('[SANITY OK]')


if __name__ == '__main__':
	main()
