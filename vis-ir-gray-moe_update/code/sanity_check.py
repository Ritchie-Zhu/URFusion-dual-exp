"""Forward / identity / grad checks before training."""

import os
import sys

import torch

UPDATE_CODE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(UPDATE_CODE, '..', '..'))
BASE_CODE = os.path.join(PROJECT, 'vis-ir-gray', 'code')
sys.path.insert(0, BASE_CODE)
sys.path.insert(0, UPDATE_CODE)

from dual_spatial_moe_fusion import DualSpatialResMoENetGray


def main():
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	model = DualSpatialResMoENetGray().to(device)
	model.train()
	y_vis = torch.rand(2, 1, 160, 160, device=device)
	ir_gray = torch.rand(2, 1, 160, 160, device=device)
	y_f, aux = model(y_vis, ir_gray)

	assert y_f.shape == (2, 1, 160, 160), y_f.shape
	assert y_f.min() >= 0 and y_f.max() <= 1
	ident_vis = (aux['f_cap'] + aux['f_vis_u'] - aux['f_vis']).abs().max().item()
	ident_ir = (aux['f_cap'] + aux['f_ir_u'] - aux['f_ir']).abs().max().item()
	assert ident_vis < 1e-6, ident_vis
	assert ident_ir < 1e-6, ident_ir
	g = aux['g']
	assert g.shape == (2, 1, 160, 160)
	assert (g > 0).all() and (g < 1).all()
	print(f'g_mean={g.mean().item():.4f} (expect ~0.5 at init)')

	class _DummyC(torch.nn.Module):
		def forward(self, x):
			return x.repeat(1, 64, 1, 1)

	c = _DummyC().to(device).eval()
	for p in c.parameters():
		p.requires_grad = False
	# skip full loss here if VGG not needed; check route grad on g
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
	print('experts have grad through y_f')
	print('[SANITY OK]')


if __name__ == '__main__':
	main()
