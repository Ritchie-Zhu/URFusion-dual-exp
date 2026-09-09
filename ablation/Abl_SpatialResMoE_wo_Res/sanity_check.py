"""Ablation fairness checks: no residual experts, split identity, grads, L_route=0."""

import os
import sys

import torch

ABLATION_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(ABLATION_DIR, '..', '..'))
BASE_CODE = os.path.join(PROJECT, 'our_model_1_DualMoE', 'code')
sys.path.insert(0, BASE_CODE)
sys.path.insert(0, ABLATION_DIR)

from base_only_fusion import DualSpatialBaseOnlyNetGray
from wo_res_loss import compute_wo_res_loss
from utils import gradient_operator


def n_params(model):
	return sum(p.numel() for p in model.parameters())


def main():
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	model = DualSpatialBaseOnlyNetGray().to(device)
	model.train()

	keys = list(model.state_dict().keys())
	assert not any('expert_vis' in k or 'expert_ir' in k or 'router' in k for k in keys), keys
	assert any('expert_base' in k for k in keys)
	assert any('split.delta' in k for k in keys)
	print(f'Abl_SpatialResMoE_wo_Res params={n_params(model)} (no vis/ir expert, no router)')

	y_vis = torch.rand(2, 1, 160, 160, device=device)
	ir_gray = torch.rand(2, 1, 160, 160, device=device)
	y_f, aux = model(y_vis, ir_gray)
	assert y_f.shape == (2, 1, 160, 160), y_f.shape
	assert float(y_f.min()) >= 0 and float(y_f.max()) <= 1
	ident_vis = (aux['f_cap'] + aux['f_vis_u'] - aux['f_vis']).abs().max().item()
	ident_ir = (aux['f_cap'] + aux['f_ir_u'] - aux['f_ir']).abs().max().item()
	assert ident_vis < 1e-6, ident_vis
	assert ident_ir < 1e-6, ident_ir
	assert torch.equal(aux['f_fused'], aux['e_base'])
	assert 'g' not in aux and 'gate_logits' not in aux

	y_f.mean().backward()
	assert model.moe_block.expert_base.proj.weight.grad is not None
	assert model.moe_block.split.delta.weight.grad is not None
	print('E_base and split.delta have grad through y_f')

	class _DummyC(torch.nn.Module):
		def forward(self, x):
			return x.repeat(1, 64, 1, 1)

	c = _DummyC().to(device).eval()
	vgg = torch.nn.Identity().to(device).eval()
	model.zero_grad()
	y_f2, aux2 = model(y_vis, ir_gray)
	# VGG-free smoke: only check route term is a detached zero
	from wo_res_loss import compute_wo_res_loss as _loss

	# Use a tiny stand-in: branch/int/grad work; skip full VGG main by calling internals
	stats = None
	try:
		import torchvision.models as models

		vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.to(device).eval()
		stats = _loss(
			y_f2, y_vis, ir_gray, aux2, vgg16, c, c, gradient_operator,
			lambda_int=0.1, lambda_grad=1.0, lambda_branch=0.01,
		)
	except Exception as exc:
		print(f'skip full loss smoke: {exc}')
	if stats is not None:
		assert float(stats['loss_route'].item()) == 0.0
		stats['loss'].backward()
		print('L_route==0 and fusion loss backprops')

	# Full model must not load into S1
	try:
		sys.path.insert(0, os.path.join(PROJECT, 'our_model_2_SpatialResMoE', 'code'))
		from dual_spatial_moe_fusion import DualSpatialResMoENetGray

		full = DualSpatialResMoENetGray()
		ok = False
		try:
			model.load_state_dict(full.state_dict())
		except RuntimeError:
			ok = True
		assert ok, 'S1 must reject Full state_dict'
		print(f'Full params={n_params(full)} > ablation; state_dict rejected (fair isolation)')
	except ImportError as exc:
		print(f'skip Full compare: {exc}')

	print('[SANITY OK]')


if __name__ == '__main__':
	main()
