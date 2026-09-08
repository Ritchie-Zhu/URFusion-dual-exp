"""Same fusion losses as S1. No L_route, no L_aux, no load-balance.

C still supervises F through L_main / L_branch. The C-mask is logged against
gate `a` only; it is not a routing target.
"""

import torch
import torch.nn as nn

from losses.gray_fusion_loss import (
	compute_branch_loss,
	compute_gradient_loss,
	compute_intensity_loss,
	compute_main_loss,
)
from utils import align_spatial, gradient_operator


def compute_conflict_add_loss(
	y_f,
	y_vis,
	ir_gray,
	aux,
	vgg16,
	c_vis_gray,
	c_ir_gray,
	gradient_fn,
	lambda_int=0.1,
	lambda_grad=1.0,
	lambda_branch=0.01,
	vis_grad_bias=1.5,
):
	crit_l1 = nn.L1Loss()
	loss_main = compute_main_loss(
		y_f, y_vis, ir_gray, vgg16, c_vis_gray, c_ir_gray, vis_grad_bias=vis_grad_bias
	)
	loss_int = compute_intensity_loss(y_f, y_vis, ir_gray, crit_l1=crit_l1)
	loss_grad = compute_gradient_loss(y_f, y_vis, ir_gray, gradient_fn, crit_l1=crit_l1)

	with torch.no_grad():
		c_vis = c_vis_gray(y_vis)
		c_ir = c_ir_gray(ir_gray)
		g_vis = gradient_operator(c_vis).mean(dim=1, keepdim=True)
		g_ir = gradient_operator(c_ir).mean(dim=1, keepdim=True)
		mask = (g_vis * vis_grad_bias > g_ir).float()

	loss_branch = compute_branch_loss(
		aux['f_vis_proj'], aux['f_ir_proj'], c_vis, c_ir, crit_l1=crit_l1
	)

	zero = y_f.new_zeros(())
	loss_total = (
		loss_main
		+ lambda_int * loss_int
		+ lambda_grad * loss_grad
		+ lambda_branch * loss_branch
	)

	a = aux['a']
	mask_a = align_spatial(mask, a)
	e_base = aux['e_base'].abs().mean()
	e_conflict = aux['e_conflict'].abs().mean()
	res_conflict = (a * aux['e_conflict']).abs().mean()

	return {
		'loss': loss_total,
		'loss_main': loss_main,
		'loss_int': loss_int,
		'loss_grad': loss_grad,
		'loss_branch': loss_branch,
		'loss_route': zero,
		'a_mean': a.mean(),
		'a_std': a.std(unbiased=False),
		'mask_mean': mask.mean(),
		'mask_std': mask.std(unbiased=False),
		'a_vs_mask': (a.detach() - mask_a).abs().mean(),
		'e_base_l1': e_base,
		'e_conflict_l1': e_conflict,
		'res_conflict': res_conflict,
		'cap_l1': aux['f_cap'].abs().mean(),
		'vis_u_l1': aux['f_vis_u'].abs().mean(),
		'ir_u_l1': aux['f_ir_u'].abs().mean(),
		'diff_l1': aux['diff'].abs().mean(),
		'e_vis': aux['e_vis'],
		'e_ir': aux['e_ir'],
		'p_vis': aux['p_vis'],
		'p_ir': aux['p_ir'],
	}
