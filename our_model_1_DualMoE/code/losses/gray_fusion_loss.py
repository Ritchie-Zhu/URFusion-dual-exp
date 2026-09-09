"""Gray-Y fusion training losses (C-guided main + intensity/grad/branch + MoE aux)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import align_spatial, build_adaptive_mask, cv_squared, probabilistic_topk_load_mc, vgg_first_layer
from ycbcr import gray_to_vgg3


def compute_main_loss(y_f, y_vis, ir_gray, vgg16, c_vis_gray, c_ir_gray, vis_grad_bias=1.5):
	"""L_main = L1(VGG(Y_F), C_guide) with adaptive mask on frozen C features."""
	crit_l1 = nn.L1Loss()
	f_fuse = vgg_first_layer(vgg16, gray_to_vgg3(y_f))

	with torch.no_grad():
		c_vis = c_vis_gray(y_vis)
		c_ir = c_ir_gray(ir_gray)
	mask = build_adaptive_mask(c_vis, c_ir, vis_grad_bias=vis_grad_bias)
	c_guide = mask * c_vis + (1.0 - mask) * c_ir
	c_guide = align_spatial(c_guide, f_fuse)
	return crit_l1(f_fuse, c_guide.detach())


def compute_intensity_loss(y_f, y_vis, ir_gray, crit_l1=None):
	if crit_l1 is None:
		crit_l1 = nn.L1Loss()
	target = torch.max(y_vis, ir_gray)
	return crit_l1(y_f, target)


def compute_gradient_loss(y_f, y_vis, ir_gray, gradient_fn, crit_l1=None):
	if crit_l1 is None:
		crit_l1 = nn.L1Loss()
	g_f = gradient_fn(y_f)
	g_target = torch.max(gradient_fn(y_vis), gradient_fn(ir_gray))
	return crit_l1(g_f, g_target)


def compute_branch_loss(f_vis_proj, f_ir_proj, c_vis, c_ir, crit_l1=None):
	if crit_l1 is None:
		crit_l1 = nn.L1Loss()
	f_vis_proj = align_spatial(f_vis_proj, c_vis)
	f_ir_proj = align_spatial(f_ir_proj, c_ir)
	return crit_l1(f_vis_proj, c_vis.detach()) + crit_l1(f_ir_proj, c_ir.detach())


def compute_moe_aux_loss(gate, clean_logits, raw_noise_std, top_k, mc_samples, noise_epsilon):
	importance = gate.sum(dim=0)
	load_vec = probabilistic_topk_load_mc(
		clean_logits.detach(), raw_noise_std.detach(), top_k, mc_samples, noise_epsilon
	)
	imp_loss = cv_squared(importance)
	ld_loss = cv_squared(load_vec)
	return imp_loss + ld_loss, imp_loss, ld_loss, importance, load_vec


def compute_gray_fusion_loss(
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
	lambda_aux=0.03,
	vis_grad_bias=1.5,
	top_k=2,
	mc_samples=4,
	noise_epsilon=1e-2,
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
	loss_branch = compute_branch_loss(aux['f_vis_proj'], aux['f_ir_proj'], c_vis, c_ir, crit_l1=crit_l1)

	aux_raw, imp_loss, ld_loss, importance, load_vec = compute_moe_aux_loss(
		aux['gate'],
		aux['clean_logits'],
		aux['raw_noise_std'],
		top_k,
		mc_samples,
		noise_epsilon,
	)
	loss_aux = lambda_aux * aux_raw

	loss_total = (
		loss_main
		+ lambda_int * loss_int
		+ lambda_grad * loss_grad
		+ lambda_branch * loss_branch
		+ loss_aux
	)

	return {
		'loss': loss_total,
		'loss_main': loss_main,
		'loss_int': loss_int,
		'loss_grad': loss_grad,
		'loss_branch': loss_branch,
		'aux_loss': loss_aux,
		'importance_loss': imp_loss,
		'load_loss': ld_loss,
		'importance': importance,
		'load_vec': load_vec,
		'e_vis': aux['e_vis'],
		'e_ir': aux['e_ir'],
		'p_vis': aux['p_vis'],
		'p_ir': aux['p_ir'],
	}
