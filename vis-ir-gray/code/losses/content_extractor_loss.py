"""
Content extractor losses — weights match vis-ir/code/train_content_extractor_{vis,ir}.py.
Only the input domain changed (VIS: Y 1ch; IR: grayscale 1ch). VGG sees gray_to_vgg3(x).
"""

import torch
import torch.nn as nn

from utils import contrastive_loss, guided_filter, vgg_first_layer
from ycbcr import gray_to_vgg3


def compute_content_extractor_loss_vis(model_R, vgg16, x1, x2, tau=0.4):
	"""
	VIS gray C: same weights as train_content_extractor_vis.py
	L = L_feature + 0.5*L_sim + 1e-5*L_contrastive (contrastive includes internal 0.5 factor).
	"""
	crit = nn.MSELoss()
	s1_ex_feas = model_R(x1)
	s2_ex_feas = model_R(x2)

	v1 = vgg_first_layer(vgg16, gray_to_vgg3(x1))
	v2 = vgg_first_layer(vgg16, gray_to_vgg3(x2))

	all_f1_samples = torch.cat((s1_ex_feas, s2_ex_feas), 0)
	n = x1.shape[0]
	loss_contrastive = 0.5 * contrastive_loss(all_f1_samples, n, tau)

	v1_gf = guided_filter(v1, v2)
	v2_gf = guided_filter(v2, v1)
	loss_feature = crit(v1_gf, s1_ex_feas) + crit(v2_gf, s2_ex_feas)

	pooling1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
	s1_ex_feas_d1 = pooling1(s1_ex_feas)
	s2_ex_feas_d1 = pooling1(s2_ex_feas)
	loss_sim = crit(s1_ex_feas, s2_ex_feas) * 0.2 + crit(s1_ex_feas_d1, s2_ex_feas_d1) * 0.8

	loss = loss_feature + 0.00001 * loss_contrastive + 0.5 * loss_sim
	return {
		'loss': loss,
		'loss_feature': loss_feature,
		'loss_sim': loss_sim,
		'loss_contrastive': loss_contrastive,
	}


def compute_content_extractor_loss_ir(model_R, vgg16, x1, x2, tau=0.4):
	"""
	IR gray C: same weights as train_content_extractor_ir.py
	Internal L_sim uses 1.0/4.0 on full/downsampled features; external multiplier is 0.06.
	"""
	crit = nn.MSELoss()
	s1_ex_feas = model_R(x1)
	s2_ex_feas = model_R(x2)

	v1 = vgg_first_layer(vgg16, gray_to_vgg3(x1))
	v2 = vgg_first_layer(vgg16, gray_to_vgg3(x2))

	all_f1_samples = torch.cat((s1_ex_feas, s2_ex_feas), 0)
	n = x1.shape[0]
	loss_contrastive = 0.5 * contrastive_loss(all_f1_samples, n, tau)

	v1_gf = guided_filter(v1, v2)
	v2_gf = guided_filter(v2, v1)
	loss_feature = crit(v1_gf, s1_ex_feas) + crit(v2_gf, s2_ex_feas)

	pooling1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
	s1_ex_feas_d1 = pooling1(s1_ex_feas)
	s2_ex_feas_d1 = pooling1(s2_ex_feas)
	loss_sim = crit(s1_ex_feas, s2_ex_feas) * 1.0 + crit(s1_ex_feas_d1, s2_ex_feas_d1) * 4.0

	loss = loss_feature + 0.06 * loss_sim + 0.00001 * loss_contrastive
	return {
		'loss': loss,
		'loss_feature': loss_feature,
		'loss_sim': loss_sim,
		'loss_contrastive': loss_contrastive,
	}
