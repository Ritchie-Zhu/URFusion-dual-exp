"""Shared utilities for our_model_1_DualMoE (subset ported from vis-ir/code/utils.py)."""

import csv
import json
import os

import torch
import torch.nn.functional as F


def contrastive_loss(all_samples, N, tau):
	for si in range(2 * N):
		this_sample = all_samples[si : si + 1, :, :, :]
		this_sample_tile = this_sample.repeat((2 * N, 1, 1, 1))
		sim = -torch.mean((this_sample_tile - all_samples) ** 2, dim=[1, 2, 3]) / tau
		sim = sim.unsqueeze(-1)
		if si == 0:
			similarities = sim
		else:
			similarities = torch.cat((similarities, sim), -1)

	for si in range(2 * N):
		sj = (si + N) % (2 * N)
		pos = similarities[si, sj]
		neg_num = 0
		for sk in range(2 * N):
			if sk != sj:
				neg = similarities[si, sk].unsqueeze(0)
				if neg_num == 0:
					neg_num = neg_num + 1
					negs = neg
				else:
					negs = torch.cat((negs, neg), 0)
		softmax = -torch.logsumexp(pos, 0) + torch.logsumexp(negs, 0)
		softmax = softmax.unsqueeze(-1)
		if si == 0:
			softmaxes = softmax
		else:
			softmaxes = torch.cat((softmaxes, softmax), 0)

	return torch.mean(softmaxes, 0)


def box_filter(x, r):
	return F.avg_pool2d(x, (2 * r + 1, 2 * r + 1), stride=1, padding=r)


def guided_filter(X, G, r=1, eps=0.0001):
	mean_G = box_filter(G, r)
	mean_X = box_filter(X, r)
	mean_Ip = box_filter(G * X, r)
	cov_Ip = mean_Ip - mean_G * mean_X

	mean_GG = box_filter(G * G, r)
	var_G = mean_GG - mean_G * mean_G

	a = cov_Ip / (var_G + eps)
	b = mean_X - a * mean_G

	mean_a = box_filter(a, r)
	mean_b = box_filter(b, r)
	return mean_a * G + mean_b


def gradient_operator(x):
	kernel_x = torch.tensor([[0, 0, 0], [1.0, -1.0, 0], [0, 0, 0]], dtype=torch.float)
	kernel_y = torch.tensor([[0, 1.0, 0], [0, -1.0, 0], [0, 0, 0]], dtype=torch.float)
	kernel_x = kernel_x.view(1, 1, 3, 3).repeat(x.size(1), x.size(1), 1, 1)
	kernel_y = kernel_y.view(1, 1, 3, 3).repeat(x.size(1), x.size(1), 1, 1)
	kernel_x = kernel_x.to(x.device)
	kernel_y = kernel_y.to(x.device)
	gradient_x = torch.abs(F.conv2d(x, kernel_x, stride=1, padding=1))
	gradient_y = torch.abs(F.conv2d(x, kernel_y, stride=1, padding=1))
	return gradient_x + gradient_y


def build_adaptive_mask(fea_a, fea_b, vis_grad_bias=1.5):
	grad_a = gradient_operator(fea_a)
	grad_b = gradient_operator(fea_b)
	return (grad_a * vis_grad_bias > grad_b).float()


def align_spatial(feat, ref):
	if feat.shape[-2:] != ref.shape[-2:]:
		feat = F.interpolate(feat, size=ref.shape[-2:], mode='bilinear', align_corners=False)
	return feat


def cv_squared(v, eps=1e-6):
	mu = v.mean()
	sig = v.std(unbiased=False)
	return (sig / (mu.abs() + eps)) ** 2


def probabilistic_topk_load_mc(clean_logits, raw_noise_std, top_k, n_mc, noise_epsilon):
	"""Expected top-k membership per expert (MC average over batch)."""
	b, e = clean_logits.shape
	device = clean_logits.device
	dtype = clean_logits.dtype
	std = F.softplus(raw_noise_std) + noise_epsilon
	acc = torch.zeros(e, device=device, dtype=dtype)
	for _ in range(n_mc):
		noise = torch.randn(b, e, device=device, dtype=dtype)
		noisy = clean_logits + noise * std
		_, topi = torch.topk(noisy, top_k, dim=1)
		mask = torch.zeros_like(noisy)
		mask.scatter_(1, topi, 1.0)
		acc = acc + mask.sum(dim=0)
	return acc / float(n_mc * b)


def load_frozen_structure_encoder(ckpt_root, experiment, device, ckpt_path=None):
	from models import StructureEncoderGray

	model = StructureEncoderGray(in_channels=1, out_channels=64).to(device)
	if ckpt_path:
		path = ckpt_path
	elif os.path.isfile(experiment):
		path = experiment
	else:
		path = resolve_latest_ckpt_path(ckpt_root, experiment)
	if not os.path.exists(path):
		raise FileNotFoundError(f'Content encoder ckpt not found: {path}')
	ckpt = torch.load(path, map_location=device)
	model.load_state_dict(ckpt['model_R_state_dict'])
	model.eval()
	for p in model.parameters():
		p.requires_grad = False
	return model


def vgg_first_layer(vgg16, img_3ch):
	"""VGG16 features layer 0 + leaky_relu."""
	x = img_3ch
	for name, layer in vgg16._modules.items():
		x = layer(x)
		if name == '0':
			return F.leaky_relu(x)
	raise RuntimeError('VGG features layer "0" not found')


def write_epoch_logs(log_dir, experiment, row, fields=None):
	if fields is None:
		fields = ['epoch', 'loss', 'loss_contrastive', 'loss_feature', 'loss_sim']
	os.makedirs(log_dir, exist_ok=True)
	csv_path = os.path.join(log_dir, f'epoch_{experiment}_summary.csv')
	json_path = os.path.join(log_dir, f'last_epoch_{experiment}.json')
	write_header = not os.path.exists(csv_path)
	with open(csv_path, 'a', newline='', encoding='utf-8') as f:
		w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
		if write_header:
			w.writeheader()
		w.writerow(row)
	with open(json_path, 'w', encoding='utf-8') as f:
		json.dump({k: float(v) for k, v in row.items()}, f, indent=2)


def resolve_latest_ckpt_path(ckpt_root, experiment):
	"""Resume ckpt: prefer {ckpt_root}/{experiment}/{experiment}_ckpt.pth, else legacy flat file."""
	new_path = os.path.join(ckpt_root, experiment, f'{experiment}_ckpt.pth')
	legacy_path = os.path.join(ckpt_root, f'{experiment}_ckpt.pth')
	if os.path.exists(new_path):
		return new_path
	if os.path.exists(legacy_path):
		return legacy_path
	return new_path


def save_latest_ckpt(state_dict, ckpt_root, experiment):
	"""Latest checkpoint for resume: ../train-jobs/ckpt/{experiment}/{experiment}_ckpt.pth"""
	exp_dir = os.path.join(ckpt_root, experiment)
	os.makedirs(exp_dir, exist_ok=True)
	path = os.path.join(exp_dir, f'{experiment}_ckpt.pth')
	torch.save(state_dict, path)
	return path


def save_epoch_interval_ckpt(state_dict, ckpt_root, experiment, epoch_display, interval=10):
	"""
	Periodic snapshot every `interval` epochs:
	../train-jobs/ckpt/{experiment}/epoch_{NNN}/ckpt.pth
	"""
	ep = int(epoch_display)
	if ep <= 0 or ep % int(interval) != 0:
		return None
	epoch_dir = os.path.join(ckpt_root, experiment, f'epoch_{ep:03d}')
	os.makedirs(epoch_dir, exist_ok=True)
	path = os.path.join(epoch_dir, 'ckpt.pth')
	torch.save(state_dict, path)
	return path
