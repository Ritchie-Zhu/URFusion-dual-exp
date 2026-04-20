import os
import sys
import time
import math
import shutil
import argparse
import csv
import json
from datetime import datetime
from subprocess import call
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import torchvision.models as models
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
from dataset import *
from utils import *
from model import *
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
# from sklearn.cluster import KMeans

eps=1e-6

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--device', type=int, default=3)
	parser.add_argument('--experiment', default='content-fusion',
						help='prefix of outputs, e.g., experiment_best_model.pth will be saved to ckpt/')
	parser.add_argument('--structure_vis_ckpt', default='content-extractor-vis', help='prefix of structure_ckpt, e.g., structure_model.pth will be saved to ckpt/')
	parser.add_argument('--structure_ir_ckpt', default='content-extractor-ir',  help='prefix of structure_ckpt, e.g., structure_model.pth will be saved to ckpt/')
	parser.add_argument('--baseDir', type=str, default='../dataset/', help='baseDir/train, baseDir/val will be used')
	parser.add_argument('--testDir', type=str, default='../dataset/test/', help='path to test images')
	parser.add_argument('--numEpoch', type=int, default=80)
	parser.add_argument('--patchsize', type=int, default=160)
	parser.add_argument('--batchsize', type=int, default=12)
	parser.add_argument(
		'--weather_aug_p',
		type=float,
		default=0.0,
		help='Per-sample prob to apply synthetic weather (haze/fog/rain-blur) on VIS only during fusion training. '
			 '0 keeps original behavior; try 0.3–0.5 for weather-robust fusion.',
	)
	parser.add_argument(
		'--fusion_model',
		type=str,
		default='fusionnet',
		choices=('fusionnet', 'noise_top1', 'noise_top2'),
		help='fusionnet: baseline FusionNet; noise_top1 / noise_top2: noisy dual-head router + sparse top-k MoE.',
	)
	parser.add_argument('--lambda_aux', type=float, default=0.01, help='Weight for MoE aux (CV^2 importance + CV^2 load).')
	parser.add_argument(
		'--lambda_aux_top1_warmup',
		type=float,
		default=0.02,
		help='noise_top1 only: stronger lambda_aux for early epochs (see --aux_warmup_epochs); mitigates early expert skew.',
	)
	parser.add_argument(
		'--aux_warmup_epochs',
		type=int,
		default=10,
		help='noise_top1 only: for epoch index 1..N (inclusive), use lambda_aux_top1_warmup; then use --lambda_aux. Set 0 to disable.',
	)
	parser.add_argument(
		'--mc_samples',
		'--load_mc_samples',
		type=int,
		default=8,
		help='Monte Carlo draws for probabilistic load estimate (alias: --load_mc_samples).',
	)
	parser.add_argument(
		'--noise_epsilon',
		type=float,
		default=1e-2,
		help='Minimum noise std in MC load: softplus(raw_std) + noise_epsilon.',
	)
	parser.add_argument(
		'--fixed_test_vis_dir',
		type=str,
		default='',
		help='Optional: directory of fixed test VIS images for epoch-end PNGs (default: testDir/vis).',
	)
	args = parser.parse_args()
	if args.fusion_model in ('noise_top1', 'noise_top2') and args.experiment == 'content-fusion':
		args.experiment = args.fusion_model
	return args


def _cv_squared(v, eps=1e-6):
	"""Squared coefficient of variation across 4 expert statistics."""
	mu = v.mean()
	sig = v.std(unbiased=False)
	return (sig / (mu.abs() + eps)) ** 2


def _probabilistic_topk_load_mc(clean_logits, raw_noise_std, top_k, n_mc, noise_epsilon):
	"""
	Expected top-k membership per expert (MC): noisy_logits = clean + N(0,1)*std,
	std = softplus(raw_noise_std) + noise_epsilon. Averaged over MC and batch.
	clean_logits, raw_noise_std: [B, 4]
	"""
	b, e = clean_logits.shape
	device = clean_logits.device
	dtype = clean_logits.dtype
	std = torch.nn.functional.softplus(raw_noise_std) + noise_epsilon
	acc = torch.zeros(e, device=device, dtype=dtype)
	for _ in range(n_mc):
		noise = torch.randn(b, e, device=device, dtype=dtype)
		noisy = clean_logits + noise * std
		_, topi = torch.topk(noisy, top_k, dim=1)
		mask = torch.zeros_like(noisy)
		mask.scatter_(1, topi, 1.0)
		acc = acc + mask.sum(dim=0)
	return acc / float(n_mc * b)


def _append_epoch_noise_csv(path, rowdict, fieldnames):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	write_header = not os.path.exists(path)
	with open(path, 'a', newline='', encoding='utf-8') as f:
		w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
		if write_header:
			w.writeheader()
		w.writerow(rowdict)


def _noise_moe_csv_fieldnames(top_k):
	"""noise_top1 CSV: required columns only; noise_top2 keeps extended log."""
	if top_k == 1:
		return [
			'epoch',
			'loss',
			'loss_structure',
			'loss_color',
			'aux_loss',
			'importance_loss',
			'load_loss',
			'hit_rate_0',
			'hit_rate_1',
			'hit_rate_2',
			'hit_rate_3',
			'importance_0',
			'importance_1',
			'importance_2',
			'importance_3',
			'top1_expert_count_0',
			'top1_expert_count_1',
			'top1_expert_count_2',
			'top1_expert_count_3',
			'lambda_aux_applied',
			'in_aux_warmup',
		]
	return [
		'epoch',
		'loss',
		'loss_structure',
		'loss_color',
		'aux_loss',
		'importance_loss',
		'load_loss',
		'epoch_mean_hit_rate',
		'epoch_mean_importance',
		'hit_rate_0',
		'hit_rate_1',
		'hit_rate_2',
		'hit_rate_3',
		'importance_0',
		'importance_1',
		'importance_2',
		'importance_3',
		'expert_slot_count_0',
		'expert_slot_count_1',
		'expert_slot_count_2',
		'expert_slot_count_3',
		'top_k',
		'top2_expert_activation_count_0',
		'top2_expert_activation_count_1',
		'top2_expert_activation_count_2',
		'top2_expert_activation_count_3',
	]


def _save_fixed_noise_moe_samples(
	model_F, model_A, device, vis_dir, ir_dir, names, out_dir, epoch_display, begin_epoch, epoch,
):
	os.makedirs(out_dir, exist_ok=True)
	model_F.eval()
	model_A.eval()
	prev_det = None
	if hasattr(model_F, 'moe_block') and hasattr(model_F.moe_block, 'deterministic_inference'):
		prev_det = model_F.moe_block.deterministic_inference
		model_F.moe_block.deterministic_inference = True
	with torch.no_grad():
		for name in names:
			vp = os.path.join(vis_dir, name)
			ip = os.path.join(ir_dir, name)
			if not (os.path.isfile(vp) and os.path.isfile(ip)):
				continue
			from PIL import Image
			import numpy as np
			import torchvision.transforms as T
			tot = T.ToTensor()
			vis = tot(Image.open(vp).convert('RGB')).unsqueeze(0).to(device)
			ir_g = tot(Image.open(ip).convert('L')).unsqueeze(0).to(device)
			ir_rgb = ir_g.repeat(1, 3, 1, 1)
			a1, b1, a2, b2, a3, b3, r1, r2 = model_A(vis)
			fused, _, _ = model_F(torch.cat((vis, ir_rgb), 1), a1, b1, a2, b2, a3, b3, r1, r2, modulation=False)
			base = os.path.splitext(name)[0]
			torchvision.utils.save_image(vis, os.path.join(out_dir, f'fixed_epoch_{epoch_display:03d}_{base}_source1_vis.png'))
			torchvision.utils.save_image(ir_rgb, os.path.join(out_dir, f'fixed_epoch_{epoch_display:03d}_{base}_source2_ir_rgb.png'))
			torchvision.utils.save_image(fused ** 0.7, os.path.join(out_dir, f'fixed_epoch_{epoch_display:03d}_{base}_fused.png'))
	if prev_det is not None:
		model_F.moe_block.deterministic_inference = prev_det
	model_F.train()


def train_one_epoch_noise_moe(
	loaders,
	model_R_vis,
	model_R_ir,
	model_A,
	model_F,
	model_F_vis,
	optimizer,
	vgg16,
	epoch,
	num_epochs,
	begin_time,
	begin_epoch,
	device,
	top_k,
	lambda_aux,
	mc_samples,
	noise_epsilon,
	in_aux_warmup=0,
):
	model_R_vis = model_R_vis.eval()
	model_R_ir = model_R_ir.eval()
	model_F_vis = model_F_vis.eval()
	model_F.train()

	total_i = len(loaders['train'])
	sum_loss = sum_ls = sum_lc = sum_aux = sum_imp = sum_ld = 0.0
	n_batches = 0
	hit_counts = torch.zeros(4, device=device)
	total_samples = 0
	importance_epoch = torch.zeros(4, device=device)

	for i, sample in enumerate(loaders['train']):
		source1_batch_ori = sample['img1'].to(device)
		source2_y = sample['img2'].to(device)
		source2_cb = torch.full_like(source2_y, 0.5).to(device)
		source2_cr = torch.full_like(source2_y, 0.5).to(device)
		source2_batch = torch.cat((source2_y, source2_cb, source2_cr), 1)
		source2_batch = ycbcr2rgb(source2_batch)

		perm = torch.randperm(3).to(device)
		source1_batch = source1_batch_ori[:, perm, :, :]

		batchsize = source1_batch.size(0)
		brightness_factors = (0.5 + torch.rand(batchsize, 1, 1, 1)).to(device)
		source1_batch = torch.clamp(source1_batch ** brightness_factors, 0, 1)

		optimizer.zero_grad()
		a1, b1, a2, b2, a3, b3, r1, r2 = model_A(source1_batch)

		fused_vis_batch, _, _ = model_F_vis(
			torch.cat((source1_batch, source1_batch), 1), a1, b1, a2, b2, a3, b3, r1, r2, modulation=False
		)
		fused_batch, _, aux = model_F(
			torch.cat((source1_batch, source2_batch), 1), a1, b1, a2, b2, a3, b3, r1, r2, modulation=False
		)

		gate = aux['gate']
		clean = aux['clean_logits']
		raw_std = aux['raw_noise_std']

		importance = gate.sum(dim=0)
		load_vec = _probabilistic_topk_load_mc(
			clean.detach(), raw_std.detach(), top_k, mc_samples, noise_epsilon
		)
		imp_loss = _cv_squared(importance)
		ld_loss = _cv_squared(load_vec)
		aux_loss = lambda_aux * (imp_loss + ld_loss)

		vgg_layers = ['0']
		x = fused_batch
		y = source1_batch
		z = source2_batch
		for name, layer in vgg16._modules.items():
			x = layer(x)
			y = layer(y)
			z = layer(z)
			if name in vgg_layers:
				f_stru_feas = x
				s1_stru_feas = y
				s2_stru_feas = z

		s1_stru_feas = nn.functional.leaky_relu(s1_stru_feas)
		s2_stru_feas = nn.functional.leaky_relu(s2_stru_feas)
		f_stru_feas = nn.functional.leaky_relu(f_stru_feas)

		crit_l2 = nn.MSELoss()
		crit_l1 = nn.L1Loss()
		s1_fea_grad = gradient_operator(s1_stru_feas)
		s2_fea_grad = gradient_operator(s2_stru_feas)
		fea_grad_max_mask = (s1_fea_grad * 1.5 > s2_fea_grad).float()
		ex_fuse_fea = s1_stru_feas * fea_grad_max_mask + s2_stru_feas * (1 - fea_grad_max_mask)
		loss_structure = crit_l1(f_stru_feas, ex_fuse_fea)

		fused_vis_batch_ycbcr = rgb2ycbcr(fused_vis_batch)
		fused_batch_ycbcr = rgb2ycbcr(fused_batch)
		loss_color = crit_l2(fused_vis_batch_ycbcr[:, 1, :, :], fused_batch_ycbcr[:, 1, :, :]) + crit_l2(
			fused_vis_batch_ycbcr[:, 2, :, :], fused_batch_ycbcr[:, 2, :, :]
		)

		loss = loss_structure + 0.5 * loss_color + aux_loss
		if not math.isfinite(loss.item()):
			raise RuntimeError(
				f'Non-finite loss at epoch {epoch + begin_epoch + 1} batch {i}: loss={loss.item()}'
			)
		loss.backward()
		optimizer.step()

		bs = source1_batch.size(0)
		total_samples += bs
		if top_k == 1:
			_, sel = torch.max(gate, dim=1)
			for e in range(4):
				hit_counts[e] += (sel == e).float().sum()
		else:
			for e in range(4):
				hit_counts[e] += (gate[:, e] > 1e-4).float().sum()

		importance_epoch += importance

		sum_loss += loss.item()
		sum_ls += loss_structure.item()
		sum_lc += loss_color.item()
		sum_aux += aux_loss.item()
		sum_imp += imp_loss.item()
		sum_ld += ld_loss.item()
		n_batches += 1

	epoch_display = epoch + begin_epoch + 1
	mean_hit = (hit_counts / max(total_samples, 1)).detach().cpu().tolist()
	mean_imp = (importance_epoch / max(total_samples, 1)).detach().cpu().tolist()
	topk_expert_count = [int(hit_counts[j].item()) for j in range(4)]
	epoch_mean_importance = float(sum(mean_imp) / 4.0)
	avg_loss = sum_loss / max(n_batches, 1)
	avg_ls = sum_ls / max(n_batches, 1)
	avg_lc = sum_lc / max(n_batches, 1)
	avg_aux = sum_aux / max(n_batches, 1)
	avg_imp = sum_imp / max(n_batches, 1)
	avg_ld = sum_ld / max(n_batches, 1)

	if top_k == 1:
		hit_list = ', '.join(f'{mean_hit[j]:.5f}' for j in range(4))
		imp_list = ', '.join(f'{mean_imp[j]:.5f}' for j in range(4))
		print(f'Epoch {epoch_display} finished')
		print(f'lambda_aux_applied = {lambda_aux:.6f}')
		print(f'aux_warmup_active = {bool(in_aux_warmup)}')
		print(f'loss = {avg_loss:.6f}')
		print(f'loss_structure = {avg_ls:.6f}')
		print(f'loss_color = {avg_lc:.6f}')
		print(f'aux_loss = {avg_aux:.6f}')
		print(f'importance_loss = {avg_imp:.6f}')
		print(f'load_loss = {avg_ld:.6f}')
		print(f'epoch_mean_hit_rate = [{hit_list}]')
		print(f'epoch_mean_importance = [{imp_list}]')
		print(f'top1_expert_count = {topk_expert_count}')
		print(f'time_elapsed = {datetime.now() - begin_time}', flush=True)
	else:
		print(
			f'[noise_moe epoch {epoch_display}/{num_epochs}] batches={n_batches} '
			f'loss={avg_loss:.5f} '
			f'loss_structure={avg_ls:.5f} loss_color={avg_lc:.5f} '
			f'aux_loss={avg_aux:.5f} '
			f'importance_loss={avg_imp:.5f} load_loss={avg_ld:.5f} '
			f'epoch_mean_hit_rate={sum(mean_hit) / 4.0:.5f} epoch_mean_importance={epoch_mean_importance:.5f} '
			f'top_k={top_k} expert_slot_counts={topk_expert_count} '
			f'time={datetime.now() - begin_time}',
			flush=True,
		)

	row = {
		'epoch': epoch_display,
		'loss': sum_loss / max(n_batches, 1),
		'loss_structure': sum_ls / max(n_batches, 1),
		'loss_color': sum_lc / max(n_batches, 1),
		'aux_loss': sum_aux / max(n_batches, 1),
		'importance_loss': sum_imp / max(n_batches, 1),
		'load_loss': sum_ld / max(n_batches, 1),
		'epoch_mean_hit_rate': sum(mean_hit) / 4.0,
		'epoch_mean_importance': epoch_mean_importance,
		'hit_rate_0': mean_hit[0],
		'hit_rate_1': mean_hit[1],
		'hit_rate_2': mean_hit[2],
		'hit_rate_3': mean_hit[3],
		'importance_0': mean_imp[0],
		'importance_1': mean_imp[1],
		'importance_2': mean_imp[2],
		'importance_3': mean_imp[3],
		'expert_slot_count_0': topk_expert_count[0],
		'expert_slot_count_1': topk_expert_count[1],
		'expert_slot_count_2': topk_expert_count[2],
		'expert_slot_count_3': topk_expert_count[3],
		'top_k': top_k,
	}
	if top_k == 1:
		for j in range(4):
			row[f'top1_expert_count_{j}'] = topk_expert_count[j]
		row['lambda_aux_applied'] = float(lambda_aux)
		row['in_aux_warmup'] = int(in_aux_warmup)
		if total_samples > 0:
			max_share = max(topk_expert_count) / float(total_samples)
			if max_share >= 0.99:
				print(
					f'WARNING: possible expert collapse — max top1 share this epoch = {max_share:.4f}',
					flush=True,
				)
	else:
		for j in range(4):
			row[f'top2_expert_activation_count_{j}'] = topk_expert_count[j]
	return row


def train(loaders, model_R_vis, model_R_ir, model_A, model_F, model_F_vis, optimizer, writer, epoch, num_epochs, begin_time, begin_epoch, device):
	model_R_vis = model_R_vis.eval()
	model_R_ir = model_R_ir.eval()
	model_F_vis = model_F_vis.eval()

	vgg16 = models.vgg16(pretrained=True).features
	vgg16.cuda().eval()
	vgg16.to(device)

	print(f'--- Epoch {epoch + begin_epoch + 1} ---')
	total_i = len(loaders['train'])

	for i, sample in enumerate(loaders['train']):
		source1_batch_ori = sample['img1'].to(device)
		source2_y = sample['img2'].to(device)
		source2_cb = torch.full_like(source2_y, 0.5).to(device)
		source2_cr = torch.full_like(source2_y, 0.5).to(device)
		source2_batch = torch.cat((source2_y, source2_cb, source2_cr), 1)
		source2_batch = ycbcr2rgb(source2_batch)


		perm = torch.randperm(3).to(device)
		source1_batch = source1_batch_ori[:, perm, :, :]

		batchsize = source1_batch.size(0)
		patchsize = source1_batch.size(2)

		brightness_factors = (0.5 + torch.rand(batchsize, 1, 1, 1)).to(device)
		source1_batch = source1_batch ** brightness_factors
		source1_batch = torch.clamp(source1_batch, 0, 1)

		optimizer.zero_grad()

		a1, b1, a2, b2, a3, b3, r1, r2 = model_A(source1_batch)

		fused_vis_batch, _ = model_F_vis(torch.cat((source1_batch, source1_batch), 1), a1, b1, a2, b2, a3, b3, r1, r2,
								 modulation=False)

		fused_batch, _ = model_F(torch.cat((source1_batch, source2_batch), 1), a1, b1, a2, b2, a3, b3, r1, r2, modulation=False)


		vgg_layers = ['0']
		x = fused_batch
		y = source1_batch
		z = source2_batch

		for name, layer in vgg16._modules.items():
			x = layer(x)
			y = layer(y)
			z = layer(z)
			if name in vgg_layers:
				f_stru_feas = x
				s1_stru_feas = y
				s2_stru_feas = z

		s1_stru_feas = nn.functional.leaky_relu(s1_stru_feas)
		s2_stru_feas = nn.functional.leaky_relu(s2_stru_feas)
		f_stru_feas = nn.functional.leaky_relu(f_stru_feas)

		crit_l2 = nn.MSELoss()
		crit_l1 = nn.L1Loss()


		s1_fea_grad = gradient_operator(s1_stru_feas)
		s2_fea_grad = gradient_operator(s2_stru_feas)
		fea_grad_max_mask = (s1_fea_grad * 1.5 > s2_fea_grad).float()

		ex_fuse_fea = s1_stru_feas * fea_grad_max_mask + s2_stru_feas * (1 - fea_grad_max_mask)

		loss_structure = crit_l1(f_stru_feas, ex_fuse_fea)

		fused_vis_batch_ycbcr = rgb2ycbcr(fused_vis_batch)
		fused_batch_ycbcr = rgb2ycbcr(fused_batch)

		loss_color = crit_l2(fused_vis_batch_ycbcr[:,1,:,:], fused_batch_ycbcr[:,1,:,:]) + crit_l2(fused_vis_batch_ycbcr[:,2,:,:], fused_batch_ycbcr[:,2,:,:])
		loss = loss_structure + loss_color * 0.5
		loss_vis = crit_l2(fused_batch, source1_batch)

		loss.backward(retain_graph=True)
		optimizer.step()

		timeElapsed = datetime.now() - begin_time

		if (i+1) % 5 == 0:
			print(
				'Epoch: [%d/%d], Iter: [%d/%d], Loss: %.5f, Time: ' % (epoch + 1 + begin_epoch, num_epochs, i + 1, total_i, loss.item()),
				timeElapsed)
			step = i + 1 + total_i * (begin_epoch + epoch)
			writer.add_scalar(tag="loss", scalar_value=loss, global_step=step)
			writer.add_scalar(tag="loss_structure", scalar_value=loss_structure, global_step=step)
			writer.add_scalar(tag="loss_color", scalar_value=loss_color, global_step=step)
			writer.add_scalar(tag="loss_vis", scalar_value=loss_vis, global_step=step)
			writer.add_image("source1", torchvision.utils.make_grid(source1_batch[0:3, :, :, :]), global_step=step)
			writer.add_image("source2", torchvision.utils.make_grid(source2_batch[0:3, :, :, :]), global_step=step)
			p = np.random.randint(0, 61)
			writer.add_image("fea_ex_fuse", torchvision.utils.make_grid(ex_fuse_fea[0:3, p:p + 3, :, :]),
							 global_step=step)
			writer.add_image("fea_vgg", torchvision.utils.make_grid(f_stru_feas[0:3, p:p + 3, :, :]), global_step=step)
			writer.add_image("fused_img", torchvision.utils.make_grid(fused_batch[0:3, :, :, :] ** 0.7),
							 global_step=step)
			writer.add_image("fused_vis", torchvision.utils.make_grid(fused_vis_batch[0:3, :, :, :]),
							 global_step=step)
			writer.add_image("a", torchvision.utils.make_grid(source1_batch[0:3, :, :, :]), global_step=step)
		prev = datetime.now()

args = parse_args()

if torch.cuda.is_available():
	device = torch.device(f'cuda:{args.device}')
	torch.cuda.manual_seed(1234)
else:
	device = torch.device('cpu')
	torch.manual_seed(1234)

basedir = args.baseDir
train_dir = os.path.join(basedir, 'train')

trans_to_tensor = transforms.ToTensor()
trans_crop = transforms.RandomCrop(
	args.patchsize, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'
)
trans_compose = transforms.Compose([trans_to_tensor, trans_crop])
train_dataset = SICE_F_stru(train_dir, transform=trans_compose, weather_aug_p=args.weather_aug_p)
trainloader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, pin_memory=True)

loaders = {'train': trainloader}
hp = dict(lr=1e-4, wd=0, lr_decay_factor=0.998)

model_R_vis = Structure_Encoder()
model_R_vis.to(device)
model_R_ir = Structure_Encoder()
model_R_ir.to(device)

model_A = A2V_Encoder()
model_A.to(device)

if args.fusion_model == 'noise_top1':
	model_F = FusionNetWithNoiseTop1MoE()
	model_F_vis = FusionNetWithNoiseTop1MoE()
	top_k = 1
elif args.fusion_model == 'noise_top2':
	model_F = FusionNetWithNoiseTop2MoE()
	model_F_vis = FusionNetWithNoiseTop2MoE()
	top_k = 2
else:
	model_F = FusionNet()
	model_F_vis = FusionNet()
	top_k = None

model_F.to(device)
model_F_vis.to(device)

if hasattr(model_F, 'moe_block'):
	model_F.moe_block.noise_epsilon = float(args.noise_epsilon)
	model_F_vis.moe_block.noise_epsilon = float(args.noise_epsilon)

optimizer = optim.Adam(model_F.parameters(), lr=hp['lr'], weight_decay=hp['wd'])

experiment = args.experiment
num_epochs = args.numEpoch
checkpoint_path = '../train-jobs/ckpt/' + experiment + '_ckpt.pth'

checkpoint_stru_vis = torch.load(
	'../train-jobs/ckpt/' + args.structure_vis_ckpt + '_ckpt.pth', map_location=device
)
model_R_vis.load_state_dict(checkpoint_stru_vis['model_R_state_dict'])
checkpoint_stru_ir = torch.load(
	'../train-jobs/ckpt/' + args.structure_ir_ckpt + '_ckpt.pth', map_location=device
)
model_R_ir.load_state_dict(checkpoint_stru_ir['model_R_state_dict'])

if os.path.exists(checkpoint_path):
	print('---Continue Training---')
	checkpoint = torch.load(checkpoint_path, map_location=device)
	model_F.load_state_dict(checkpoint['model_F_state_dict'])
	begin_epoch = checkpoint['epoch']
	print('begin epoch: ', begin_epoch + 1)
else:
	begin_epoch = 0

print(
	f'[START TRAINING JOB] -{experiment} fusion_model={args.fusion_model} on '
	f'{datetime.now().strftime("%b %d %Y %H:%M:%S")}'
)
begin_time = datetime.now()

if args.fusion_model in ('noise_top1', 'noise_top2'):
	vgg16 = models.vgg16(pretrained=True).features
	vgg16.to(device).eval()
	if args.fusion_model == 'noise_top1':
		warm_n = max(0, int(args.aux_warmup_epochs))
		if warm_n > 0:
			print(
				f'noise_top1 stabilized: epochs 1..{warm_n} use lambda_aux_top1_warmup='
				f'{args.lambda_aux_top1_warmup}, then lambda_aux={args.lambda_aux} | '
				f'load_mc_samples={args.mc_samples}',
				flush=True,
			)
		else:
			print(
				f'noise_top1: aux warmup disabled (aux_warmup_epochs=0); lambda_aux={args.lambda_aux} | '
				f'load_mc_samples={args.mc_samples}',
				flush=True,
			)
	metrics_dir = os.path.join('../train-jobs/metrics_logs', experiment)
	os.makedirs(metrics_dir, exist_ok=True)
	csv_path = os.path.join(metrics_dir, f'epoch_{experiment}_summary.csv')
	last_json_path = os.path.join(metrics_dir, f'last_epoch_{experiment}.json')
	fixed_dir = os.path.join('../train-jobs/fixed_samples', experiment)
	test_vis_dir = args.fixed_test_vis_dir if args.fixed_test_vis_dir else os.path.join(
		args.testDir.rstrip(os.sep), 'vis'
	)
	test_ir_dir = os.path.join(os.path.dirname(test_vis_dir.rstrip(os.sep)), 'ir')
	fixed_names = []
	if os.path.isdir(test_vis_dir):
		for fn in sorted(os.listdir(test_vis_dir)):
			low = fn.lower()
			if low.endswith(('.png', '.jpg', '.jpeg')):
				fixed_names.append(fn)
			if len(fixed_names) >= 3:
				break
	fieldnames = _noise_moe_csv_fieldnames(top_k)
	for epoch in range(num_epochs - begin_epoch):
		epoch_display = epoch + begin_epoch + 1
		if args.fusion_model == 'noise_top1':
			warm_n = max(0, int(args.aux_warmup_epochs))
			if warm_n > 0 and epoch_display <= warm_n:
				lambda_aux_epoch = float(args.lambda_aux_top1_warmup)
				in_warm = 1
			else:
				lambda_aux_epoch = float(args.lambda_aux)
				in_warm = 0
		else:
			lambda_aux_epoch = float(args.lambda_aux)
			in_warm = 0
		stats = train_one_epoch_noise_moe(
			loaders,
			model_R_vis,
			model_R_ir,
			model_A,
			model_F,
			model_F_vis,
			optimizer,
			vgg16,
			epoch,
			num_epochs,
			begin_time,
			begin_epoch,
			device,
			top_k,
			lambda_aux_epoch,
			args.mc_samples,
			args.noise_epsilon,
			in_aux_warmup=in_warm,
		)
		try:
			_append_epoch_noise_csv(csv_path, stats, fieldnames)
		except OSError as e:
			raise RuntimeError(f'epoch CSV write failed ({csv_path}): {e}') from e
		try:
			with open(last_json_path, 'w', encoding='utf-8') as jf:
				json.dump(
					{k: (float(v) if isinstance(v, (float, int)) else v) for k, v in stats.items()},
					jf,
					indent=2,
				)
		except OSError as e:
			raise RuntimeError(f'epoch JSON write failed ({last_json_path}): {e}') from e
		if fixed_names:
			_save_fixed_noise_moe_samples(
				model_F,
				model_A,
				device,
				test_vis_dir,
				test_ir_dir,
				fixed_names,
				fixed_dir,
				stats['epoch'],
				begin_epoch,
				epoch,
			)
		state_dict = {'model_F_state_dict': model_F.state_dict(), 'epoch': epoch + 1 + begin_epoch}
		try:
			torch.save(state_dict, checkpoint_path)
		except Exception as e:
			raise RuntimeError(f'checkpoint save failed ({checkpoint_path}): {e}') from e
else:
	writer = SummaryWriter(log_dir='../train-jobs/log/' + experiment)
	loss_history, best_loss = [], float('inf')
	for epoch in range(num_epochs - begin_epoch):
		train(
			loaders,
			model_R_vis,
			model_R_ir,
			model_A,
			model_F,
			model_F_vis,
			optimizer,
			writer,
			epoch,
			num_epochs,
			begin_time,
			begin_epoch,
			device,
		)
		state_dict = {'model_F_state_dict': model_F.state_dict(), 'epoch': epoch + 1 + begin_epoch}
		torch.save(state_dict, checkpoint_path)
