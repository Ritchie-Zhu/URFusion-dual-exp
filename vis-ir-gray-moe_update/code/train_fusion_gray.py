"""Train spatial residual vis/ir MoE on training_noleak. Outputs stay under vis-ir-gray-moe_update."""

import argparse
import math
import os
import sys
from datetime import datetime

import cv2
import torch
import torch.optim as optim
import torchvision.models as models
import torchvision.utils
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

UPDATE_CODE = os.path.dirname(os.path.abspath(__file__))
UPDATE_ROOT = os.path.abspath(os.path.join(UPDATE_CODE, '..'))
PROJECT = os.path.abspath(os.path.join(UPDATE_ROOT, '..'))
BASE_CODE = os.path.join(PROJECT, 'vis-ir-gray', 'code')
sys.path.insert(0, BASE_CODE)
sys.path.insert(0, UPDATE_CODE)

from dataset import M3FDFusionTrain
from dual_spatial_moe_fusion import DualSpatialResMoENetGray
from spatial_moe_loss import compute_spatial_moe_loss
from train_health import format_health_line, health_flags
from utils import (
	gradient_operator,
	load_frozen_structure_encoder,
	resolve_latest_ckpt_path,
	save_epoch_interval_ckpt,
	save_latest_ckpt,
)
from ycbcr import rgb_to_ycbcr_tensor

DEFAULT_CKPT_ROOT = os.path.join(UPDATE_ROOT, 'train-jobs', 'ckpt')
DEFAULT_LOG_ROOT = os.path.join(UPDATE_ROOT, 'train-jobs', 'metrics_logs')
DEFAULT_FIXED_ROOT = os.path.join(UPDATE_ROOT, 'train-jobs', 'fixed_examples')
DEFAULT_CONTENT_ROOT = os.path.join(PROJECT, 'vis-ir-gray', 'train-jobs', 'ckpt')

PREFERRED_FIXED = [
	'00306.png',
	'02033.png',
	'00000.png',
	'00031.png',
	'01505.png',
	'01562.png',
	'03769.png',
	'04053.png',
]

FUSION_LOG_FIELDS = [
	'epoch',
	'loss',
	'loss_main',
	'loss_int',
	'loss_grad',
	'loss_branch',
	'loss_route',
	'g_mean',
	'g_std',
	'mask_mean',
	'mask_std',
	'e_base_l1',
	'res_vis',
	'res_ir',
	'cap_l1',
	'vis_u_l1',
	'ir_u_l1',
	'e_vis',
	'e_ir',
	'p_vis',
	'p_ir',
	'y_vis_mean',
	'y_vis_std',
	'ir_mean',
	'ir_std',
	'health',
]


def parse_args():
	parser = argparse.ArgumentParser(description='Train spatial residual MoE fusion')
	parser.add_argument('--device', type=int, default=0)
	parser.add_argument('--experiment', default='SpatialResMoE_3')
	parser.add_argument('--baseDir', type=str, default=os.path.join(PROJECT, 'datasets', 'training_noleak'))
	parser.add_argument('--testDir', type=str, default=os.path.join(PROJECT, 'datasets', 'M3FD', 'test'))
	parser.add_argument('--vis_content_ckpt', type=str, default='Vis_Content_noleak')
	parser.add_argument('--ir_content_ckpt', type=str, default='ir_Content_noleak')
	parser.add_argument('--vis_content_ckpt_path', type=str, default='')
	parser.add_argument('--ir_content_ckpt_path', type=str, default='')
	parser.add_argument('--content_ckpt_root', type=str, default=DEFAULT_CONTENT_ROOT)
	parser.add_argument('--numEpoch', type=int, default=80)
	parser.add_argument('--patchsize', type=int, default=160)
	parser.add_argument('--batchsize', type=int, default=12)
	parser.add_argument('--ckptRoot', type=str, default=DEFAULT_CKPT_ROOT)
	parser.add_argument('--logRoot', type=str, default=DEFAULT_LOG_ROOT)
	parser.add_argument('--fixedRoot', type=str, default=DEFAULT_FIXED_ROOT)
	parser.add_argument('--ckpt_interval', type=int, default=10)
	parser.add_argument('--num_workers', type=int, default=12)
	parser.add_argument('--cudnn_benchmark', action='store_true')
	parser.add_argument('--lambda_int', type=float, default=0.1)
	parser.add_argument('--lambda_grad', type=float, default=1.0)
	parser.add_argument('--lambda_branch', type=float, default=0.01)
	parser.add_argument('--lambda_route', type=float, default=0.05)
	parser.add_argument('--vis_grad_bias', type=float, default=1.5)
	parser.add_argument('--save_fixed_every', type=int, default=10)
	parser.add_argument('--n_fixed', type=int, default=8)
	parser.add_argument('--shutdown_on_finish', action='store_true')
	return parser.parse_args()


def write_epoch_logs(log_dir, experiment, row, fields):
	import csv
	import json

	os.makedirs(log_dir, exist_ok=True)
	csv_path = os.path.join(log_dir, f'epoch_{experiment}_summary.csv')
	json_path = os.path.join(log_dir, f'last_epoch_{experiment}.json')
	write_header = not os.path.exists(csv_path)
	with open(csv_path, 'a', newline='', encoding='utf-8') as f:
		w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
		if write_header:
			w.writeheader()
		w.writerow(row)
	serial = {}
	for k, v in row.items():
		if k == 'health':
			serial[k] = str(v)
		else:
			serial[k] = float(v)
	with open(json_path, 'w', encoding='utf-8') as f:
		json.dump(serial, f, indent=2)
	health_path = os.path.join(log_dir, 'health_epoch_trace.txt')
	with open(health_path, 'a', encoding='utf-8') as f:
		f.write(f"ep{int(row['epoch']):03d} {format_health_line(row)}\n")


def apply_vis_brightness_aug(vis_rgb, device):
	perm = torch.randperm(3, device=device)
	vis_rgb = vis_rgb[:, perm, :, :]
	batchsize = vis_rgb.size(0)
	brightness = 0.5 + torch.rand(batchsize, 1, 1, 1, device=device)
	return torch.clamp(vis_rgb ** brightness, 0.0, 1.0)


def collect_fixed_names(test_vis_dir, max_n=8):
	if not os.path.isdir(test_vis_dir):
		return []
	all_names = sorted(
		fn
		for fn in os.listdir(test_vis_dir)
		if fn.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
	)
	picked = []
	have = set(all_names)
	for name in PREFERRED_FIXED:
		if name in have and name not in picked:
			picked.append(name)
		if len(picked) >= max_n:
			return picked
	if all_names:
		step = max(1, len(all_names) // max_n)
		for name in all_names[::step]:
			if name not in picked:
				picked.append(name)
			if len(picked) >= max_n:
				break
	return picked[:max_n]


def save_fixed_y_samples(model_F, device, vis_dir, ir_dir, names, out_dir, epoch_display):
	epoch_dir = os.path.join(out_dir, f'epoch_{epoch_display:03d}')
	os.makedirs(epoch_dir, exist_ok=True)
	model_F.eval()
	to_tensor = transforms.ToTensor()
	with torch.no_grad():
		for name in names:
			vis_path = os.path.join(vis_dir, name)
			ir_path = os.path.join(ir_dir, name)
			if not (os.path.isfile(vis_path) and os.path.isfile(ir_path)):
				continue
			vis_rgb = to_tensor(Image.open(vis_path).convert('RGB')).unsqueeze(0).to(device)
			ir_gray = to_tensor(cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)).unsqueeze(0).to(device)
			y_vis, _, _ = rgb_to_ycbcr_tensor(vis_rgb)
			y_f, aux = model_F(y_vis, ir_gray)
			base = os.path.splitext(name)[0]
			torchvision.utils.save_image(y_f, os.path.join(epoch_dir, f'{base}_Y_fused.png'))
			torchvision.utils.save_image(aux['g'], os.path.join(epoch_dir, f'{base}_gate.png'))
	model_F.train()
	print(f'  saved {len(names)} fixed Y+gate -> {epoch_dir}', flush=True)


def _scalar(v):
	if torch.is_tensor(v):
		return float(v.detach().item())
	return float(v)


def train_one_epoch(loaders, model_F, c_vis_gray, c_ir_gray, vgg16, optimizer, device, args):
	model_F.train()
	acc = {k: 0.0 for k in FUSION_LOG_FIELDS if k not in ('epoch', 'health')}
	n_batches = 0

	for sample in loaders['train']:
		vis_rgb = sample['img1'].to(device)
		ir_gray = sample['img2'].to(device)
		if ir_gray.size(1) == 3:
			ir_gray = ir_gray[:, 0:1, :, :]
		vis_rgb = apply_vis_brightness_aug(vis_rgb, device)
		y_vis, _, _ = rgb_to_ycbcr_tensor(vis_rgb)

		optimizer.zero_grad()
		y_f, aux = model_F(y_vis, ir_gray)
		stats = compute_spatial_moe_loss(
			y_f,
			y_vis,
			ir_gray,
			aux,
			vgg16,
			c_vis_gray,
			c_ir_gray,
			gradient_operator,
			lambda_int=args.lambda_int,
			lambda_grad=args.lambda_grad,
			lambda_branch=args.lambda_branch,
			lambda_route=args.lambda_route,
			vis_grad_bias=args.vis_grad_bias,
		)
		loss = stats['loss']
		if not math.isfinite(float(loss.item())):
			raise RuntimeError(f'Non-finite loss: {loss.item()}')

		loss.backward()
		optimizer.step()

		n_batches += 1
		for key in acc:
			if key in stats:
				acc[key] += _scalar(stats[key])
		acc['y_vis_mean'] += float(y_vis.mean().item())
		acc['y_vis_std'] += float(y_vis.std(unbiased=False).item())
		acc['ir_mean'] += float(ir_gray.mean().item())
		acc['ir_std'] += float(ir_gray.std(unbiased=False).item())

	nb = max(n_batches, 1)
	row = {k: acc[k] / nb for k in acc}
	row['health'] = health_flags(row)
	return row


def main():
	args = parse_args()
	if torch.cuda.is_available():
		device = torch.device(f'cuda:{args.device}')
		torch.cuda.manual_seed(1234)
	else:
		device = torch.device('cpu')
		torch.manual_seed(1234)
	torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)

	train_dir = os.path.join(args.baseDir, 'train')
	trainloader = DataLoader(
		M3FDFusionTrain(
			train_dir,
			transform=transforms.Compose(
				[transforms.ToTensor(), transforms.RandomCrop(args.patchsize)]
			),
		),
		batch_size=args.batchsize,
		shuffle=True,
		pin_memory=True,
		num_workers=int(args.num_workers),
		persistent_workers=(int(args.num_workers) > 0),
	)

	model_F = DualSpatialResMoENetGray().to(device)
	optimizer = optim.Adam(model_F.parameters(), lr=1e-4, weight_decay=0)
	c_vis_gray = load_frozen_structure_encoder(
		args.content_ckpt_root,
		args.vis_content_ckpt,
		device,
		ckpt_path=args.vis_content_ckpt_path or None,
	)
	c_ir_gray = load_frozen_structure_encoder(
		args.content_ckpt_root,
		args.ir_content_ckpt,
		device,
		ckpt_path=args.ir_content_ckpt_path or None,
	)
	vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.to(device).eval()

	experiment = args.experiment
	os.makedirs(args.ckptRoot, exist_ok=True)
	checkpoint_path = resolve_latest_ckpt_path(args.ckptRoot, experiment)
	log_dir = os.path.join(args.logRoot, experiment)
	fixed_dir = os.path.join(args.fixedRoot, experiment)

	test_vis_dir = os.path.join(args.testDir, 'vis')
	if not os.path.isdir(test_vis_dir):
		test_vis_dir = os.path.join(args.testDir, 'VIS')
	test_ir_dir = os.path.join(os.path.dirname(test_vis_dir.rstrip(os.sep)), 'ir')
	if not os.path.isdir(test_ir_dir):
		test_ir_dir = os.path.join(os.path.dirname(test_vis_dir.rstrip(os.sep)), 'IR')
	fixed_names = collect_fixed_names(test_vis_dir, max_n=args.n_fixed)

	begin_epoch = 0
	if os.path.exists(checkpoint_path):
		print('---Continue Training---')
		checkpoint = torch.load(checkpoint_path, map_location=device)
		model_F.load_state_dict(checkpoint['model_F_state_dict'])
		begin_epoch = int(checkpoint['epoch'])
		print('begin epoch:', begin_epoch + 1)

	print(f'[START] {experiment} spatial residual MoE')
	print(f'  code={UPDATE_CODE}')
	print(f'  data={os.path.abspath(train_dir)}')
	print(f'  C={args.vis_content_ckpt} / {args.ir_content_ckpt}')
	print(f'  lambda_route={args.lambda_route} (no L_aux)')
	print(f'  ckpt={os.path.abspath(checkpoint_path)}')
	print(f'  fixed every {args.save_fixed_every} ep, n={len(fixed_names)}: {fixed_names}')
	print(
		'  HEALTH: OK | GATE_COLLAPSE g_mean<0.05|>0.95 | GATE_FLAT g_std<0.02 | '
		'UNIQUE_DEAD both unique L1<1e-3 | RESIDUAL_DEAD (res_vis+res_ir)<0.01*e_base'
	)
	begin_time = datetime.now()

	for epoch in range(args.numEpoch - begin_epoch):
		row = train_one_epoch(
			{'train': trainloader}, model_F, c_vis_gray, c_ir_gray, vgg16, optimizer, device, args
		)
		epoch_display = epoch + begin_epoch + 1
		row['epoch'] = int(epoch_display)
		print(
			f'[F_gray ep {epoch_display}/{args.numEpoch}] '
			f'loss={row["loss"]:.4f} L_main={row["loss_main"]:.4f} '
			f'L_int={row["loss_int"]:.4f} L_grad={row["loss_grad"]:.4f} '
			f'L_branch={row["loss_branch"]:.4f} L_route={row["loss_route"]:.4f} '
			f'elapsed={str(datetime.now() - begin_time).split(".")[0]}',
			flush=True,
		)
		print(format_health_line(row), flush=True)

		state_dict = {'model_F_state_dict': model_F.state_dict(), 'epoch': epoch_display}
		save_latest_ckpt(state_dict, args.ckptRoot, experiment)
		epoch_ckpt = save_epoch_interval_ckpt(
			state_dict, args.ckptRoot, experiment, epoch_display, interval=args.ckpt_interval
		)
		if epoch_ckpt is not None:
			print(f'  saved epoch snapshot: {epoch_ckpt}', flush=True)
		write_epoch_logs(log_dir, experiment, row, fields=FUSION_LOG_FIELDS)
		if args.save_fixed_every > 0 and fixed_names and epoch_display % args.save_fixed_every == 0:
			save_fixed_y_samples(
				model_F, device, test_vis_dir, test_ir_dir, fixed_names, fixed_dir, epoch_display
			)

	print(f'[DONE] {experiment}')
	if args.shutdown_on_finish:
		print('[SHUTDOWN] Training finished; shutting down instance via /usr/bin/shutdown', flush=True)
		os.system('/usr/bin/shutdown')


if __name__ == '__main__':
	main()
