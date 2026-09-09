"""Train (Abl_DualMoE_wo_MoE): dual PreNet + single expert. lambda_aux defaults to 0."""

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

ABLATION_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_CODE = os.path.abspath(os.path.join(ABLATION_DIR, '..', '..', 'our_model_1_DualMoE', 'code'))
sys.path.insert(0, BASE_CODE)
sys.path.insert(0, ABLATION_DIR)
os.chdir(ABLATION_DIR)

from dataset import M3FDFusionTrain
from dual_conv_fusion import DualConvFusionNetGray
from losses.gray_fusion_loss import compute_gray_fusion_loss
from utils import (
	gradient_operator,
	load_frozen_structure_encoder,
	resolve_latest_ckpt_path,
	save_epoch_interval_ckpt,
	save_latest_ckpt,
	write_epoch_logs,
)
from ycbcr import rgb_to_ycbcr_tensor

DEFAULT_CKPT_ROOT = os.path.abspath(os.path.join(BASE_CODE, '..', 'train-jobs', 'ckpt'))
DEFAULT_LOG_ROOT = os.path.abspath(os.path.join(BASE_CODE, '..', 'train-jobs', 'metrics_logs'))
DEFAULT_FIXED_ROOT = os.path.abspath(os.path.join(BASE_CODE, '..', 'train-jobs', 'fixed_examples'))

FUSION_LOG_FIELDS = [
	'epoch', 'loss', 'loss_main', 'loss_int', 'loss_grad', 'loss_branch', 'aux_loss',
	'importance_loss', 'load_loss',
	'hit_rate_0', 'hit_rate_1', 'hit_rate_2', 'hit_rate_3',
	'importance_0', 'importance_1', 'importance_2', 'importance_3',
	'e_vis', 'e_ir', 'p_vis', 'p_ir',
	'y_vis_mean', 'y_vis_std', 'ir_mean', 'ir_std',
]


def parse_args():
	parser = argparse.ArgumentParser(description=' dual PreNet + single expert (Abl_DualMoE_wo_MoE)')
	parser.add_argument('--device', type=int, default=0)
	parser.add_argument('--experiment', default='Abl_DualMoE_wo_MoE')
	parser.add_argument('--fusion_model', default='dual_conv', choices=['dual_conv'])
	parser.add_argument('--baseDir', type=str, default='../../datasets/training_noleak')
	parser.add_argument('--testDir', type=str, default='../../datasets/M3FD/test')
	parser.add_argument('--vis_content_ckpt', type=str, default='Vis_Content_noleak')
	parser.add_argument('--ir_content_ckpt', type=str, default='ir_Content_noleak')
	parser.add_argument('--vis_content_ckpt_path', type=str, default='')
	parser.add_argument('--ir_content_ckpt_path', type=str, default='')
	parser.add_argument('--content_ckpt_root', type=str, default=DEFAULT_CKPT_ROOT)
	parser.add_argument('--pretrained_ckpt', type=str, default='')
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
	parser.add_argument('--lambda_aux', type=float, default=0.0, help='MoE removed; keep 0')
	parser.add_argument('--vis_grad_bias', type=float, default=1.5)
	parser.add_argument('--mc_samples', type=int, default=4)
	parser.add_argument('--noise_epsilon', type=float, default=1e-2)
	parser.add_argument('--top_k', type=int, default=2)
	parser.add_argument('--save_fixed_every', type=int, default=20)
	parser.add_argument('--shutdown_on_finish', action='store_true')
	return parser.parse_args()


def apply_vis_brightness_aug(vis_rgb, device):
	perm = torch.randperm(3, device=device)
	vis_rgb = vis_rgb[:, perm, :, :]
	batchsize = vis_rgb.size(0)
	brightness = (0.5 + torch.rand(batchsize, 1, 1, 1, device=device))
	return torch.clamp(vis_rgb ** brightness, 0.0, 1.0)


def collect_fixed_names(test_vis_dir, max_n=3):
	names = []
	if not os.path.isdir(test_vis_dir):
		return names
	for fn in sorted(os.listdir(test_vis_dir)):
		if fn.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
			names.append(fn)
		if len(names) >= max_n:
			break
	return names


def save_fixed_y_samples(model_F, device, vis_dir, ir_dir, names, out_dir, epoch_display):
	os.makedirs(out_dir, exist_ok=True)
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
			y_f, _ = model_F(y_vis, ir_gray)
			base = os.path.splitext(name)[0]
			torchvision.utils.save_image(
				y_f, os.path.join(out_dir, f'fixed_epoch_{epoch_display:03d}_{base}_Y_fused.png')
			)
	model_F.train()


def build_model(args, device):
	return DualConvFusionNetGray().to(device)


def train_one_epoch(loaders, model_F, c_vis_gray, c_ir_gray, vgg16, optimizer, device, args):
	model_F.train()
	sum_loss = sum_main = sum_int = sum_grad = sum_branch = 0.0
	sum_aux = sum_imp = sum_ld = 0.0
	sum_e_vis = sum_e_ir = sum_p_vis = sum_p_ir = 0.0
	sum_y_mean = sum_y_std = sum_ir_mean = sum_ir_std = 0.0
	n_batches = 0
	total_samples = 0

	for sample in loaders['train']:
		vis_rgb = sample['img1'].to(device)
		ir_gray = sample['img2'].to(device)
		if ir_gray.size(1) == 3:
			ir_gray = ir_gray[:, 0:1, :, :]
		vis_rgb = apply_vis_brightness_aug(vis_rgb, device)
		y_vis, _, _ = rgb_to_ycbcr_tensor(vis_rgb)

		optimizer.zero_grad()
		y_f, aux = model_F(y_vis, ir_gray)
		stats = compute_gray_fusion_loss(
			y_f, y_vis, ir_gray, aux, vgg16, c_vis_gray, c_ir_gray, gradient_operator,
			lambda_int=args.lambda_int, lambda_grad=args.lambda_grad,
			lambda_branch=args.lambda_branch, lambda_aux=args.lambda_aux,
			vis_grad_bias=args.vis_grad_bias, top_k=args.top_k,
			mc_samples=args.mc_samples, noise_epsilon=args.noise_epsilon,
		)
		loss = stats['loss']
		if not math.isfinite(float(loss.item())):
			raise RuntimeError(f'Non-finite loss: {loss.item()}')

		loss.backward()
		optimizer.step()

		total_samples += y_vis.size(0)
		n_batches += 1
		sum_loss += float(loss.item())
		sum_main += float(stats['loss_main'].item())
		sum_int += float(stats['loss_int'].item())
		sum_grad += float(stats['loss_grad'].item())
		sum_branch += float(stats['loss_branch'].item())
		sum_aux += float(stats['aux_loss'].item())
		sum_imp += float(stats['importance_loss'].item())
		sum_ld += float(stats['load_loss'].item())
		sum_e_vis += float(stats['e_vis'].detach().item())
		sum_e_ir += float(stats['e_ir'].detach().item())
		sum_p_vis += float(stats['p_vis'].detach().item())
		sum_p_ir += float(stats['p_ir'].detach().item())
		sum_y_mean += float(y_vis.mean().item())
		sum_y_std += float(y_vis.std(unbiased=False).item())
		sum_ir_mean += float(ir_gray.mean().item())
		sum_ir_std += float(ir_gray.std(unbiased=False).item())

	nb = max(n_batches, 1)
	return {
		'loss': sum_loss / nb, 'loss_main': sum_main / nb, 'loss_int': sum_int / nb,
		'loss_grad': sum_grad / nb, 'loss_branch': sum_branch / nb, 'aux_loss': sum_aux / nb,
		'importance_loss': sum_imp / nb, 'load_loss': sum_ld / nb,
		'hit_rate_0': 1.0, 'hit_rate_1': 0.0, 'hit_rate_2': 0.0, 'hit_rate_3': 0.0,
		'importance_0': 0.0, 'importance_1': 0.0, 'importance_2': 0.0, 'importance_3': 0.0,
		'e_vis': sum_e_vis / nb, 'e_ir': sum_e_ir / nb,
		'p_vis': sum_p_vis / nb, 'p_ir': sum_p_ir / nb,
		'y_vis_mean': sum_y_mean / nb, 'y_vis_std': sum_y_std / nb,
		'ir_mean': sum_ir_mean / nb, 'ir_std': sum_ir_std / nb,
	}


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
	trans_compose = transforms.Compose([transforms.ToTensor(), transforms.RandomCrop(args.patchsize)])
	trainloader = DataLoader(
		M3FDFusionTrain(train_dir, transform=trans_compose),
		batch_size=args.batchsize, shuffle=True, pin_memory=True,
		num_workers=int(args.num_workers),
		persistent_workers=(int(args.num_workers) > 0),
	)

	model_F = build_model(args, device)
	optimizer = optim.Adam(model_F.parameters(), lr=1e-4, weight_decay=0)
	c_vis_gray = load_frozen_structure_encoder(
		args.content_ckpt_root, args.vis_content_ckpt, device,
		ckpt_path=args.vis_content_ckpt_path or None,
	)
	c_ir_gray = load_frozen_structure_encoder(
		args.content_ckpt_root, args.ir_content_ckpt, device,
		ckpt_path=args.ir_content_ckpt_path or None,
	)
	vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.to(device).eval()

	experiment = args.experiment
	os.makedirs(args.ckptRoot, exist_ok=True)
	checkpoint_path = resolve_latest_ckpt_path(args.ckptRoot, experiment)
	log_dir = os.path.join(args.logRoot, experiment)
	fixed_dir = os.path.join(args.fixedRoot, experiment)

	test_vis_dir = os.path.join(args.testDir, 'VIS')
	if not os.path.isdir(test_vis_dir):
		test_vis_dir = os.path.join(args.testDir, 'vis')
	test_ir_dir = os.path.join(os.path.dirname(test_vis_dir.rstrip(os.sep)), 'IR')
	if not os.path.isdir(test_ir_dir):
		test_ir_dir = os.path.join(os.path.dirname(test_vis_dir.rstrip(os.sep)), 'ir')
	fixed_names = collect_fixed_names(test_vis_dir)

	begin_epoch = 0
	if os.path.exists(checkpoint_path):
		print('---Continue Training---')
		checkpoint = torch.load(checkpoint_path, map_location=device)
		model_F.load_state_dict(checkpoint['model_F_state_dict'])
		begin_epoch = checkpoint['epoch']
		print('begin epoch:', begin_epoch + 1)
	elif args.pretrained_ckpt and os.path.isfile(args.pretrained_ckpt):
		print(f'---Warm-start from {args.pretrained_ckpt} (strict=False)---')
		pretrained = torch.load(args.pretrained_ckpt, map_location=device)
		state = pretrained.get('model_F_state_dict', pretrained)
		missing, unexpected = model_F.load_state_dict(state, strict=False)
		print(f'  missing keys: {len(missing)}, unexpected keys: {len(unexpected)}')

	print(f'[START A2] {experiment} | fusion_model={args.fusion_model}')
	print(f'  ablation={ABLATION_DIR}')
	print(f'  data={os.path.abspath(train_dir)}')
	print(f'  moe=False (single FusionExpertBlock), lambda_aux={args.lambda_aux}')
	print(f'  ckpt={os.path.abspath(checkpoint_path)}')
	begin_time = datetime.now()

	for epoch in range(args.numEpoch - begin_epoch):
		row = train_one_epoch({'train': trainloader}, model_F, c_vis_gray, c_ir_gray, vgg16, optimizer, device, args)
		epoch_display = epoch + begin_epoch + 1
		row['epoch'] = int(epoch_display)
		print(
			f'[F_gray ep {epoch_display}/{args.numEpoch}] loss={row["loss"]:.4f} '
			f'L_main={row["loss_main"]:.4f} L_int={row["loss_int"]:.4f} '
			f'L_grad={row["loss_grad"]:.4f} L_branch={row["loss_branch"]:.4f} '
			f'L_aux={row["aux_loss"]:.4f} p_vis={row["p_vis"]:.3f} p_ir={row["p_ir"]:.3f} '
			f'elapsed={str(datetime.now() - begin_time).split(".")[0]}',
			flush=True,
		)
		state_dict = {'model_F_state_dict': model_F.state_dict(), 'epoch': epoch_display}
		save_latest_ckpt(state_dict, args.ckptRoot, experiment)
		epoch_ckpt = save_epoch_interval_ckpt(state_dict, args.ckptRoot, experiment, epoch_display, interval=args.ckpt_interval)
		if epoch_ckpt is not None:
			print(f'  saved epoch snapshot: {epoch_ckpt}', flush=True)
		write_epoch_logs(log_dir, experiment, row, fields=FUSION_LOG_FIELDS)
		if args.save_fixed_every > 0 and fixed_names and epoch_display % args.save_fixed_every == 0:
			save_fixed_y_samples(model_F, device, test_vis_dir, test_ir_dir, fixed_names, fixed_dir, epoch_display)

	print(f'[DONE A2] {experiment}')
	if args.shutdown_on_finish:
		print('[SHUTDOWN] Training finished; shutting down instance via /usr/bin/shutdown', flush=True)
		os.system('/usr/bin/shutdown')


if __name__ == '__main__':
	main()
