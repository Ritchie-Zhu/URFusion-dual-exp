"""
Train C_vis_gray on M3FD VIS Y-channel pairs.

Loss weights match vis-ir/code/train_content_extractor_vis.py (input domain only changed).
C input: [B,1,H,W] Y. VGG auxiliary branch: gray_to_vgg3(Y).
"""

import argparse
import os
from datetime import datetime

import torch
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import M3FDVisYPairs
from losses.content_extractor_loss import compute_content_extractor_loss_vis
from models import StructureEncoderGray
from utils import resolve_latest_ckpt_path, save_epoch_interval_ckpt, save_latest_ckpt, write_epoch_logs


def parse_args():
	parser = argparse.ArgumentParser(description='Train gray VIS content encoder (C_vis_gray)')
	parser.add_argument('--device', type=int, default=0)
	parser.add_argument('--experiment', default='content_encoder_vis_gray')
	parser.add_argument(
		'--baseDir',
		type=str,
		default='../../vis-ir/dataset/M3FD/',
		help='M3FD root; uses baseDir/train/VIS',
	)
	parser.add_argument('--numEpoch', type=int, default=20)
	parser.add_argument('--patchsize', type=int, default=160)
	parser.add_argument('--batchsize', type=int, default=12)
	parser.add_argument('--ckptRoot', type=str, default='../train-jobs/ckpt')
	parser.add_argument('--logRoot', type=str, default='../train-jobs/metrics_logs')
	parser.add_argument('--ckpt_interval', type=int, default=10, help='Save epoch snapshot every N epochs')
	parser.add_argument('--num_workers', type=int, default=0)
	parser.add_argument('--cudnn_benchmark', action='store_true')
	return parser.parse_args()


def train_one_epoch(loaders, model_R, vgg16, optimizer_R, device):
	model_R.train()
	sum_loss = sum_con = sum_feat = sum_sim = 0.0
	n_batches = 0

	for sample in loaders['train']:
		optimizer_R.zero_grad()
		x1 = sample['img1'].to(device)
		x2 = sample['img2'].to(device)
		if x1.size(1) != 1:
			raise ValueError(f'C_vis_gray expects 1-channel Y, got shape {tuple(x1.shape)}')

		stats = compute_content_extractor_loss_vis(model_R, vgg16, x1, x2)
		stats['loss'].backward()
		optimizer_R.step()

		n_batches += 1
		sum_loss += float(stats['loss'].item())
		sum_con += float(stats['loss_contrastive'].item())
		sum_feat += float(stats['loss_feature'].item())
		sum_sim += float(stats['loss_sim'].item())

	nb = max(n_batches, 1)
	return {
		'loss': sum_loss / nb,
		'loss_contrastive': sum_con / nb,
		'loss_feature': sum_feat / nb,
		'loss_sim': sum_sim / nb,
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
	trans_compose = transforms.Compose([
		transforms.ToTensor(),
		transforms.RandomCrop(args.patchsize),
	])
	train_dataset = M3FDVisYPairs(train_dir, transform=trans_compose)
	trainloader = DataLoader(
		train_dataset,
		batch_size=args.batchsize,
		shuffle=True,
		pin_memory=True,
		num_workers=int(args.num_workers),
		persistent_workers=(int(args.num_workers) > 0),
	)
	loaders = {'train': trainloader}

	model_R = StructureEncoderGray(in_channels=1, out_channels=64).to(device)
	optimizer_R = optim.Adam(model_R.parameters(), lr=1e-4, weight_decay=0)

	vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.to(device).eval()

	experiment = args.experiment
	os.makedirs(args.ckptRoot, exist_ok=True)
	checkpoint_path = resolve_latest_ckpt_path(args.ckptRoot, experiment)
	log_dir = os.path.join(args.logRoot, experiment)

	begin_epoch = 0
	if os.path.exists(checkpoint_path):
		print('---Continue Training---')
		checkpoint = torch.load(checkpoint_path, map_location=device)
		model_R.load_state_dict(checkpoint['model_R_state_dict'])
		begin_epoch = checkpoint['epoch']
		print('begin epoch:', begin_epoch + 1)

	print(f'[START] {experiment} | data={os.path.abspath(train_dir)}')
	print(f'  latest_ckpt={os.path.abspath(checkpoint_path)}')
	print(f'  epoch_ckpt_dir={os.path.abspath(os.path.join(args.ckptRoot, experiment))}/epoch_{{NNN}}/ckpt.pth (every {args.ckpt_interval} epochs)')
	begin_time = datetime.now()

	for epoch in range(args.numEpoch - begin_epoch):
		row = train_one_epoch(loaders, model_R, vgg16, optimizer_R, device)
		epoch_display = epoch + begin_epoch + 1
		row['epoch'] = int(epoch_display)
		print(
			f'[C_vis_gray] epoch {epoch_display}/{args.numEpoch} '
			f'loss={row["loss"]:.6f} loss_feature={row["loss_feature"]:.6f} '
			f'loss_contrastive={row["loss_contrastive"]:.6f} loss_sim={row["loss_sim"]:.6f} '
			f'elapsed={datetime.now() - begin_time}',
			flush=True,
		)
		state_dict = {'model_R_state_dict': model_R.state_dict(), 'epoch': epoch_display}
		save_latest_ckpt(state_dict, args.ckptRoot, experiment)
		epoch_ckpt = save_epoch_interval_ckpt(
			state_dict, args.ckptRoot, experiment, epoch_display, interval=args.ckpt_interval
		)
		if epoch_ckpt is not None:
			print(f'  saved epoch snapshot: {epoch_ckpt}', flush=True)
		write_epoch_logs(log_dir, experiment, row)

	print(f'[DONE] {experiment}')


if __name__ == '__main__':
	main()
