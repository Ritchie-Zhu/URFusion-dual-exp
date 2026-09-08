"""LRRNet IVIF inference wrapper with VIS-chroma RGB output."""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torchvision.utils
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from tqdm import tqdm

PROJECT = '/root/autodl-tmp/URFusion-main'
LRRNET_DIR = os.path.join(PROJECT, 'LRRNet')
ORCH_DIR = os.path.join(PROJECT, 'metrics_save/new_methods_gifnet_lrrnet_sage')
CODE_DIR = os.path.join(PROJECT, 'vis-ir-gray/code')

sys.path.insert(0, LRRNET_DIR)
os.chdir(LRRNET_DIR)

from args import Args as args  # noqa: E402
from net_lista import LRR_NET  # noqa: E402
import utils  # noqa: E402

sys.path.insert(0, CODE_DIR)
from ycbcr import rgb_to_ycbcr_tensor, ycbcr_to_rgb_tensor  # noqa: E402

sys.path.insert(0, ORCH_DIR)
from common import list_paired_names, resolve_vis_ir_dirs  # noqa: E402

EPSILON = 1e-5
DEFAULT_CKPT = os.path.join(
	LRRNET_DIR,
	'model/final_lrr_net_lam2_1.5_wir_3.0_lam3_gram_2000_epoch_4_block_4.model',
)


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', required=True)
	parser.add_argument('--test_dir', required=True)
	parser.add_argument('--output_dir', required=True)
	parser.add_argument('--checkpoint', default=DEFAULT_CKPT)
	parser.add_argument('--num_block', type=int, default=4)
	parser.add_argument('--device', type=int, default=0)
	parser.add_argument('--force', action='store_true', help='overwrite existing outputs')
	return parser.parse_args()


def load_model(path, num_block, device):
	model_or = LRR_NET(args.s, args.n, args.channel, args.stride, num_block, 'cat')
	if num_block <= 4:
		model = model_or
	else:
		model = torch.nn.DataParallel(model_or, list(range(torch.cuda.device_count())))
	model.load_state_dict(torch.load(path, map_location='cpu'))
	model.eval()
	return model.to(device)


def tensor_to_y_uint8(fuse_tensor):
	"""Match LRRNet utils.save_image min-max normalization -> uint8 gray Y."""
	fuse = fuse_tensor.float()
	if fuse.is_cuda:
		fuse = fuse.cpu().data[0].numpy()
	else:
		fuse = fuse.clamp(0, 255).data[0].numpy()
	fuse = (fuse - np.min(fuse)) / (np.max(fuse) - np.min(fuse) + EPSILON)
	fuse = (fuse * 255.0).astype(np.float32)
	if fuse.shape[0] == 1:
		fuse = fuse[0]
	return fuse


def fuse_one_rgb(model, ir_path, vis_path, device, to_tensor):
	img_ir = utils.get_train_images(ir_path, height=None, width=None, flag=False)
	img_vi = utils.get_train_images(vis_path, height=None, width=None, flag=False)
	img_ir = Variable(img_ir, requires_grad=False).to(device)
	img_vi = Variable(img_vi, requires_grad=False).to(device)
	img_ir = utils.normalize_tensor(img_ir)
	img_vi = utils.normalize_tensor(img_vi)
	with torch.no_grad():
		out = model(img_ir, img_vi)['fuse']
	y_np = tensor_to_y_uint8(out)
	vis_rgb = to_tensor(Image.open(vis_path).convert('RGB')).unsqueeze(0).to(device)
	_, cb, cr = rgb_to_ycbcr_tensor(vis_rgb)
	# ycbcr_to_rgb_tensor expects Y/Cb/Cr in [0, 1], not [0, 255]
	y_t = torch.from_numpy(y_np).unsqueeze(0).unsqueeze(0).float().to(device) / 255.0
	rgb = ycbcr_to_rgb_tensor(y_t, cb, cr)
	return rgb


def main():
	cli = parse_args()
	os.makedirs(cli.output_dir, exist_ok=True)
	device = torch.device(f'cuda:{cli.device}' if torch.cuda.is_available() else 'cpu')
	vis_dir, ir_dir = resolve_vis_ir_dirs(cli.test_dir)
	names = list_paired_names(cli.test_dir)
	print(f'[LRRNet] {cli.dataset}: {len(names)} pairs -> {cli.output_dir}')

	model = load_model(cli.checkpoint, cli.num_block, device)
	to_tensor = transforms.ToTensor()
	t0 = time.time()
	done = skipped = 0
	for name in tqdm(names, desc=f'LRRNet_{cli.dataset}'):
		out_path = os.path.join(cli.output_dir, name)
		if (
			not cli.force
			and os.path.isfile(out_path)
			and os.path.getsize(out_path) > 512
		):
			skipped += 1
			continue
		ir_path = os.path.join(ir_dir, name)
		vis_path = os.path.join(vis_dir, name)
		rgb = fuse_one_rgb(model, ir_path, vis_path, device, to_tensor)
		torchvision.utils.save_image(rgb, out_path)
		done += 1

	elapsed = time.time() - t0
	print(
		f'[LRRNet DONE] saved={done} skipped={skipped} total={len(names)} '
		f'elapsed={elapsed:.1f}s -> {cli.output_dir}'
	)


if __name__ == '__main__':
	main()
