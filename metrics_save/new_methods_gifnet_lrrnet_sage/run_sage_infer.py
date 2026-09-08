"""SAGE student-network inference wrapper — does not modify SAGE/test.py."""

import argparse
import os
import shutil
import sys
import time

import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT = '/root/autodl-tmp/URFusion-main'
SAGE_DIR = os.path.join(PROJECT, 'SAGE')
ORCH_DIR = os.path.join(PROJECT, 'metrics_save/new_methods_gifnet_lrrnet_sage')
sys.path.insert(0, ORCH_DIR)
sys.path.insert(0, SAGE_DIR)
os.chdir(SAGE_DIR)

import utils  # noqa: E402
from dataset.dataset_test import Data  # noqa: E402
from model_sub.model import Network  # noqa: E402
from common import list_paired_names, resolve_vis_ir_dirs, sage_staging_dir  # noqa: E402


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', required=True)
	parser.add_argument('--test_dir', required=True)
	parser.add_argument('--output_dir', required=True)
	parser.add_argument('--checkpoint', default=os.path.join(SAGE_DIR, 'checkpoints/checkpoints.pt'))
	parser.add_argument('--device', type=int, default=0)
	return parser.parse_args()


def prepare_sage_layout(test_dir, dataset):
	staging = sage_staging_dir(dataset)
	ir_out = os.path.join(staging, 'Ir')
	vis_out = os.path.join(staging, 'Vis')
	if os.path.isdir(staging):
		shutil.rmtree(staging)
	os.makedirs(ir_out)
	os.makedirs(vis_out)
	vis_dir, ir_dir = resolve_vis_ir_dirs(test_dir)
	for name in list_paired_names(test_dir):
		os.symlink(os.path.abspath(os.path.join(ir_dir, name)), os.path.join(ir_out, name))
		os.symlink(os.path.abspath(os.path.join(vis_dir, name)), os.path.join(vis_out, name))
	return staging


def main():
	cli = parse_args()
	os.makedirs(cli.output_dir, exist_ok=True)
	staging = prepare_sage_layout(cli.test_dir, cli.dataset)
	n_expected = len(list_paired_names(cli.test_dir))
	device = torch.device(f'cuda:{cli.device}' if torch.cuda.is_available() else 'cpu')
	test_loader = DataLoader(Data(mode='test', img_dir=staging), batch_size=1, num_workers=0)
	model = Network()
	model.load_state_dict(torch.load(cli.checkpoint, map_location='cpu', weights_only=True))
	model.to(device).eval()
	print(f'[SAGE] {cli.dataset}: {n_expected} pairs -> {cli.output_dir}')
	t0 = time.time()
	with torch.no_grad():
		for data in tqdm(test_loader, desc=f'SAGE_{cli.dataset}'):
			names, exts = data['name'], data['ext']
			ir, y, cb, cr = data['ir'], data['y'], data['cb'], data['cr']
			ir, y, cb, cr = utils.togpu_4(device, ir, y, cb, cr)
			output, _ = model(y, ir)
			output_colored = utils.YCrCb2RGB(torch.cat((output, cb, cr), dim=1))
			for i, (name, ext) in enumerate(zip(names, exts)):
				torchvision.utils.save_image(output_colored[i : i + 1], os.path.join(cli.output_dir, f'{name}{ext}'))
	print(f'[SAGE DONE] {n_expected} in {time.time()-t0:.1f}s')


if __name__ == '__main__':
	main()
