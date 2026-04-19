"""
Run MSRS-style fusion test for noise_top2 experiment (FusionNetWithNoiseTop2MoE).
Same data path and A2V/vis.mat usage as test.py.
"""
import argparse
import os
import time

import numpy as np
import scipy.io
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import SICE_TEST
from model import FusionNetWithNoiseTop2MoE, A2V_Encoder
from utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--testDir', type=str, default='../dataset/test/')
    parser.add_argument('--task', type=str, default='')
    parser.add_argument(
        '--fusion_ckpt',
        type=str,
        default='noise_top2',
        help='Checkpoint prefix under ../train-jobs/ckpt/<prefix>_ckpt.pth',
    )
    parser.add_argument('--A2V_ckpt', type=str, default='A2V-msrs')
    parser.add_argument('--outputDir', type=str, default='../results/noise_top2_test_fused/')
    parser.add_argument(
        '--vis_degrade',
        type=str,
        default='none',
        choices=('none', 'haze', 'heavy_haze'),
    )
    return parser.parse_args()


args = parse_args()

if torch.cuda.is_available():
    device = torch.device(f'cuda:{args.device}')
    torch.cuda.manual_seed(1234)
else:
    device = torch.device('cpu')
    torch.manual_seed(1234)

model_F = FusionNetWithNoiseTop2MoE()
model_A = A2V_Encoder()
model_F.to(device)
model_F.eval()
model_A.to(device)
model_A.eval()

ckpt_path = os.path.join('../train-jobs/ckpt/', args.fusion_ckpt + '_ckpt.pth')
fusion_checkpoint = torch.load(ckpt_path, map_location=device)
fusion_epoch = fusion_checkpoint['epoch']
model_F.load_state_dict(fusion_checkpoint['model_F_state_dict'])
print('Loaded fusion checkpoint:', os.path.abspath(ckpt_path))
print('fusion epoch:', fusion_epoch)

A2V_checkpoint = torch.load('../train-jobs/ckpt/' + args.A2V_ckpt + '_ckpt.pth', map_location=device)
A2V_epoch = A2V_checkpoint['epoch']
model_A.load_state_dict(A2V_checkpoint['model_A_state_dict'])
print('A2V epoch:', A2V_epoch)

test_dataset = SICE_TEST(args.testDir + args.task, transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

os.makedirs(args.outputDir + args.task, exist_ok=True)

with torch.no_grad():
    vector_data = scipy.io.loadmat('../train-jobs/vis.mat')
    centroid = vector_data['centroid']
    centroid = torch.tensor(centroid).to(device)
    a1 = centroid[:, 0:1]
    b1 = centroid[:, 1:2]
    a2 = centroid[:, 2:3]
    b2 = centroid[:, 3:4]
    a3 = centroid[:, 4:5]
    b3 = centroid[:, 5:6]
    r1 = centroid[:, 6:7]
    r2 = centroid[:, 7:8]

    times = []
    for i, sample in enumerate(test_loader):
        t0 = time.time()
        names = sample['name']
        source1 = sample['img1'].to(device)
        if args.vis_degrade == 'haze':
            t, A = 0.55, 0.88
            source1 = torch.clamp(source1 * t + A * (1.0 - t), 0.0, 1.0)
        elif args.vis_degrade == 'heavy_haze':
            t, A = 0.35, 0.92
            source1 = torch.clamp(source1 * t + A * (1.0 - t), 0.0, 1.0)
        source2 = sample['img2'].to(device)
        source2 = source2.repeat(1, 3, 1, 1)
        fused_img, _, _ = model_F(
            torch.cat((source1, source2), 1),
            a1, b1, a2, b2, a3, b3, r1, r2,
            modulation=True,
        )
        times.append(time.time() - t0)
        for name in names:
            print(name)
            torchvision.utils.save_image(fused_img, os.path.join(args.outputDir + args.task, name))

    print('average time:', float(np.mean(times)) if times else 0.0)
