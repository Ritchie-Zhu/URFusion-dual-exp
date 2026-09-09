import os
import subprocess
import time

PROJECT = '/root/autodl-tmp/URFusion-main'
PY = '/root/autodl-tmp/conda/envs/urfusion/bin/python'
TEST_ONE = '/tmp/llvip_one'
OUT = '/tmp/llvip_smoke'
os.makedirs(OUT, exist_ok=True)
os.makedirs(f'{TEST_ONE}/vis', exist_ok=True)
os.makedirs(f'{TEST_ONE}/ir', exist_ok=True)

test_full = f'{PROJECT}/datasets/LLVIP/test'
for sub in ('vis', 'ir'):
	src = os.path.join(test_full, sub, '190001.jpg')
	dst = os.path.join(TEST_ONE, sub, '190001.jpg')
	if not os.path.exists(dst):
		subprocess.run(['cp', src, dst], check=True)

os.makedirs('/tmp/llvip_textif/eval/Visible', exist_ok=True)
os.makedirs('/tmp/llvip_textif/eval/Infrared', exist_ok=True)
for sub, src in (('Visible', 'vis'), ('Infrared', 'ir')):
	dst = f'/tmp/llvip_textif/eval/{sub}/190001.jpg'
	if os.path.lexists(dst):
		os.remove(dst)
	os.symlink(f'{TEST_ONE}/{src}/190001.jpg', dst)


def run(cmd, cwd):
	t0 = time.time()
	proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
	dt = time.time() - t0
	tail = (proc.stdout + proc.stderr).strip().splitlines()[-4:]
	return proc.returncode == 0, dt, tail


jobs = [
	(
		'MUFusion',
		run(
			[
				PY,
				'-c',
				"import test_m3fd, sys; sys.argv=['x','--dataset','LLVIP','--test_dir','/tmp/llvip_one','--output_dir','/tmp/llvip_smoke/MUFusion']; test_m3fd.main()",
			],
			f'{PROJECT}/MUFusion/ir_vis',
		),
	),
	(
		'U2Fusion',
		run(
			[
				PY,
				'-c',
				"import test_m3fd, sys; sys.argv=['x','--dataset','LLVIP','--test_dir','/tmp/llvip_one','--output_dir','/tmp/llvip_smoke/U2Fusion']; test_m3fd.main()",
			],
			f'{PROJECT}/U2Fusion',
		),
	),
	(
		'Fusion_training_1',
		run(
			[
				PY,
				'test_fusion_gray.py',
				'--dataset',
				'LLVIP',
				'--test_dir',
				TEST_ONE,
				'--output_root',
				OUT,
				'--experiment',
				'Fusion_training_1',
			],
			f'{PROJECT}/our_model_1_DualMoE/code',
		),
	),
	(
		'MetaFusion',
		run(
			[
				PY,
				'-c',
				"import test_m3fd, sys; sys.argv=['x','--test_ir_root','/tmp/llvip_one/ir','--test_vis_root','/tmp/llvip_one/vis','--save_path','/tmp/llvip_smoke/MetaFusion']; test_m3fd.main()",
			],
			f'{PROJECT}/MetaFusion',
		),
	),
	(
		'EMMA',
		run(
			[
				PY,
				'-c',
				f"""
import os, sys, time, cv2, torch
sys.path.insert(0, '{PROJECT}/EMMA')
from nets.Ufuser import Ufuser
from test_msrs import fuse_one
model = Ufuser().cuda().eval()
model.load_state_dict(torch.load('{PROJECT}/EMMA/model/EMMA.pth', map_location='cuda'))
t0 = time.time()
out = fuse_one(model, 'cuda', '{TEST_ONE}/ir/190001.jpg', '{TEST_ONE}/vis/190001.jpg')
os.makedirs('/tmp/llvip_smoke/EMMA', exist_ok=True)
cv2.imwrite('/tmp/llvip_smoke/EMMA/190001.png', out)
print('elapsed', time.time() - t0)
""",
			],
			PROJECT,
		),
	),
	(
		'Text-IF',
		run(
			[
				PY,
				'-c',
				"import test_msrs, sys; sys.argv=['x','--dataset_path','/tmp/llvip_textif/eval','--save_path','/tmp/llvip_smoke/Text-IF','--limit','1']; test_msrs.main()",
			],
			f'{PROJECT}/Text-IF',
		),
	),
	(
		'URFusion',
		run(
			[
				PY,
				'test_M3FD.py',
				'--fusion_model',
				'fusionnet',
				'--fusion_pth',
				'../train-jobs/ckpt/content-fusion-msrs_ckpt.pth',
				'--vis_mat',
				'../train-jobs/vis.mat',
				'--testDir',
				TEST_ONE + '/',
				'--out_name',
				'llvip_smoke_urfusion',
			],
			f'{PROJECT}/vis-ir/code',
		),
	),
]

print('LLVIP single-image smoke tests (1280x1024):')
results = {}
for name, (ok, dt, tail) in jobs:
	status = 'OK' if ok else 'FAIL'
	results[name] = (ok, dt)
	print(f'  {name:18s} {status:4s}  {dt:6.2f}s')
	for line in tail:
		print(f'    {line}')

n_full = 3463
print('\nEstimated full LLVIP inference (3463 images, rough):')
for name, (ok, dt) in results.items():
	if ok:
		total_h = dt * n_full / 3600
		print(f'  {name:18s} ~{dt*n_full/60:.0f} min ({total_h:.1f} h) @ {dt:.2f}s/img')
