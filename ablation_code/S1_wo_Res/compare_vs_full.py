"""Print 6-col + 25-col vs SpatialResMoE_3 from official eval json."""
import argparse
import json
import os

SIX = ['NMI', 'Qy', 'MI', 'VIF', 'Qabf', 'VIFF']
NAMES = [
	'CE', 'NMI', 'QNCIE', 'TE', 'EI', 'Qy', 'Qcb', 'EN', 'MI', 'SF', 'AG', 'SD',
	'CC', 'SCD', 'VIF', 'MSE', 'PSNR', 'Qabf', 'Nabf', 'SSIM', 'MS_SSIM', 'VIFF',
	'NIQE', 'BRISQUE', 'MUSIQ',
]
LOWER_BETTER = {'CE', 'MSE', 'Nabf', 'NIQE', 'BRISQUE'}


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--ours', required=True)
	parser.add_argument('--full', required=True)
	parser.add_argument('--dataset', required=True)
	parser.add_argument('--out_dir', required=True)
	args = parser.parse_args()

	ours = json.load(open(args.ours, encoding='utf-8'))
	full = json.load(open(args.full, encoding='utf-8'))
	os.makedirs(args.out_dir, exist_ok=True)
	n = int(ours.get('n_images', 0))
	lines = [f'{args.dataset} n={n}  Abl_SR_wo_Res vs SpatialResMoE_3 (Full)']
	lines.append('--- 6-col (higher better) ---')
	lines.append(f"{'metric':>12}  {'S1':>10}  {'Full':>10}  {'delta':>10}")
	six_lines = list(lines)
	for m in SIX:
		a = float(ours[f'{m}_mean'])
		b = float(full[f'{m}_mean'])
		row = f"{m:>12}  {a:10.4f}  {b:10.4f}  {a - b:+10.4f}"
		lines.append(row)
		six_lines.append(row)
	lines.append('--- 25-col ---')
	lines.append(f"{'metric':>12}  {'S1':>10}  {'Full':>10}  {'delta':>10}  note")
	s1_better = 0
	for m in NAMES:
		a = float(ours[f'{m}_mean'])
		b = float(full[f'{m}_mean'])
		d = a - b
		if m in LOWER_BETTER:
			win = d < 0
			note = 'lower better'
		else:
			win = d > 0
			note = 'higher better'
		if win:
			s1_better += 1
		mark = 'S1' if win else 'Full'
		lines.append(f"{m:>12}  {a:10.4f}  {b:10.4f}  {d:+10.4f}  {note}  {mark}")
	lines.append(f'S1 better on {s1_better}/25 cells')
	text = '\n'.join(lines)
	print(text)
	open(os.path.join(args.out_dir, 'six_col.txt'), 'w', encoding='utf-8').write('\n'.join(six_lines) + '\n')
	open(os.path.join(args.out_dir, 'all25.txt'), 'w', encoding='utf-8').write(text + '\n')


if __name__ == '__main__':
	main()
