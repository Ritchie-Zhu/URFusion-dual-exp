"""Print 6-col + 25-col between two official eval jsons."""
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
	parser.add_argument('--ref', required=True)
	parser.add_argument('--ours_name', default='ours')
	parser.add_argument('--ref_name', default='ref')
	parser.add_argument('--dataset', required=True)
	parser.add_argument('--out_dir', required=True)
	parser.add_argument('--tag', default='vs_ref')
	args = parser.parse_args()

	ours = json.load(open(args.ours, encoding='utf-8'))
	ref = json.load(open(args.ref, encoding='utf-8'))
	os.makedirs(args.out_dir, exist_ok=True)
	n = int(ours.get('n_images', 0))
	a_name = args.ours_name
	b_name = args.ref_name
	lines = [f'{args.dataset} n={n}  {a_name} vs {b_name}']
	lines.append('--- 6-col (higher better) ---')
	lines.append(f"{'metric':>12}  {a_name:>12}  {b_name:>12}  {'delta':>10}")
	six_lines = list(lines)
	for m in SIX:
		a = float(ours[f'{m}_mean'])
		b = float(ref[f'{m}_mean'])
		row = f"{m:>12}  {a:12.4f}  {b:12.4f}  {a - b:+10.4f}"
		lines.append(row)
		six_lines.append(row)
	lines.append('--- 25-col ---')
	lines.append(f"{'metric':>12}  {a_name:>12}  {b_name:>12}  {'delta':>10}  note")
	ours_better = 0
	for m in NAMES:
		a = float(ours[f'{m}_mean'])
		b = float(ref[f'{m}_mean'])
		d = a - b
		if m in LOWER_BETTER:
			win = d < 0
			note = 'lower better'
		else:
			win = d > 0
			note = 'higher better'
		if win:
			ours_better += 1
		mark = a_name if win else b_name
		lines.append(f"{m:>12}  {a:12.4f}  {b:12.4f}  {d:+10.4f}  {note}  {mark}")
	lines.append(f'{a_name} better on {ours_better}/25 cells')
	text = '\n'.join(lines)
	print(text)
	open(os.path.join(args.out_dir, f'six_col_{args.tag}.txt'), 'w', encoding='utf-8').write(
		'\n'.join(six_lines) + '\n'
	)
	open(os.path.join(args.out_dir, f'all25_{args.tag}.txt'), 'w', encoding='utf-8').write(text + '\n')


if __name__ == '__main__':
	main()
