"""Print main-table 6 metrics vs Full and A2 from official eval json."""

import argparse
import json
import os

SIX = ['NMI', 'Qy', 'MI', 'VIF', 'Qabf', 'VIFF']


def load_means(path):
	with open(path, encoding='utf-8') as f:
		data = json.load(f)
	return {k: float(data[f'{k}_mean']) for k in SIX}, int(data.get('n_images', 0))


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--ours', required=True)
	parser.add_argument('--full', default='')
	parser.add_argument('--a2', default='')
	parser.add_argument('--name', default='SpatialResMoE_3')
	parser.add_argument('--dataset', default='MSRS')
	args = parser.parse_args()

	ours, n = load_means(args.ours)
	refs = []
	if args.full and os.path.isfile(args.full):
		refs.append(('Fusion_noleak_1', load_means(args.full)[0]))
	if args.a2 and os.path.isfile(args.a2):
		refs.append(('Abl_wo_MoE', load_means(args.a2)[0]))

	cols = ['metric', args.name] + [n for n, _ in refs]
	print(f'{args.dataset} n={n}  (higher is better)')
	print('  '.join(f'{c:>16}' for c in cols))
	for m in SIX:
		row = [f'{m:>16}', f'{ours[m]:16.4f}']
		for _, vals in refs:
			delta = ours[m] - vals[m]
			row.append(f'{vals[m]:8.4f} ({delta:+.4f})')
		print('  '.join(row))


if __name__ == '__main__':
	main()
