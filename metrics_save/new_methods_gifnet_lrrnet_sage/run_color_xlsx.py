#!/usr/bin/env python3
"""Run color_methods_metrics on any dataset xlsx without editing that module."""
import importlib.machinery
import importlib.util
import os
import sys
from pathlib import Path

import argparse

PROJECT = Path('/root/autodl-tmp/URFusion-main')
METRICS_PY = PROJECT / 'metrics_py'


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', required=True, help='e.g. LLVIP, MSRS, DroneVehicle')
	return parser.parse_args()


def main():
	cli = parse_args()
	source = PROJECT / 'metrics_save' / f'{cli.dataset}.xlsx'
	output = PROJECT / 'metrics_save' / f'{cli.dataset}_colored.xlsx'

	sys.path.insert(0, str(METRICS_PY))
	os.chdir(METRICS_PY)

	pyc = METRICS_PY / '__pycache__' / 'color_methods_metrics.cpython-310.pyc'
	loader = importlib.machinery.SourcelessFileLoader('color_methods_metrics', str(pyc))
	spec = importlib.util.spec_from_loader('color_methods_metrics', loader)
	cm = importlib.util.module_from_spec(spec)
	cm.__file__ = str(METRICS_PY / 'color_methods_metrics.py')
	sys.modules['color_methods_metrics'] = cm
	loader.exec_module(cm)

	cm.SOURCE_XLSX = source
	cm.OUTPUT_XLSX = output
	cm.SHEET_NAME = 'MeanMetrics'
	cm.main()
	print(f'Colored excel -> {output}')


if __name__ == '__main__':
	main()
