#!/usr/bin/env python3
"""
Run eval_multi_time from Jun-26 bytecode cache.
Preloads Metric_torch + eval_torch from pyc (no .py on disk).
Does NOT modify metrics_py/*.py on disk.
"""
import importlib.machinery
import importlib.util
import os
import sys

METRICS_PY = '/root/autodl-tmp/URFusion-main/metrics_py'


def load_pyc(name: str):
	pyc = os.path.join(METRICS_PY, '__pycache__', f'{name}.cpython-310.pyc')
	if not os.path.isfile(pyc):
		raise FileNotFoundError(f'Missing bytecode cache: {pyc}')
	loader = importlib.machinery.SourcelessFileLoader(name, pyc)
	spec = importlib.util.spec_from_loader(name, loader)
	mod = importlib.util.module_from_spec(spec)
	mod.__file__ = os.path.join(METRICS_PY, f'{name}.py')
	sys.modules[name] = mod
	loader.exec_module(mod)
	return mod


def main():
	os.chdir(METRICS_PY)
	sys.path.insert(0, METRICS_PY)
	load_pyc('Metric_torch')
	load_pyc('eval_torch')
	mod = load_pyc('eval_multi_time')
	mod.main()


if __name__ == '__main__':
	main()
