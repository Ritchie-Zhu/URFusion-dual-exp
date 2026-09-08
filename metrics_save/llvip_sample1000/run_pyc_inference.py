"""Load and run test_m3fd.cpython-310.pyc from method repos (MUFusion / U2Fusion / MetaFusion)."""

import argparse
import importlib.machinery
import importlib.util
import os
import sys


def load_pyc_module(work_dir: str, module_name: str = 'test_m3fd'):
	pyc_path = os.path.join(work_dir, '__pycache__', f'{module_name}.cpython-310.pyc')
	fake_py = os.path.join(work_dir, f'{module_name}.py')
	if not os.path.isfile(pyc_path):
		raise FileNotFoundError(f'Missing pyc: {pyc_path}')
	loader = importlib.machinery.SourcelessFileLoader(module_name, pyc_path)
	spec = importlib.util.spec_from_loader(module_name, loader)
	mod = importlib.util.module_from_spec(spec)
	mod.__file__ = fake_py
	sys.modules[module_name] = mod
	loader.exec_module(mod)
	return mod


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--work_dir', required=True)
	parser.add_argument('extra_args', nargs=argparse.REMAINDER)
	args = parser.parse_args()

	work_dir = os.path.abspath(args.work_dir)
	os.chdir(work_dir)
	sys.path.insert(0, work_dir)

	argv = [os.path.join(work_dir, 'test_m3fd.py')]
	if args.extra_args:
		if args.extra_args[0] == '--':
			argv.extend(args.extra_args[1:])
		else:
			argv.extend(args.extra_args)
	sys.argv = argv

	mod = load_pyc_module(work_dir)
	if not hasattr(mod, 'main'):
		raise RuntimeError('pyc module has no main()')
	mod.main()


if __name__ == '__main__':
	main()
