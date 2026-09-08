"""
Fast Tier-A IVIF metrics (10 core metrics, CPU-optimized).

Metrics: EN, SD, SF, AG, MI, SSIM, Qabf, SCD, VIF, Nabf
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from scipy.stats import pearsonr
from sewar.full_ref import vifp
from skimage.metrics import structural_similarity

FAST_METRIC_NAMES = [
	'EN', 'SD', 'SF', 'AG', 'MI',
	'SSIM', 'Qabf', 'SCD', 'VIF', 'Nabf',
]

_EPS = 1e-12


def _read_gray_u8(path: str) -> np.ndarray:
	img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
	if img is None:
		raise FileNotFoundError(path)
	if img.ndim == 3:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	return np.clip(img, 0, 255).astype(np.uint8)


def _read_rgb_gray_u8(path: str) -> np.ndarray:
	img = cv2.imread(path, cv2.IMREAD_COLOR)
	if img is None:
		raise FileNotFoundError(path)
	return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _match_shape_u8(a: np.ndarray, ref: np.ndarray) -> np.ndarray:
	if a.shape == ref.shape:
		return a
	return cv2.resize(a, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_LINEAR)


def _entropy_shannon_u8(gray_u8: np.ndarray) -> float:
	hist, _ = np.histogram(gray_u8.ravel(), bins=256, range=(0, 256))
	p = hist.astype(np.float64)
	p = p[p > 0] / (p.sum() + _EPS)
	return float(-np.sum(p * np.log2(p)))


def _mutual_information_u8(a_u8: np.ndarray, b_u8: np.ndarray) -> float:
	h2, _, _ = np.histogram2d(
		a_u8.ravel(), b_u8.ravel(), bins=256, range=[[0, 256], [0, 256]]
	)
	pxy = h2.astype(np.float64)
	s = pxy.sum()
	if s <= 0:
		return 0.0
	pxy /= s
	px = pxy.sum(axis=1)
	py = pxy.sum(axis=0)
	mask = pxy > 0
	px_m = px[:, None]
	py_m = py[None, :]
	mi = np.sum(
		pxy[mask] * np.log2(pxy[mask] / (px_m * py_m + _EPS)[mask] + _EPS)
	)
	return float(mi)


def _avg_gradient(gray_u8: np.ndarray) -> float:
	g = gray_u8.astype(np.float64)
	gx = np.diff(g, axis=1)[:-1, :]
	gy = np.diff(g, axis=0)[:, :-1]
	return float(np.mean(np.sqrt(gx * gx + gy * gy) / np.sqrt(2.0)))


def _spatial_frequency(gray_u8: np.ndarray) -> float:
	g = gray_u8.astype(np.float64)
	rf = np.diff(g, axis=0)
	cf = np.diff(g, axis=1)
	return float(np.sqrt(np.mean(rf * rf) + np.mean(cf * cf)))


def _sobel_mag(gray_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	g = gray_u8.astype(np.float64)
	gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
	gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
	return np.abs(gx), np.abs(gy)


def _qabf_ratio(gx, gxf):
	out = np.empty_like(gx, dtype=np.float64)
	mask_gt = gx > gxf
	mask_eq = gx == gxf
	out[mask_gt] = gxf[mask_gt] / (gx[mask_gt] + _EPS)
	out[mask_eq] = 1.0
	mask_lt = ~(mask_gt | mask_eq)
	out[mask_lt] = gx[mask_lt] / (gxf[mask_lt] + _EPS)
	return out


def _qabf_pair(gx, gy, gxf, gyf):
	return float(np.mean(_qabf_ratio(gx, gxf) * _qabf_ratio(gy, gyf)))


def _qabf(a_u8: np.ndarray, b_u8: np.ndarray, f_u8: np.ndarray) -> float:
	ga, aa = _sobel_mag(a_u8)
	gb, ab = _sobel_mag(b_u8)
	gf, af = _sobel_mag(f_u8)
	return _qabf_pair(ga, aa, gf, af) + _qabf_pair(gb, ab, gf, af)


def _scd(a_u8: np.ndarray, b_u8: np.ndarray, f_u8: np.ndarray) -> float:
	df_a = f_u8.astype(np.float64) - a_u8.astype(np.float64)
	df_b = f_u8.astype(np.float64) - b_u8.astype(np.float64)
	c1 = pearsonr(df_a.ravel(), b_u8.ravel())[0]
	c2 = pearsonr(df_b.ravel(), a_u8.ravel())[0]
	if np.isnan(c1):
		c1 = 0.0
	if np.isnan(c2):
		c2 = 0.0
	return float(c1 + c2)


def _nabf(a_u8: np.ndarray, b_u8: np.ndarray, f_u8: np.ndarray) -> float:
	da = np.abs(f_u8.astype(np.float64) - a_u8.astype(np.float64))
	db = np.abs(f_u8.astype(np.float64) - b_u8.astype(np.float64))
	return float((da.mean() + db.mean()) / 2.0)


def _vif(a_u8: np.ndarray, b_u8: np.ndarray, f_u8: np.ndarray) -> float:
	try:
		v1 = float(vifp(a_u8, f_u8))
		v2 = float(vifp(b_u8, f_u8))
		return (v1 + v2) / 2.0
	except Exception:
		return 0.0


def evaluation_one_fast(ir_path: str, vi_path: str, f_path: str):
	"""Returns dict metric_name -> value."""
	a_u8 = _read_rgb_gray_u8(vi_path)
	b_u8 = _read_gray_u8(ir_path)
	f_u8 = _read_rgb_gray_u8(f_path)
	b_u8 = _match_shape_u8(b_u8, f_u8)
	a_u8 = _match_shape_u8(a_u8, f_u8)

	en = _entropy_shannon_u8(f_u8)
	sd = float(np.std(f_u8.astype(np.float64)))
	sf = _spatial_frequency(f_u8)
	ag = _avg_gradient(f_u8)
	mi = _mutual_information_u8(a_u8, f_u8) + _mutual_information_u8(b_u8, f_u8)
	ssim = (
		structural_similarity(a_u8, f_u8, data_range=255)
		+ structural_similarity(b_u8, f_u8, data_range=255)
	) / 2.0
	qabf = _qabf(a_u8, b_u8, f_u8)
	scd = _scd(a_u8, b_u8, f_u8)
	vif = _vif(a_u8, b_u8, f_u8)
	nabf = _nabf(a_u8, b_u8, f_u8)

	return {
		'EN': en,
		'SD': sd,
		'SF': sf,
		'AG': ag,
		'MI': mi,
		'SSIM': float(ssim),
		'Qabf': qabf,
		'SCD': scd,
		'VIF': vif,
		'Nabf': nabf,
	}
