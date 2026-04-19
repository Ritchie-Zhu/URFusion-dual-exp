"""
Offline metrics for IVIF fusion (same definitions as eval_msrs_gray / DenseMOE runs).
Requires: numpy, opencv-python, scikit-image.
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def _to_gray_01(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr.ndim == 2:
        g = img_bgr.astype(np.float32)
    else:
        g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = g / 255.0
    return np.clip(g, 0.0, 1.0)


def _entropy_shannon_log2(fused_u8: np.ndarray) -> float:
    hist, _ = np.histogram(fused_u8.ravel(), bins=256, range=(0, 256))
    p = hist.astype(np.float64)
    total = p.sum()
    if total <= 0:
        return 0.0
    p = p[p > 0] / total
    return float(-np.sum(p * np.log2(p)))


def _average_gradient_01(gray_01: np.ndarray) -> float:
    """AG: sqrt(gx^2+gy^2)/sqrt(2) on [0,1], mean; aligned to (H-1,W-1)."""
    g = gray_01.astype(np.float64)
    gx = np.diff(g, axis=1)  # (H, W-1)
    gy = np.diff(g, axis=0)  # (H-1, W)
    gx = gx[:-1, :]  # (H-1, W-1)
    gy = gy[:, :-1]  # (H-1, W-1)
    mag = np.sqrt(gx * gx + gy * gy) / np.sqrt(2.0)
    return float(np.mean(mag))


def _mutual_information(u8_a: np.ndarray, u8_b: np.ndarray) -> float:
    """256x256 joint histogram, MI in bits (log2)."""
    h2, _, _ = np.histogram2d(
        u8_a.ravel(),
        u8_b.ravel(),
        bins=256,
        range=[[0, 256], [0, 256]],
    )
    pxy = h2.astype(np.float64)
    s = pxy.sum()
    if s <= 0:
        return 0.0
    pxy /= s
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    mi = 0.0
    eps = 1e-12
    for i in range(256):
        for j in range(256):
            p = pxy[i, j]
            if p > 0:
                mi += p * (np.log2(p + eps) - np.log2(px[i] + eps) - np.log2(py[j] + eps))
    return float(mi)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--visible_dir', type=str, required=True)
    p.add_argument('--infrared_dir', type=str, required=True)
    p.add_argument('--fused_dir', type=str, required=True)
    p.add_argument('--output_dir', type=str, required=True)
    p.add_argument('--summary_name', type=str, default='metrics_summary.json')
    p.add_argument('--per_image_name', type=str, default='metrics_per_image.csv')
    return p.parse_args()


def main():
    args = parse_args()
    vis_dir = Path(args.visible_dir)
    ir_dir = Path(args.infrared_dir)
    fused_dir = Path(args.fused_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    names = sorted([f.name for f in fused_dir.iterdir() if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp')])
    vis_names = set(f.name for f in vis_dir.iterdir() if f.is_file())
    ir_names = set(f.name for f in ir_dir.iterdir() if f.is_file())
    missing_vis = [n for n in names if n not in vis_names]
    missing_ir = [n for n in names if n not in ir_names]
    if missing_vis or missing_ir:
        raise SystemExit(f'Missing paired files. vis missing: {missing_vis[:5]}... ir missing: {missing_ir[:5]}...')

    rows = []
    metrics_lists = {k: [] for k in ['EN', 'SD', 'AG', 'SSIM_vis', 'MI_vis', 'MI_ir']}

    for name in names:
        p_vis = vis_dir / name
        p_ir = ir_dir / name
        p_fu = fused_dir / name
        im_v = cv2.imread(str(p_vis), cv2.IMREAD_UNCHANGED)
        im_i = cv2.imread(str(p_ir), cv2.IMREAD_UNCHANGED)
        im_f = cv2.imread(str(p_fu), cv2.IMREAD_UNCHANGED)
        if im_v is None or im_i is None or im_f is None:
            raise SystemExit(f'Failed to read one of: {p_vis}, {p_ir}, {p_fu}')

        gv = _to_gray_01(im_v)
        gi = _to_gray_01(im_i)
        gf = _to_gray_01(im_f)

        fused_u8 = np.clip(np.round(gf * 255.0), 0, 255).astype(np.uint8)
        vis_u8 = np.clip(np.round(gv * 255.0), 0, 255).astype(np.uint8)
        ir_u8 = np.clip(np.round(gi * 255.0), 0, 255).astype(np.uint8)

        en = _entropy_shannon_log2(fused_u8)
        sd = float(np.std(gf))
        ag = _average_gradient_01(gf)
        ssim_v = float(ssim(gv, gf, data_range=1.0))
        mi_v = _mutual_information(fused_u8, vis_u8)
        mi_i = _mutual_information(fused_u8, ir_u8)

        rows.append((name, en, sd, ag, ssim_v, mi_v, mi_i))
        for k, v in zip(metrics_lists.keys(), [en, sd, ag, ssim_v, mi_v, mi_i]):
            metrics_lists[k].append(v)

    summary = {}
    key_map = {
        'EN': 'EN',
        'SD': 'SD',
        'AG': 'AG',
        'SSIM_vis': 'SSIM_vis',
        'MI_vis': 'MI_vis',
        'MI_ir': 'MI_ir',
    }
    for k in metrics_lists:
        arr = np.array(metrics_lists[k], dtype=np.float64)
        summary[f'{key_map[k]}_mean'] = float(arr.mean())
        summary[f'{key_map[k]}_std'] = float(arr.std(ddof=0))

    csv_path = out_dir / args.per_image_name
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('image_name,EN,SD,AG,SSIM_vis,MI_vis,MI_ir\n')
        for name, en, sd, ag, sv, mv, mi in rows:
            f.write(f'{name},{en:.8f},{sd:.8f},{ag:.8f},{sv:.8f},{mv:.8f},{mi:.8f}\n')

    json_path = out_dir / args.summary_name
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print('Saved per-image CSV:', csv_path)
    print('Saved summary JSON:', json_path)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
