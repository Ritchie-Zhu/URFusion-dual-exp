#!/usr/bin/env python3
"""Feasibility probe: 8-worker official evaluation_one vs existing Fusion_noleak_1 csv.

Does not append Excel and does not run ablations.
"""
from __future__ import annotations

import csv
import importlib.machinery
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

PROJECT = Path("/root/autodl-tmp/URFusion-main")
METRICS_PY = str(PROJECT / "metrics_py")
METRICS_SAVE = PROJECT / "metrics_save"
SIX = ["NMI", "Qy", "MI", "VIF", "Qabf", "VIFF"]
IQA = ["NIQE", "BRISQUE", "MUSIQ"]

_et = None

DATASETS = {
    "LLVIP": {
        "fused": PROJECT / "our_model_1_DualMoE/results/Fusion_noleak_1_LLVIP/RGB_fused",
        "vis": PROJECT / "datasets/LLVIP/test_sample1000/vis",
        "ir": PROJECT / "datasets/LLVIP/test_sample1000/ir",
        "csv": METRICS_SAVE / "Fusion_noleak_1_LLVIP/metrics_per_image.csv",
        "n": 32,
        "ref_s_per_img": 5.443,
    },
    "MSRS": {
        "fused": PROJECT / "our_model_1_DualMoE/results/Fusion_noleak_1_MSRS/RGB_fused",
        "vis": PROJECT / "datasets/MSRS/test/vis",
        "ir": PROJECT / "datasets/MSRS/test/ir",
        "csv": METRICS_SAVE / "Fusion_noleak_1_MSRS/metrics_per_image.csv",
        "n": 8,
        "ref_s_per_img": 1.354,
    },
    "M3FD": {
        "fused": PROJECT / "our_model_1_DualMoE/results/Fusion_noleak_1_M3FD/RGB_fused",
        "vis": PROJECT / "datasets/M3FD/test/vis",
        "ir": PROJECT / "datasets/M3FD/test/ir",
        "csv": METRICS_SAVE / "Fusion_noleak_1_M3FD/metrics_per_image.csv",
        "n": 8,
        "ref_s_per_img": 3.278,
    },
}


def load_pyc(name: str):
    pyc = os.path.join(METRICS_PY, "__pycache__", f"{name}.cpython-310.pyc")
    loader = importlib.machinery.SourcelessFileLoader(name, pyc)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = os.path.join(METRICS_PY, f"{name}.py")
    sys.modules[name] = mod
    loader.exec_module(mod)
    return mod


def to_float(value) -> float:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
    except Exception:
        pass
    return float(value)


def init_worker():
    global _et
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.chdir(METRICS_PY)
    if METRICS_PY not in sys.path:
        sys.path.insert(0, METRICS_PY)
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass
    load_pyc("Metric_torch")
    _et = load_pyc("eval_torch")


def eval_one(task):
    dataset, name, ir_path, vis_path, fused_path = task
    result = _et.evaluation_one(ir_path, vis_path, fused_path)
    row = {"dataset": dataset, "image_name": name}
    for key, value in zip(_et.STATIC_METRIC_NAMES, result):
        row[key] = to_float(value)
    return row


def load_ref_csv(path: Path) -> dict:
    rows = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["image_name"]] = row
    return rows


def list_names(fused: Path, vis: Path, ir: Path, n: int):
    names = []
    for p in sorted(fused.iterdir()):
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
            continue
        if (vis / p.name).is_file() and (ir / p.name).is_file():
            names.append(p.name)
        if len(names) >= n:
            break
    return names


def max_abs(ref_row, pred_row, keys):
    worst = 0.0
    worst_key = None
    per = {}
    for key in keys:
        d = abs(float(ref_row[key]) - float(pred_row[key]))
        per[key] = d
        if d >= worst:
            worst = d
            worst_key = key
    return worst, worst_key, per


def main():
    from multiprocessing import set_start_method

    set_start_method("spawn", force=True)
    n_workers = 8
    tasks = []
    refs = {}
    for dataset, cfg in DATASETS.items():
        refs[dataset] = load_ref_csv(cfg["csv"])
        names = list_names(cfg["fused"], cfg["vis"], cfg["ir"], cfg["n"])
        if len(names) < cfg["n"]:
            raise RuntimeError(f"{dataset}: only {len(names)} paired images")
        for name in names:
            tasks.append(
                (
                    dataset,
                    name,
                    str(cfg["ir"] / name),
                    str(cfg["vis"] / name),
                    str(cfg["fused"] / name),
                )
            )
        print(f"[probe] {dataset}: {len(names)} images, first={names[0]}", flush=True)

    expected_seq = sum(DATASETS[d]["n"] * DATASETS[d]["ref_s_per_img"] for d in DATASETS)
    print(f"[probe] workers={n_workers}  n_tasks={len(tasks)}  expected_seq~{expected_seq:.1f}s", flush=True)

    t0 = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=n_workers, initializer=init_worker) as pool:
        for row in pool.map(eval_one, tasks):
            rows.append(row)
    wall = time.perf_counter() - t0

    metric_names = [k for k in rows[0].keys() if k not in {"dataset", "image_name"}]
    summary = {
        "n_workers": n_workers,
        "n_images": len(rows),
        "wall_s": wall,
        "s_per_img": wall / len(rows),
        "expected_seq_s": expected_seq,
        "speedup_vs_recorded_seq": expected_seq / wall,
        "datasets": {},
        "ok": True,
    }

    print(f"\n[probe] wall={wall:.1f}s  {wall/len(rows):.3f}s/img  speedup_vs_recorded={expected_seq/wall:.2f}x", flush=True)
    print(f"{'set':8s} {'n':>4} {'six_max':>12} {'six_at':>8} {'all_max':>12} {'iqa_max':>12}", flush=True)

    by_ds = {}
    for row in rows:
        by_ds.setdefault(row["dataset"], []).append(row)

    for dataset, ds_rows in by_ds.items():
        six_worst = 0.0
        all_worst = 0.0
        iqa_worst = 0.0
        six_at = None
        all_at = None
        fail_six = 0
        for row in ds_rows:
            ref = refs[dataset][row["image_name"]]
            w6, k6, _ = max_abs(ref, row, SIX)
            wa, ka, _ = max_abs(ref, row, metric_names)
            wi, _, _ = max_abs(ref, row, [k for k in IQA if k in metric_names])
            if w6 > six_worst:
                six_worst, six_at = w6, (row["image_name"], k6)
            if wa > all_worst:
                all_worst, all_at = wa, (row["image_name"], ka)
            iqa_worst = max(iqa_worst, wi)
            if w6 > 1e-8:
                fail_six += 1
        ok = fail_six == 0
        summary["ok"] = summary["ok"] and ok
        summary["datasets"][dataset] = {
            "n": len(ds_rows),
            "six_max_abs": six_worst,
            "six_at": list(six_at) if six_at else None,
            "all25_max_abs": all_worst,
            "all25_at": list(all_at) if all_at else None,
            "iqa_max_abs": iqa_worst,
            "n_six_mismatch": fail_six,
            "ok": ok,
        }
        print(
            f"{dataset:8s} {len(ds_rows):4d} {six_worst:12.3e} {str(six_at[1] if six_at else ''):>8} "
            f"{all_worst:12.3e} {iqa_worst:12.3e}  {'OK' if ok else 'FAIL'}",
            flush=True,
        )

    out = METRICS_SAVE / "noleak_eval" / "probe_eval_mp_result.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[probe] saved {out}", flush=True)
    print(f"[probe] overall_ok={summary['ok']}", flush=True)
    if not summary["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
