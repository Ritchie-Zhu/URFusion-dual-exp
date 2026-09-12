# 运行环境：Python、依赖与线程

与本机人工复现时一致。Python 固定为 conda 环境里的解释器，不要用系统 python。

## 本机记录（AutoDL，2026-09）

- OS: Ubuntu 22.04，CUDA 11.8
- GPU: RTX 4090 D 24 GB；Fusion 训练约占用 8 GB
- Python: **3.10.20**
- 路径：`/root/autodl-tmp/conda/envs/urfusion/bin/python`

| 包 | 版本 |
|---|---|
| torch | 2.7.1+cu118 |
| torchvision | 0.22.1+cu118 |
| numpy | 1.24.3 |
| opencv-python | 4.13.0 |
| Pillow | 10.4.0 |
| ultralytics | 8.4.90（仅检测评测） |
| openpyxl | 3.1.5 |
| natsort | 8.4.0 |
| tqdm | 4.67.3 |
| PyYAML | 6.0.3 |

## 线程（必须）

```bash
export OMP_NUM_THREADS=1
# 评测时再加：
export MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

`OMP` 开多了，官方 25 列会卡死或数值不稳。

## 锁定训练协议

- 数据：`datasets/training_noleak`（9987 pairs，见 `部署说明_仓库与数据集如何接路径.md`）
- batch 12，patch 160，80 epoch，Adam 1e-4，seed 1234，`num_workers` 12
- 冻结 C：`Vis_Content_noleak` / `ir_Content_noleak`
- 推理：Y 融合 + VIS Cb/Cr；不跑 C
- 融合评测：官方 25 项，`run_eval_mp.py`，8 workers，禁止 `eval_torch_fast`（见 `改融合网络指令_融合与检测成功标准.md` / `对照指标_融合25项与目标检测指标.md`）
- 检测评测：同一 `best.pt`，`test_iv_2000`

单轮 Fusion 约 1.5–2 分钟，80 epoch 约 2–2.5 小时。三测试集官方评测约 34 分钟。

## 建议安装

```bash
conda create -n urfusion python=3.10 -y
conda activate urfusion
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
pip install numpy==1.24.3 opencv-python==4.13.0 Pillow==10.4.0
pip install ultralytics==8.4.90 openpyxl==3.1.5 natsort==8.4.0 tqdm==4.67.3 PyYAML==6.0.3
```

无 tmux 时用 `setsid` / `nohup`。未经同意不要 `shutdown`。
