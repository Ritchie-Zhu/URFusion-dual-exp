# AI4R 部署：三件东西如何落到同一棵目录树

工作仓、融合数据包、检测数据包是分开给的。脚本里大量路径写死为  
`/root/autodl-tmp/URFusion-main`。请按下面解压/改名，不要各放各的盘符。

仓库：https://github.com/Ritchie-Zhu/URFusion-dual-exp （分支 `dev`）

```bash
git clone -b dev git@github.com:Ritchie-Zhu/URFusion-dual-exp.git /root/autodl-tmp/URFusion-main
# 或 HTTPS:
# git clone -b dev https://github.com/Ritchie-Zhu/URFusion-dual-exp.git /root/autodl-tmp/URFusion-main
cd /root/autodl-tmp/URFusion-main
```

下面 `SRC_*` 换成你下载的两个 zip 的解压位置。

## 融合数据包 `experimental_dataset.zip`

解压后顶层是 `experimental_dataset/`（`train/`、`test/`、`README.txt`）。  
训练脚本找的是 `datasets/training_noleak`，评测找的是官方测试集路径：

```bash
mkdir -p datasets
# 训练 9987 pairs（目录里是 train/VIS 与 train/IR）
ln -sfn /绝对路径/experimental_dataset datasets/training_noleak

mkdir -p datasets/MSRS datasets/M3FD datasets/LLVIP
ln -sfn /绝对路径/experimental_dataset/test/MSRS  datasets/MSRS/test
ln -sfn /绝对路径/experimental_dataset/test/M3FD  datasets/M3FD/test
ln -sfn /绝对路径/experimental_dataset/test/LLVIP datasets/LLVIP/test_sample1000
```

若你把 zip 直接解到本仓 `datasets/` 下，则：

```bash
cd /root/autodl-tmp/URFusion-main
ln -sfn "$(pwd)/datasets/experimental_dataset" datasets/training_noleak
mkdir -p datasets/MSRS datasets/M3FD datasets/LLVIP
ln -sfn "$(pwd)/datasets/experimental_dataset/test/MSRS"  datasets/MSRS/test
ln -sfn "$(pwd)/datasets/experimental_dataset/test/M3FD"  datasets/M3FD/test
ln -sfn "$(pwd)/datasets/experimental_dataset/test/LLVIP" datasets/LLVIP/test_sample1000
```

训练：`--baseDir .../datasets/training_noleak`  
评测测试集：`datasets/MSRS/test`（361）、`datasets/M3FD/test`（300）、`datasets/LLVIP/test_sample1000`（1000）。  
上色脚本接受 `vis/ir` 或 `VIS/IR`。

## 检测数据包 `detection_dataset.zip`

评测用，不是重训 YOLO。解压后顶层是 `detection_dataset/`。

```bash
cd /root/autodl-tmp/URFusion-main
mkdir -p datasets/yolo/labels datasets/yolo/runs/yolo11s_iv_det/weights
ln -sfn /绝对路径/detection_dataset/test_iv_2000          datasets/yolo/test_iv_2000
ln -sfn /绝对路径/detection_dataset/labels/test           datasets/yolo/labels/test
ln -sfn /绝对路径/detection_dataset/split_manifest.csv    datasets/yolo/split_manifest.csv
ln -sfn /绝对路径/detection_dataset/weights/best.pt       datasets/yolo/runs/yolo11s_iv_det/weights/best.pt
```

不要用 `datasets/yolo/eval_staging/` 当测试集（其中有坏链，评测脚本会现场重建）。

## 冻结 C（训练新 F 必需）

clone 之后应已有（本仓已跟踪终稿，不要用带 `training` 的泄漏权重）：

```
our_model_1_DualMoE/train-jobs/ckpt/Vis_Content_noleak/Vis_Content_noleak_ckpt.pth
our_model_1_DualMoE/train-jobs/ckpt/ir_Content_noleak/ir_Content_noleak_ckpt.pth
```

训练脚本：`--vis_content_ckpt Vis_Content_noleak --ir_content_ckpt ir_Content_noleak`。  
推理不加载 C。A2/S1 的 F 权重不是训练必需，6 列数字见仓内 `AI4R_URFusion/repro_six_col.txt`。

## 官方 25 列评测

必须：`metrics_save/llvip_sample1000/run_eval_mp.py` → `eval_torch.evaluation_one`。  
禁止：`eval_torch_fast`。

clone 后应已有：

```
metrics_py/__pycache__/eval_torch.cpython-310.pyc
metrics_py/__pycache__/Metric_torch.cpython-310.pyc
metrics_py/__pycache__/Qabf.cpython-310.pyc
metrics_py/__pycache__/Nabf.cpython-310.pyc
metrics_py/__pycache__/ssim.cpython-310.pyc
```

若 clone 后缺这些文件，用同目录 `AI4R_Doc/eval_pyc.zip` 解压到 `metrics_py/__pycache__/`。

环境与线程见 `04_ENVIRONMENT.md`。

## 目录名对照（旧材料 → 现仓）

| 旧名 | 现名 |
|---|---|
| `vis-ir-gray/` | `our_model_1_DualMoE/` |
| `vis-ir-gray-moe_update/` | `our_model_2_SpatialResMoE/` |
| `vis-ir-gray-moe_2/` | `our_model_3_SpatialResMoE_2/` |
| `vis-ir-gray-moe_3/` | `our_model_4_ConflictAdd/` |
| `ablation_code/` | `ablation/` |
| A2 `Abl_wo_MoE` | `Abl_DualMoE_wo_MoE` |
| S1 `Abl_SR_wo_Res` | `Abl_SpatialResMoE_wo_Res` |

不要改 `vis-ir/code`。不要覆盖 `Fusion_noleak_1` 行为与官方 `metrics_save/{LLVIP,MSRS,M3FD}.xlsx`。
