# our_model_1_DualMoE — Gray-Y Dual-Branch MoE URFusion

独立 pipeline，与 `vis-ir/` 旧版（RGB + A2V + color loss）完全分离。

## Pipeline 概要

```
I_vis_RGB  →  Y_vis, Cb_vis, Cr_vis
Y_vis, I_ir  →  F_dual_moe_gray  →  Y_F
Y_F + Cb_vis + Cr_vis  →  RGB_fused
```

- **F** 只做灰度亮度融合，输出 `Y_F [B,1,H,W]`
- **颜色** 推理时从可见光 Cb/Cr 确定性迁移，训练不经过 colorize
- **C_gray** 仅训练阶段监督 F，不参与推理
- **无** A2V、centroid、vis.mat、modulation

## 目录结构

仓库里只保留 **代码骨架**；`train-jobs/`、`results/` 等输出目录 **不在仓库中预建**，与 `vis-ir/` 旧脚本一样，在训练/测试运行时通过 `os.makedirs(..., exist_ok=True)` 自动创建。

```
our_model_1_DualMoE/
├── README.md
└── code/                     # 工作目录（在此运行脚本）
    ├── models/
    ├── losses/
    ├── dataset.py            # [待写]
    ├── ycbcr.py              # [待写]
    ├── utils.py              # [待写]
    ├── train_c_vis_gray.py   # [待写]
    ├── train_c_ir_gray.py    # [待写]
    ├── train_fusion_gray.py  # [待写]
    ├── test_fusion_gray.py   # [待写]
    └── evaluate_metrics.py   # [待写]
```

### 运行时自动生成的路径（约定，与 vis-ir 一致）

| 用途 | 默认路径（相对 `code/`） | 创建时机 |
|------|--------------------------|----------|
| C / F checkpoint | `../train-jobs/ckpt/{experiment}_ckpt.pth` 等 | 训练脚本启动 / 保存时 |
| 训练 metrics CSV/JSON | `../train-jobs/metrics_logs/{experiment}/` | 首个 epoch 写日志时 |
| 固定样本可视化 | `../train-jobs/fixed_samples/{experiment}/` | 配置了 test 图且保存时 |
| 推理 Y_F / RGB_fused | `--outputDir` 下子目录 | 测试脚本写盘前 |

路径均可通过 `--experiment`、`--ckptRoot`、`--outputDir` 等参数覆盖；脚本内统一在写盘前 `os.makedirs(..., exist_ok=True)`。

## 数据集路径

不复制数据，训练/测试时指向上级 dataset，例如：

```bash
--baseDir ../vis-ir/dataset/M3FD/
# 或
--baseDir ../dataset/M3FD/
```

## 计划中的 code/models 文件

| 文件 | 说明 |
|------|------|
| `structure_encoder.py` | 1ch in → 64ch out 内容提取器 C |
| `fusion_pre.py` | 1ch 模态 pre-net |
| `fusion_post.py` | 1ch 输出 post-net（无 modulation） |
| `dual_branch_fusion.py` | Gray 双分支 F 整体 |
| `dual_moe.py` | 从 vis-ir 拷贝，MoE 跨模态融合 |
| `fusion_expert.py` | Expert block |
| `spatial_attention.py` | SA 模块 |

## 计划中的 code/losses 文件

| 文件 | 说明 |
|------|------|
| `gray_fusion_loss.py` | L_main, L_int, L_grad, L_branch, L_aux |

## 实施顺序

1. ~~`ycbcr.py` + 1ch `StructureEncoderGray`~~ ✅
2. ~~`train_c_vis_gray.py` / `train_c_ir_gray.py`~~ ✅
3. Gray F 模型 + `train_fusion_gray.py`
4. `test_fusion_gray.py` + smoke test

## 训练 C（M3FD，在 `our_model_1_DualMoE/code/` 下执行）

```bash
# C_vis_gray
python train_c_vis_gray.py --device 0 --experiment content_encoder_vis_gray

# C_ir_gray
python train_c_ir_gray.py --device 0 --experiment content_encoder_ir_gray
```

默认数据：`../../vis-ir/dataset/M3FD/`  
Checkpoint：`../train-jobs/ckpt/{experiment}_ckpt.pth`  
日志：`../train-jobs/metrics_logs/{experiment}/`

## 与 vis-ir 的关系

- **不修改** `vis-ir/` 下任何文件
- MoE 等稳定模块首次从 `vis-ir/code/models/` 拷贝，之后在 gray 侧独立演进
- 不做跨目录 `sys.path` import
