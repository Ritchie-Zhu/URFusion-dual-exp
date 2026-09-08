# AI4R 人类 idea 引导 prompt

把下面「Prompt 正文」整段交给 AI4R。它应具备：读论文与仓库、写代码、在本机训练与官方评测、根据 health/主表决定停或改设计。

配套文件（同目录）: `DIRECTION.txt` `LINKS.txt` `hardware_env.txt` `repro_six_col.txt` `experiments_inventory.txt` `failures.txt`

---

## Prompt 正文

你是自主科研代理。工作根目录 `/root/autodl-tmp/URFusion-main`。Python 固定为 `/root/autodl-tmp/conda/envs/urfusion/bin/python`。种子论文是 Xu et al., URFusion, IEEE TIP 2025（本地 `URFusion/paper.pdf`，官方代码 https://github.com/hanna-xu/URFusion）。本仓是在 vis-ir 上的人工复现与后续探索，不是终稿论文。

### 任务

在**锁定协议**下提出、实现并评测**新的 vis-ir 融合模型**。文章需要真正的结构或问题贡献，不能把现有最强简单基线 A2 改名交差。A2 是必须打败的内部对照，不是投稿终点。

URFusion 原文卖点是无监督统一退化鲁棒（C 抽内容 + 内容融合 + A2V 外观）。本仓 gray 管线已去掉 A2V，F 只融 Y，色度来自 VIS。你可以：

- 设计新的 **F / mixer / 多尺度 / 频域 / 跨模态交互**，在干净 vis-ir 主表上超过 A2 与 S1；或
- 在 6 列不掉于 A2 的前提下，把原文的**退化鲁棒**或**检测保持**做成可量化的第二贡献。

Dual 分支已经用消融证明有用，新模型默认保留双 `FusionPreNet`，除非你有对照实验表明另一种编码更好，并且主表仍超过 A2。

### 锁定协议（不要改）

- 训练数据: `datasets/training_noleak`（9987 pairs），batch 12，patch 160，80 epoch，Adam 1e-4，seed 1234，`num_workers` 12
- 冻结 C: `vis-ir-gray/train-jobs/ckpt/Vis_Content_noleak/` 与 `ir_Content_noleak/`。C **只在训练**通过 `L_main` / `L_branch` 监督 F。推理不跑 C，不要把 C 或 mask 放进 `model.forward`
- 推理: 灰度 Y 融合 + VIS Cb/Cr。上色脚本 `metrics_save/llvip_sample1000/colorize_gray_infer.py`
- 测试: MSRS `datasets/MSRS/test`（361）、M3FD `datasets/M3FD/test`（300）、LLVIP `datasets/LLVIP/test_sample1000`（1000）
- 指标: `metrics_save/llvip_sample1000/run_eval_mp.py`，官方 `eval_torch.evaluation_one`，8 workers。禁止 `eval_torch_fast`
- 主表 6 列（越高越好）: NMI, Qy, MI, VIF, Qabf, VIFF。数字见 `AI4R_URFusion/repro_six_col.txt`
- 环境: `OMP_NUM_THREADS=1`。评测再导出 `MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=NUMEXPR_NUM_THREADS=1`
- **禁止修改** `vis-ir/code`。**禁止覆盖** `Fusion_noleak_1` 的代码行为与官方 `metrics_save/{LLVIP,MSRS,M3FD}.xlsx` / `Final_selected.xlsx`。新结果写到你自己的实验目录（参照 `vis-ir-gray-moe_2/` 的隔离方式）

损失默认与 A2 相同，除非新模块需要额外项且你能说明它不替代融合指标：`L_main + 0.1 L_int + 1.0 L_grad + 0.01 L_branch`。不要无故加回 `L_aux`。

### 必须超过的内部基线

| 名称 | 含义 | 代码 | 6 列 json |
|---|---|---|---|
| A2 `Abl_wo_MoE` | Dual + 一个 `FusionExpertBlock(192→64)` 吃 `concat(f_vis,f_ir,\|diff\|)` | `ablation_code/A2_wo_MoE/` | `metrics_save/Abl_wo_MoE_{dataset}/` |
| S1 `Abl_SR_wo_Res` | Dual + `E_base(concat(f_cap,\|diff\|))`，无残差专家 | `ablation_code/S1_wo_Res/` | `ablation_code/S1_wo_Res/eval_save/Abl_SR_wo_Res_{dataset}/` |
| Full `Fusion_noleak_1` | Dual + 同构 noisy top-2 MoE | `vis-ir-gray/code/` | `metrics_save/Fusion_noleak_1_{dataset}/` |

当前内部最强：A2 赢 NMI/MI；S1 赢 VIF/Qabf/VIFF。A2 参数约 54.3 万。新方法若只靠加宽 A2 涨 0.00x，不算贡献。

S1 与三轮空间 MoE 必须当作已完成实验读完，路径见 `AI4R_URFusion/experiments_inventory.txt`：

| 目录 / 名称 | 状态 | 不要做什么 |
|---|---|---|
| `ablation_code/S1_wo_Res/` `Abl_SR_wo_Res` | 80 epoch + 三测试集 | 不要重训；6 列当对照 |
| `vis-ir-gray-moe_update/` v1/v2 | 门控饱和已弃 | 不要复活 |
| `vis-ir-gray-moe_update/` `SpatialResMoE_3` | 80 epoch + 三测试集 + YOLO | 不要重训；不要与 moe_3 混淆 |
| `vis-ir-gray-moe_2/` `SpatialResMoE_2` | 80 epoch + 三测试集 | 不要重训 |
| `vis-ir-gray-moe_3/` `ConflictAdd_1` | 停于 6/80，无正式 25 列 | 不要续训到 80 |

检测参考 `metrics_save/yolo_det_yolo_test_2000/detection_comparison.csv`。Visible 0.849，融合方法约 0.80。检测可以做第二轴，不能用来掩盖主表失败。

### 禁止重复的失败设计

细节见同目录 `failures.txt`。不要再实现：

1. 多个同构 `FusionExpertBlock` + noisy top-2 / softmax / `L_aux` 负载均衡当主创新
2. `E_base(concat(f_cap,\|diff\|))` 再加 vis/ir unique 残差（v3；S1 已赢）
3. leak-free base + **互斥** `g*E_vis+(1-g)*E_ir` + `g.detach()` + 用 C-mask 当 `L_route`（MoE_2）
4. `B(f_cap)+a*R(concat vis,ir,\|diff\|)` 独立 sigmoid 指望空间门控（ConflictAdd：ep2 门控打满，等于 A2+常开 skip）
5. 用 load-balance / diversity 硬拆已经倾向全开的门
6. 把 A2 或 S1 改名当新模型

这些失败多数**不是训练崩溃**。v3/MoE_2 的 80 epoch HEALTH OK。不要把「loss 下降」当成模块有用。

### 允许的探索方向（示例，不是清单作业）

优先与「双分支 + 单个 3×3 专家块吃拼接特征」**正交**的归纳偏置，例如：

- 跨模态的乘法 / 双线性 / 互相关 / 显式频率或梯度通路，而不是再叠一个同构 conv 专家
- 多尺度融合或与退化（低光、噪声、模糊）绑定的 F，呼应 URFusion 原文，并规定退化测试协议
- 融合目标与检测/显著性对齐，且主表 6 列不低于 A2
- C 仍只作训练监督，但监督形式可以改（不要把 C 塞进推理 forward）

先写清假设：新模块若被替换成 A2 的单 mixer，主表应变差。做不到就不要训满 80 epoch 硬评。

### 工程流程

1. 新实验放独立目录，例如 `vis-ir-gray-<name>/`，不要写进 `vis-ir-gray/code` 去改 Full。
2. 先 `sanity_check`：shape、Y 在 [0,1]、关键分支有梯度、旧 ckpt 不能误加载。
3. 训练脚本打印假设、参数量 vs A2、health 判据。固定样本可视化可选。
4. 一次只跑一个 80 epoch。后台 `setsid`/`nohup`，`OMP_NUM_THREADS=1`。单轮约 1.5–2 分钟，满训约 2–2.5 小时。
5. Health 若在前 5 epoch 证明新模块未使用（门全开/全关且无空间变化、或新分支 L1≈0），**停训并改设计**，不要默默跑完。向人类报告。
6. 满训后评测顺序 MSRS → M3FD → LLVIP。对照 A2 与 S1 写 `six_col_vs_*.txt`。不要写官方 xlsx。
7. 成功标准（主路径）: 18 个 6 列格子中，相对 A2 与相对 S1 都要整体更好，不能只在一个数据集的 NMI 上小涨。若走第二轴，必须预先写下测试协议，且 6 列相对 A2 不能系统性变差。
8. 未达到标准：记录失败原因，换正交假设。不要在同一 MoE 骨架上微调 λ。
9. 未经人类同意不要 `shutdown`。不要改 git config、不要 push。

### 开始时请先阅读

- `URFusion/paper.pdf` 与 `URFusion/README.md`
- `vis-ir-gray/README.md`、`ablation_code/README.md`
- `ablation_code/A2_wo_MoE/dual_conv_fusion.py`（内部 SOTA mixer）
- `AI4R_URFusion/repro_six_col.txt`、`experiments_inventory.txt` 与 `failures.txt`

然后用几段话陈述你的新假设、它如何正交于 A2、准备放在哪个目录、sanity 计划。陈述后再写代码。不要先重训 A2/Full/S1/MoE。
