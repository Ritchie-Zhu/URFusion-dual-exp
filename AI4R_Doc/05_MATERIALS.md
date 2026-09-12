# 之前那份 AI4R zip 是什么

是的。先前说的「仓里已有的材料 / zip」指 **2026-09-08 打的文字材料包**，不是融合图数据包，也不是检测数据包。

| 东西 | 是什么 | 在哪 |
|---|---|---|
| `AI4R_URFusion/` | 方向、链接、6 列、失败设计、给代理的 prompt | 已在 GitHub `dev` |
| `URFusion-dual-exp_AI4R/` 与 `.zip` | 同一套材料再打的包（含 `02_paper.pdf`、log 摘录） | 本机有；**不要**和 `experimental_dataset.zip`、`detection_dataset.zip` 混成一个包 |

那份材料包大约 13 MB，几乎全是论文 PDF。里面的路径还是旧名（`vis-ir-gray`、`ablation_code`、A1/A2/S1），对照见 `01_DEPLOY.md`。

**这次分开给 AI4R 的三件运行材料：**

1. GitHub 工作仓（代码 + 现已上传的 C ckpt + 评测 pyc + 本目录说明）
2. `experimental_dataset.zip`（融合训练/测试图）
3. `detection_dataset.zip`（YOLO test-2000 + `best.pt`）

数字和禁止重训的说明继续读仓内 `AI4R_URFusion/`（或旧 zip）。不必为了部署再重新打那份 13 MB 材料包。
