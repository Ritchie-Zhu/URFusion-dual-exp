本目录是给 AI4R 的文字材料，不含数据集、ckpt、融合大图。不要把本目录当成可训练代码。

| 文件 | 对应负责人条目 |
|---|---|
| DIRECTION.txt | 意向合作方向 |
| LINKS.txt | 开源仓 + 论文 PDF 位置 |
| hardware_env.txt | GPU / 显存 / 时间 / 环境 / 关键失败 |
| repro_six_col.txt | 人工复现主表数字与 json 路径 |
| experiments_inventory.txt | S1 / moe_update / moe_2 / moe_3 等代码、ckpt、log、结果路径 |
| failures.txt | 过程结论（MoE 等失败） |
| AI4R_prompt.md | 人类 idea 引导 prompt（正文可整段粘贴） |

过程 log 仍在原实验目录，打包时再拷贝，例如:
  vis-ir-gray/train-jobs/console_logs/
  vis-ir-gray-moe_update/train-jobs/console_logs/
  vis-ir-gray-moe_2/train-jobs/console_logs/
  vis-ir-gray-moe_3/train-jobs/console_logs/
  ablation_code/S1_wo_Res/train-jobs/console_logs/

论文 PDF 用 /root/autodl-tmp/URFusion-main/URFusion/paper.pdf
