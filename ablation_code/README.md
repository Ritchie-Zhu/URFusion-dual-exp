# Ablation code (A1 / A2)

Structure ablations only. **Does not modify** `vis-ir-gray/code`.
Training reuses dataset, losses, utils, and frozen C encoders from the main tree via `sys.path`.

| Folder | Experiment | Change vs Full (`Fusion_noleak_1`) |
|--------|------------|-----------------------------------|
| `A1_wo_Dual/` | `Abl_wo_Dual` | One shared `FusionPreNet` for Y_vis and IR (no separate vis/ir encoders). MoE + post + losses unchanged. |
| `A2_wo_MoE/` | `Abl_wo_MoE` | Dual PreNet kept; MoE router + 4 experts replaced by **one** `FusionExpertBlock`. `lambda_aux=0`. |

Checkpoints: `vis-ir-gray/train-jobs/ckpt/{Abl_wo_Dual,Abl_wo_MoE}/`

## Train

```bash
# A1 (~2.5h, 1 GPU)
bash ablation_code/A1_wo_Dual/run_train.sh

# A2 (~2.5h, 1 GPU) — use another GPU or run after A1
bash ablation_code/A2_wo_MoE/run_train.sh
```

Add `--shutdown_on_finish` to the python args via `"$@"` if needed, e.g. edit run script or:

```bash
bash ablation_code/A1_wo_Dual/run_train.sh --shutdown_on_finish
```

## Infer (after training)

Use each folder’s `fusion_gray_infer.py` with `--experiment Abl_wo_Dual` or `Abl_wo_MoE`.
Colorize still uses `metrics_save/llvip_sample1000/colorize_gray_infer.py` with matching `--method`.

## Eval

Reuse `metrics_save/llvip_sample1000/run_eval_mp.py` and the B1–B3 eval pipeline; point infer at the ablation `fusion_gray_infer.py` paths.
