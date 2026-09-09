# Ablations

Structure ablations only. **Does not modify** `our_model_1_DualMoE/code`.
Training reuses dataset, losses, utils, and frozen C encoders from the main tree via `sys.path`.

| Folder | Experiment | Change vs `Fusion_noleak_1` (`our_model_1_DualMoE`) |
|--------|------------|-----------------------------------------------------|
| `Abl_DualMoE_wo_Dual/` | `Abl_DualMoE_wo_Dual` | One shared `FusionPreNet` for Y_vis and IR. MoE + post + losses unchanged. |
| `Abl_DualMoE_wo_MoE/` | `Abl_DualMoE_wo_MoE` | Dual PreNet kept; MoE replaced by one `FusionExpertBlock`. `lambda_aux=0`. |
| `Abl_SpatialResMoE_wo_Res/` | `Abl_SpatialResMoE_wo_Res` | Dual + `E_base(concat(f_cap, \|diff\|))` only; no residual experts. Ablation of `our_model_2_SpatialResMoE`. |

Checkpoints: `our_model_1_DualMoE/train-jobs/ckpt/{Abl_DualMoE_wo_Dual,Abl_DualMoE_wo_MoE}/`  
and `ablation/Abl_SpatialResMoE_wo_Res/train-jobs/ckpt/`.

## Train

```bash
bash ablation/Abl_DualMoE_wo_Dual/run_train.sh
bash ablation/Abl_DualMoE_wo_MoE/run_train.sh
bash ablation/Abl_SpatialResMoE_wo_Res/run_train.sh
```
