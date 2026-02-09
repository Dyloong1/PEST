# PEST - Full Ablation Implementation

This directory contains the **most complete ablation** of PEST, including two components:

1. **Resolution Reconstruction** (`train_swin_spectral.py`) - Swin Transformer with spectral-weighted loss for sparse-to-dense 3D field reconstruction.
2. **PDE Refiner** (`train_swin_dit_pde_refiner.py`) - Swin-DiT with iterative diffusion-based refinement (K=3 steps).

For details on the plug-in refinement design, see **Appendix H: Plug-in Refinement Methods** in the paper.

> To obtain the base PEST model, remove the sparse-observation residual connection from the resolution reconstruction module.

## Structure

```
PEST/
├── Model/
│   ├── configs/
│   │   ├── swin_patched_config_fixed.py   # Base Swin config
│   │   └── swin_jhu_config.py             # JHU DNS128 configs (spectral, etc.)
│   ├── losses/                            # Spectral, physics, adaptive losses
│   ├── modules/swin_blocks.py             # Swin Transformer blocks
│   ├── swin_transformer_model_patched.py  # Swin backbone (resolution reconstruction)
│   ├── swin_dit_pde_refiner.py            # Swin-DiT + PDE-Refiner
│   └── swin_dit_model.py                  # Base Swin-DiT architecture
├── Dataset/                               # JHU dataset loaders
├── Data/                                  # Normalization statistics
├── train_swin_spectral.py                 # Train resolution reconstruction
├── train_swin_dit_pde_refiner.py          # Train PDE Refiner
├── evaluate_swin_jhu_ar.py                # Autoregressive evaluation
├── test_swin_dit_pde_refiner_ar.py        # PDE Refiner AR testing
└── test_autoregressive.py                 # General AR testing
```

## Usage

### Resolution Reconstruction (Swin Spectral)
```bash
python train_swin_spectral.py --config spectral --data_dir <JHU_DATA_PATH>
```

### PDE Refiner
```bash
python train_swin_dit_pde_refiner.py --config default --data_dir <JHU_DATA_PATH>
```

### Autoregressive Evaluation
```bash
python evaluate_swin_jhu_ar.py --checkpoint <CKPT_PATH> --rounds 3
python test_swin_dit_pde_refiner_ar.py --checkpoint <CKPT_PATH> --rounds 3
```
