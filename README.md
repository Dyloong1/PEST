# PEST: Physics-Enhanced Swin Transformer for 3D Turbulence Simulation

This repository is the official implementation for [PEST](https://arxiv.org/pdf/2601.14517).

<img src="architecture.png" alt="architecture" width="100%">

## Data

Download the preprocessed datasets (JHU DNS128 and TGV) from:

https://drive.google.com/drive/folders/1M6GJl8dGodToLKJYaNvJzglKoWMdhrkX?usp=drive_link

## Directory Structure

```
code_public/
├── PEST/           # PEST full ablation (available plug-ins: resolution reconstruction + PDE Refiner)
├── baselines/      # 9 baseline models (FNO, UFNO, TFNO, Transolver, FactFormer, DPOT, DeepONet, PI-DeepONet, PINO)
└── README.md
```

## Quick Guide

- **`PEST/`** contains the complete PEST ablation with two modules: resolution reconstruction (Swin Transformer + spectral loss) and plug-in PDE Refiner (Swin-DiT, K=3 diffusion steps). See `PEST/README.md` for training and evaluation commands. For the plug-in refinement design, refer to **Appendix H** in the paper. Remove the sparse residual connection to obtain the base PEST.

- **`baselines/`** contains 9 unified baseline implementations. All models share the same data pipeline and evaluation protocol. See `baselines/README.md` for the full list and usage.

## Model Size Variants

Each Swin model provides multiple size configs for ablation or resource-constrained environments:

### Resolution Reconstruction (Swin Spectral)
| Config Name | EMBED_DIM | DEPTHS |
|-------------|-----------|--------|
| `spectral_tiny` | 192 | [2,2,2] |
| `spectral_small` | 272 | [2,4,2] |
| `spectral_mid` | 384 | [2,6,2] |
| `spectral_large` | 512 | [2,8,4] |

```bash
python train_swin_spectral.py --config spectral_small --data_dir <DATA_PATH>
```

### PDE Refiner (Swin-DiT)
| Config Name | EMBED_DIM | DEPTHS |
|-------------|-----------|--------|
| `small` | 160 | [2,2,4,2] |
| `mid` | 224 | [2,2,8,2] |
| `large` | 320 | [2,4,12,2] |
| `xlarge` | 448 | [2,6,18,2] |

```bash
python train_swin_dit_pde_refiner.py --config small --data_dir <DATA_PATH>
```
