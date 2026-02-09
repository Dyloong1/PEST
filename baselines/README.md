# Baselines

9 baseline models for 3D turbulence temporal prediction, implemented in a unified framework.

## Models

| Category | Model | Reference |
|----------|-------|-----------|
| Spectral | FNO3D | Li et al., 2021 |
| Spectral | UFNO3D | Wen et al., 2022 |
| Spectral | TFNO3D | Kossaifi et al., 2023 |
| Transformer | Transolver3D | Wu et al., ICML 2024 |
| Transformer | FactFormer3D | Li et al., NeurIPS 2023 |
| Transformer | DPOT3D | Hao et al., ICML 2024 |
| Neural Operator | DeepONet3D | Lu et al., 2021 |
| Neural Operator | PI-DeepONet3D | Wang et al., 2021 |
| Physics-Informed | PINO3D | Li et al., 2024 |

## Usage

All models are defined in `_baseline_models.py`.

### Train
```bash
python train_all_baselines.py --models fno,ufno,transolver --datasets DNS,JHU
python train_all_baselines.py --all   # train all models on all datasets
python train_all_baselines.py --all --size large  # train all at large size
```

### Evaluate
```bash
python evaluate_all_baselines.py --checkpoint_dir <DIR>
python evaluate_per_timestep.py --checkpoint_dir <DIR>
```

## Model Size Variants

Each model provides 4 size configs: `small`, `default`, `large`, `xlarge`.

### Spectral Methods (FNO / PINO)
| Size | width | num_layers |
|------|-------|------------|
| `small` | 20 | 3 |
| `default` | 32 | 4 |
| `large` | 48 | 5 |
| `xlarge` | 64 | 6 |

### UFNO
| Size | base_width |
|------|------------|
| `small` | 20 |
| `default` | 32 |
| `large` | 48 |
| `xlarge` | 64 |

### TFNO
| Size | width | num_layers | rank |
|------|-------|------------|------|
| `small` | 48 | 4 | 12 |
| `default` | 96 | 6 | 16 |
| `large` | 128 | 8 | 24 |
| `xlarge` | 192 | 10 | 32 |

### Transolver
| Size | dim | depth | num_slices | num_heads |
|------|-----|-------|------------|-----------|
| `small` | 192 | 4 | 32 | 6 |
| `default` | 384 | 8 | 64 | 12 |
| `large` | 512 | 12 | 96 | 16 |
| `xlarge` | 768 | 16 | 128 | 24 |

### DPOT
| Size | dim | depth | modes | num_heads |
|------|-----|-------|-------|-----------|
| `small` | 192 | 6 | 12 | 6 |
| `default` | 384 | 12 | 16 | 12 |
| `large` | 512 | 16 | 20 | 16 |
| `xlarge` | 768 | 20 | 24 | 24 |

### FactFormer
| Size | dim | depth | num_heads |
|------|-----|-------|-----------|
| `small` | 160 | 6 | 5 |
| `default` | 320 | 10 | 10 |
| `large` | 448 | 14 | 14 |
| `xlarge` | 640 | 18 | 20 |

### DeepONet / PI-DeepONet
| Size | branch_dim | trunk_dim | basis_dim |
|------|------------|-----------|-----------|
| `small` | 16 | 64 | 32 |
| `default` | 32 | 128 | 64 |
| `large` | 64 | 256 | 128 |
| `xlarge` | 96 | 384 | 192 |
