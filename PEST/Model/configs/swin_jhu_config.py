"""
JHU DNS128 专用 Swin Transformer 配置

针对JHU立方体数据 (128, 128, 128) 优化:
- 均匀的 patch size (4, 4, 4)
- 更多的稀疏层 (16层, sparse_z_interval=8)
- 与现有DNS配置兼容的模型结构
"""

from .swin_patched_config_fixed import SwinPatchedConfigFixed


class SwinJHUConfig(SwinPatchedConfigFixed):
    """
    JHU DNS128 Swin Transformer 配置

    继承自 SwinPatchedConfigFixed，调整以下参数:
    - INPUT_RESOLUTION: (64, 128, 128) → (128, 128, 128)
    - PATCH_SIZE: (2, 4, 4) → (4, 4, 4)
    - NUM_SPARSE_Z_LAYERS: 12 → 16 (sparse_z_interval=8)
    - LOSS_CONFIG: 使用正确的JHU物理参数
    """

    CHECKPOINT_DIR = 'checkpoints_swin_jhu'
    EXPERIMENT_NAME = 'swin_jhu_baseline'

    IN_CHANNELS = 4
    INPUT_RESOLUTION = (128, 128, 128)

    NUM_SPARSE_Z_LAYERS = 16

    PATCH_SIZE = (4, 4, 4)
    PATCHED_RESOLUTION = (32, 32, 32)

    WINDOW_SIZE = (8, 8, 8)

    _DX = 0.0490873852
    _NU = 0.000185
    _DT = 0.01

    _NU_GRID = _NU / (_DX ** 2)

    LOSS_CONFIG = {
        'data': {
            'type': 'mse',
            'weight': 1.0,
            'target': 'missing'
        },
        'divergence': {
            'type': 'divergence_3d',
            'weight': 1.0,
            'dx': 1.0,
            'dy': 1.0,
            'dz': 1.0
        },
        'navier_stokes': {
            'type': 'navier_stokes_forced_turb',
            'weight': 0.1,
            'nu': _NU_GRID,
            'rho': 1.0,
            'dx': 1.0,
            'dy': 1.0,
            'dz': 1.0,
            'dt': _DT,
            'forcing_k_cutoff': 2.0
        }
    }

    ADAPTIVE_WEIGHT_CONFIG = {
        'enabled': True,
        'update_frequency': 50,
        'warmup_steps': 200,
        'min_weight': 0.01,
        'max_weight': 10.0,
        'target_ratio': 0.5,
        'adjustment_rate': 0.1,
        'patience': 2
    }


    def print_config(self):
        """Print full configuration with JHU-specific info."""
        print("=" * 80)
        print("JHU DNS128 Swin Transformer Configuration")
        print("=" * 80)

        print(f"\n[Data - JHU DNS128]")
        print(f"  Input channels: {self.IN_CHANNELS}")
        print(f"  Resolution: {self.INPUT_RESOLUTION} (立方体)")
        print(f"  Sparse z-layers (total count): {self.NUM_SPARSE_Z_LAYERS}")

        print(f"\n[Patch - 均匀立方体]")
        print(f"  Patch size: {self.PATCH_SIZE}")
        print(f"  Patched resolution: {self.PATCHED_RESOLUTION}")
        print(f"  Memory reduction: {self.memory_reduction_factor:.1f}x")

        print(f"\n[Window]")
        print(f"  Window size: {self.WINDOW_SIZE}")

        print(f"\n[Encoder]")
        print(f"  Embed dim: {self.EMBED_DIM}")
        print(f"  Depths: {self.DEPTHS}")
        print(f"  Num heads: {self.NUM_HEADS}")
        print(f"  Attention: {self.ATTENTION_TYPE}")

        print(f"\n[Decoder]")
        print(f"  Dim: {self.DECODER_DIM}")
        print(f"  Heads: {self.DECODER_NUM_HEADS}")

        print(f"\n[Loss]")
        for name, config in self.LOSS_CONFIG.items():
            print(f"  {name}: {config['type']} (weight={config['weight']})")

        print(f"\n[Regularization]")
        print(f"  Drop: {self.DROP_RATE}")
        print(f"  Attention drop: {self.ATTN_DROP_RATE}")
        print(f"  Drop path: {self.DROP_PATH_RATE}")
        print(f"  Gradient checkpointing: {self.USE_GRADIENT_CHECKPOINTING}")

        print(f"\n[Sparse Completion]")
        print(f"  Method: {self.COMPLETION_METHOD}")

        print("=" * 80)


class SwinJHUConfigPhysical(SwinJHUConfig):
    """
    JHU DNS128 物理单位版本配置

    使用真实物理单位 (dx=0.049, nu=0.000185)
    Loss数量级较大，需要很小的权重
    """

    CHECKPOINT_DIR = 'checkpoints_swin_jhu_physical'
    EXPERIMENT_NAME = 'swin_jhu_physical'

    _DX = 0.0490873852
    _NU = 0.000185
    _DT = 0.01

    LOSS_CONFIG = {
        'data': {
            'type': 'mse',
            'weight': 1.0,
            'target': 'missing'
        },
        'divergence': {
            'type': 'divergence_3d',
            'weight': 0.005,
            'dx': _DX,
            'dy': _DX,
            'dz': _DX
        },
        'navier_stokes': {
            'type': 'navier_stokes_forced_turb',
            'weight': 0.003,
            'nu': _NU,
            'rho': 1.0,
            'dx': _DX,
            'dy': _DX,
            'dz': _DX,
            'dt': _DT,
            'forcing_k_cutoff': 2.0
        }
    }

    ADAPTIVE_WEIGHT_CONFIG = {
        'enabled': True,
        'update_frequency': 50,
        'warmup_steps': 200,
        'min_weight': 0.0001,
        'max_weight': 0.1,
        'target_ratio': 0.5,
        'adjustment_rate': 0.1,
        'patience': 2
    }


class SwinJHUConfigLite(SwinJHUConfig):
    """
    JHU DNS128 Lite版本配置

    简化版: MSE + Divergence (不含 NS)
    使用Grid单位
    """

    CHECKPOINT_DIR = 'checkpoints_swin_jhu_lite'
    EXPERIMENT_NAME = 'swin_jhu_lite'

    EMBED_DIM = 192
    DEPTHS = [2, 4, 2]
    NUM_HEADS = [3, 6, 12]
    DECODER_DIM = 192
    DECODER_NUM_HEADS = 12
    DECODER_DEPTH = 6
    TEMPORAL_NUM_HEADS = 12

    DROP_PATH_RATE = 0.05

    LOSS_CONFIG = {
        'data': {
            'type': 'mse',
            'weight': 1.0,
            'target': 'missing'
        },
        'divergence': {
            'type': 'divergence_3d',
            'weight': 1.0,
            'dx': 1.0,
            'dy': 1.0,
            'dz': 1.0
        }
    }

    ADAPTIVE_WEIGHT_CONFIG = {
        'enabled': True,
        'update_frequency': 50,
        'warmup_steps': 200,
        'min_weight': 0.01,
        'max_weight': 10.0,
        'target_ratio': 0.5,
        'adjustment_rate': 0.1,
        'patience': 2
    }


class SwinJHUConfigFull(SwinJHUConfig):
    """
    JHU DNS128 完整物理约束版本配置

    包含完整的物理损失:
    - Data loss (MSE)
    - Divergence loss (不可压缩约束)
    - Navier-Stokes loss (动量方程约束)
    - Gradient loss (梯度一致性约束)

    使用正确的 JHU 物理参数:
    - 域: 2π × 2π × 2π
    - 网格: 128³
    - dx = 2π/128 ≈ 0.0491
    - nu ≈ 0.000185 (Rλ ~ 433)
    - dt = 0.01 (从文件名 5.020, 5.030, ... 推断)
    """

    CHECKPOINT_DIR = 'checkpoints_swin_jhu_full'
    EXPERIMENT_NAME = 'swin_jhu_full_physics'

    EMBED_DIM = 192
    DEPTHS = [2, 4, 2]
    NUM_HEADS = [3, 6, 12]
    DECODER_DIM = 192
    DECODER_NUM_HEADS = 12
    DECODER_DEPTH = 6
    TEMPORAL_NUM_HEADS = 12

    DROP_PATH_RATE = 0.05

    _DX = 0.0490873852
    _NU = 0.000185
    _DT = 0.01

    LOSS_CONFIG = {
        'data': {
            'type': 'mse',
            'weight': 1.0,
            'target': 'missing'
        },
        'divergence': {
            'type': 'divergence_3d',
            'weight': 1.0,
            'dx': _DX,
            'dy': _DX,
            'dz': _DX
        },
        'navier_stokes': {
            'type': 'navier_stokes_forced_turb',
            'weight': 1.0,
            'nu': _NU,
            'rho': 1.0,
            'dx': _DX,
            'dy': _DX,
            'dz': _DX,
            'dt': _DT,
            'forcing_k_cutoff': 2.0
        },
        'gradient': {
            'type': 'gradient_3d',
            'weight': 1.0,
            'dx': _DX,
            'dy': _DX,
            'dz': _DX
        }
    }

    ADAPTIVE_WEIGHT_CONFIG = {
        'enabled': True,
        'update_frequency': 50,
        'warmup_steps': 200,
        'min_weight': 0.01,
        'max_weight': 100.0,
        'target_ratio': 1.0,
        'adjustment_rate': 0.1,
        'patience': 3
    }


class SwinJHUConfigLarge(SwinJHUConfig):
    """
    JHU DNS128 Large版本配置

    用于追求更好性能
    """

    CHECKPOINT_DIR = 'checkpoints_swin_jhu_large'
    EXPERIMENT_NAME = 'swin_jhu_large'

    EMBED_DIM = 512
    DEPTHS = [2, 6, 4]
    NUM_HEADS = [8, 16, 32]
    DECODER_DIM = 512
    DECODER_NUM_HEADS = 32
    DECODER_DEPTH = 10
    TEMPORAL_NUM_HEADS = 32

    DROP_PATH_RATE = 0.15


class SwinJHUConfigSpectral(SwinJHUConfig):
    """
    JHU DNS128 频谱加权Loss版本配置

    使用频谱加权损失函数，对高频分量给予更大权重，
    以改善高频预测精度。

    基于Parseval定理：空间域MSE等价于频率域MSE，
    通过在频率域加权可以让优化器更关注高频分量。
    """

    CHECKPOINT_DIR = 'checkpoints_swin_jhu_spectral'
    EXPERIMENT_NAME = 'swin_jhu_spectral'


    _DX = 0.0490873852
    _NU = 0.000185
    _NU_GRID = _NU / (_DX ** 2)

    LOSS_CONFIG = {
        'data': {
            'type': 'spectral_weighted',
            'weight': 1.0,
            'high_freq_weight': 3.0,
            'weighting_mode': 'band',
            'adaptive': True,
            'adaptive_config': {
                'mode': 'equalize',
                'ema_momentum': 0.95,
                'adaptation_rate': 0.1,
                'warmup_steps': 200,
                'min_band_weight': 0.2,
                'max_band_weight': 5.0,
                'curriculum_total_steps': 5000
            },
            'band_weights': [1.0, 2.0, 3.0],
            'target': 'full'
        },
        'divergence': {
            'type': 'divergence_3d',
            'weight': 1.0,
            'dx': 1.0,
            'dy': 1.0,
            'dz': 1.0
        },
        'navier_stokes': {
            'type': 'navier_stokes_forced_turb',
            'weight': 0.01,
            'nu': _NU_GRID,
            'rho': 1.0,
            'dx': 1.0,
            'dy': 1.0,
            'dz': 1.0,
            'dt': 0.01,
            'forcing_k_cutoff': 2.0
        },
        'gradient': {
            'type': 'gradient_3d',
            'weight': 0.1
        }
    }

    ADAPTIVE_WEIGHT_CONFIG = {
        'enabled': True,
        'update_frequency': 50,
        'warmup_steps': 200,
        'min_weight': 0.001,
        'max_weight': 10.0,
        'target_ratio': 0.5,
        'adjustment_rate': 0.3,
        'patience': 2,
        'gradient_target_ratio': 0.2,
        'gradient_min_weight': 0.05,
        'gradient_max_weight': 5.0
    }


class SwinJHUConfigSpectralHigh(SwinJHUConfigSpectral):
    """
    JHU DNS128 强高频权重版本

    更激进地优化高频分量 (high_freq_weight=5.0)
    """

    CHECKPOINT_DIR = 'checkpoints_swin_jhu_spectral_high'
    EXPERIMENT_NAME = 'swin_jhu_spectral_high'

    _NU_GRID = SwinJHUConfigSpectral._NU_GRID

    LOSS_CONFIG = {
        'data': {
            'type': 'spectral_weighted',
            'weight': 1.0,
            'high_freq_weight': 5.0,
            'weighting_mode': 'linear',
            'adaptive': False,
            'target': 'full'
        },
        'divergence': {
            'type': 'divergence_3d',
            'weight': 1.0,
            'dx': 1.0,
            'dy': 1.0,
            'dz': 1.0
        },
        'navier_stokes': {
            'type': 'navier_stokes_forced_turb',
            'weight': 0.1,
            'nu': _NU_GRID,
            'rho': 1.0,
            'dx': 1.0,
            'dy': 1.0,
            'dz': 1.0,
            'dt': 0.01,
            'forcing_k_cutoff': 2.0
        }
    }


class SwinJHUConfigSpectralBand(SwinJHUConfigSpectral):
    """
    JHU DNS128 分频带加权版本

    使用分频带权重: low=1.0, mid=2.0, high=4.0
    更精细地控制各频带的优化力度
    """

    CHECKPOINT_DIR = 'checkpoints_swin_jhu_spectral_band'
    EXPERIMENT_NAME = 'swin_jhu_spectral_band'

    _NU_GRID = SwinJHUConfigSpectral._NU_GRID

    LOSS_CONFIG = {
        'data': {
            'type': 'spectral_weighted',
            'weight': 1.0,
            'high_freq_weight': 4.0,
            'weighting_mode': 'band',
            'band_weights': [1.0, 2.0, 4.0],
            'adaptive': False,
            'target': 'full'
        },
        'divergence': {
            'type': 'divergence_3d',
            'weight': 1.0,
            'dx': 1.0,
            'dy': 1.0,
            'dz': 1.0
        },
        'navier_stokes': {
            'type': 'navier_stokes_forced_turb',
            'weight': 0.1,
            'nu': _NU_GRID,
            'rho': 1.0,
            'dx': 1.0,
            'dy': 1.0,
            'dz': 1.0,
            'dt': 0.01,
            'forcing_k_cutoff': 2.0
        }
    }


class SwinJHUConfigDualResidual(SwinJHUConfigSpectral):
    """
    JHU DNS128 双残差学习版本

    在现有Swin框架中整合双残差学习：
    1. Temporal prior: 用input_last作为时间基准
    2. Spatial hint: z_interp(sparse) - temporal_base 作为变化方向提示
    3. Delta learning: 模型输出delta，最终output = temporal_base + delta

    这种设计复用现有的patch-first架构，不增加额外内存开销。

    优势:
    - 利用时间连续性：大部分区域变化不大
    - 利用稀疏观测：提供未来时刻的真实信息作为指导
    - 模型只需学习变化量delta，而非绝对值
    """

    CHECKPOINT_DIR = 'checkpoints_swin_jhu_dual_residual'
    EXPERIMENT_NAME = 'swin_jhu_dual_residual'

    DUAL_RESIDUAL = True

    DELTA_SCALE_INIT = 0.1

    _NU_GRID = SwinJHUConfigSpectral._NU_GRID

    LOSS_CONFIG = {
        'data': {
            'type': 'spectral_weighted',
            'weight': 1.0,
            'high_freq_weight': 3.0,
            'weighting_mode': 'linear',
            'adaptive': False,
            'target': 'full'
        },
        'divergence': {
            'type': 'divergence_3d',
            'weight': 1.0,
            'dx': 1.0,
            'dy': 1.0,
            'dz': 1.0
        },
        'navier_stokes': {
            'type': 'navier_stokes_forced_turb',
            'weight': 0.1,
            'nu': _NU_GRID,
            'rho': 1.0,
            'dx': 1.0,
            'dy': 1.0,
            'dz': 1.0,
            'dt': 0.01,
            'forcing_k_cutoff': 2.0
        }
    }


class SwinJHUConfigNS5x(SwinJHUConfigSpectral):
    """
    JHU DNS128 高NS权重版本配置

    与baseline (spectral) 相同，但将Navier-Stokes权重增加5倍:
    - NS weight: 0.1 → 0.5

    假设: 更强的物理约束可能改善长期预测稳定性
    """

    CHECKPOINT_DIR = 'checkpoints_swin_jhu_ns5x'
    EXPERIMENT_NAME = 'swin_jhu_ns5x'

    _NU_GRID = SwinJHUConfigSpectral._NU_GRID

    LOSS_CONFIG = {
        'data': {
            'type': 'spectral_weighted',
            'weight': 1.0,
            'high_freq_weight': 3.0,
            'weighting_mode': 'band',
            'adaptive': True,
            'adaptive_config': {
                'mode': 'equalize',
                'ema_momentum': 0.95,
                'adaptation_rate': 0.1,
                'warmup_steps': 200,
                'min_band_weight': 0.2,
                'max_band_weight': 5.0,
                'curriculum_total_steps': 5000
            },
            'band_weights': [1.0, 2.0, 3.0],
            'target': 'full'
        },
        'divergence': {
            'type': 'divergence_3d',
            'weight': 1.0,
            'dx': 1.0,
            'dy': 1.0,
            'dz': 1.0
        },
        'navier_stokes': {
            'type': 'navier_stokes_forced_turb',
            'weight': 0.5,
            'nu': _NU_GRID,
            'rho': 1.0,
            'dx': 1.0,
            'dy': 1.0,
            'dz': 1.0,
            'dt': 0.01,
            'forcing_k_cutoff': 2.0
        },
        'gradient': {
            'type': 'gradient_3d',
            'weight': 0.1
        }
    }


class SwinJHUConfigCurriculum(SwinJHUConfigSpectral):
    """
    JHU DNS128 课程学习版本配置

    使用课程学习策略逐渐增加高频权重:
    - 前20% epochs: 低频为主 (HFW=1.0)
    - 中间50% epochs: 线性增长 (HFW: 1.0→3.0)
    - 后30% epochs: 强调高频 (HFW=3.0)

    物理直觉:
    - 能量级联: 大尺度→小尺度
    - 先学好"骨架"(低频), 再填充"细节"(高频)
    - 训练更稳定，避免早期高频噪声干扰
    """

    CHECKPOINT_DIR = 'checkpoints_swin_jhu_curriculum'
    EXPERIMENT_NAME = 'swin_jhu_curriculum'

    _NU_GRID = SwinJHUConfigSpectral._NU_GRID

    LOSS_CONFIG = {
        'data': {
            'type': 'spectral_weighted',
            'weight': 1.0,
            'high_freq_weight': 3.0,
            'weighting_mode': 'linear',
            'adaptive': False,
            'target': 'full',
            'curriculum': True,
            'curriculum_start_weight': 1.0,
            'curriculum_end_weight': 3.0,
            'total_epochs': 50,
            'warmup_ratio': 0.2,
            'rampup_ratio': 0.5
        },
        'divergence': {
            'type': 'divergence_3d',
            'weight': 1.0,
            'dx': 1.0,
            'dy': 1.0,
            'dz': 1.0
        },
        'navier_stokes': {
            'type': 'navier_stokes_forced_turb',
            'weight': 0.1,
            'nu': _NU_GRID,
            'rho': 1.0,
            'dx': 1.0,
            'dy': 1.0,
            'dz': 1.0,
            'dt': 0.01,
            'forcing_k_cutoff': 2.0
        }
    }


class SwinJHUConfigSpectralProgressive(SwinJHUConfigSpectral):
    """
    JHU DNS128 Spectral + Progressive Window 配置

    基于最佳spectral配置，添加渐进窗口 (Progressive Window):
    - Stage 0: (8,8,8) - 大窗口捕获全局上下文
    - Stage 1: (8,8,8) - 中间stage保持较大感受野
    - Stage 2: (4,4,4) - 小窗口细化局部细节

    这种设计类似UNet的渐进模式，但没有下采样，
    可以测试单独的Progressive Window效果。
    """

    CHECKPOINT_DIR = 'checkpoints_swin_jhu_spectral_progressive'
    EXPERIMENT_NAME = 'swin_jhu_spectral_progressive'

    WINDOW_SIZES = [(8, 8, 8), (8, 8, 8), (4, 4, 4)]

    DEPTHS = [2, 6, 2]


class SwinJHUConfigSpectralProgressiveLarge(SwinJHUConfigSpectral):
    """
    JHU DNS128 大规模 Spectral + Progressive Window 配置

    目标: ~200M 参数量，占用 ~22-24GB VRAM
    架构:
    - EMBED_DIM: 576 (与工作的18GB模型相同)
    - DEPTHS: [2, 24, 4] = 30 total layers
    - Progressive Window: (8,8,8) → (8,8,8) → (4,4,4)

    用于测试 Progressive Window 在大模型上的效果。
    """

    CHECKPOINT_DIR = 'checkpoints_swin_jhu_spectral_progressive_large'
    EXPERIMENT_NAME = 'swin_jhu_spectral_progressive_large'

    EMBED_DIM = 396
    DEPTHS = [2, 12, 4]
    NUM_HEADS = [6, 12, 22]

    DECODER_DIM = 264
    DECODER_NUM_HEADS = 12
    DECODER_DEPTH = 4

    TEMPORAL_NUM_HEADS = 12

    WINDOW_SIZES = [(8, 8, 8), (8, 8, 8), (4, 4, 4)]

    DROP_PATH_RATE = 0.1

    USE_GRADIENT_CHECKPOINTING = True


class SwinJHUConfigSpectralProgressiveGiant(SwinJHUConfigSpectral):
    """
    JHU DNS128 较大规模 Spectral + Progressive Window 配置

    目标: ~110-120M 参数量，在32GB GPU上可运行 (实测148M OOM)
    用于公平对比 PDE-Refiner 迭代精炼 vs 单次预测
    """

    CHECKPOINT_DIR = 'checkpoints_swin_jhu_spectral_progressive_giant'
    EXPERIMENT_NAME = 'swin_jhu_spectral_progressive_giant'

    EMBED_DIM = 432
    DEPTHS = [2, 12, 4]
    NUM_HEADS = [8, 12, 24]

    DECODER_DIM = 288
    DECODER_NUM_HEADS = 16
    DECODER_DEPTH = 4

    TEMPORAL_NUM_HEADS = 12

    WINDOW_SIZES = [(8, 8, 8), (8, 8, 8), (4, 4, 4)]

    DROP_PATH_RATE = 0.1

    USE_GRADIENT_CHECKPOINTING = True


class SwinJHUConfigPDERefinerLarge(SwinJHUConfigSpectral):
    """
    JHU DNS128 大规模 PDE-Refiner 风格配置

    目标: ~150M 参数量，匹配 swin_jhu_spectral (144.7M)
    架构改进:
    - 更大的 EMBED_DIM: 384 → 512
    - 更深的 DEPTHS: [2, 6, 2] → [2, 12, 4]
    - 更多的 attention heads

    用于测试规模增大是否能进一步提升性能。
    """

    CHECKPOINT_DIR = 'checkpoints_swin_jhu_pde_refiner_large'
    EXPERIMENT_NAME = 'swin_jhu_pde_refiner_large'

    EMBED_DIM = 512
    DEPTHS = [2, 12, 4]
    NUM_HEADS = [8, 16, 32]

    DECODER_DIM = 512
    DECODER_NUM_HEADS = 32
    DECODER_DEPTH = 10

    TEMPORAL_NUM_HEADS = 32

    WINDOW_SIZES = [(8, 8, 8), (8, 8, 8), (4, 4, 4)]

    DROP_PATH_RATE = 0.15

    USE_GRADIENT_CHECKPOINTING = True


class SwinJHUConfigPDERefinerXL(SwinJHUConfigSpectral):
    """
    JHU DNS128 超大规模 PDE-Refiner 配置

    目标: ~200M+ 参数量
    用于测试更大模型的性能上限。
    """

    CHECKPOINT_DIR = 'checkpoints_swin_jhu_pde_refiner_xl'
    EXPERIMENT_NAME = 'swin_jhu_pde_refiner_xl'

    EMBED_DIM = 576
    DEPTHS = [2, 18, 4]
    NUM_HEADS = [9, 18, 36]

    DECODER_DIM = 576
    DECODER_NUM_HEADS = 36
    DECODER_DEPTH = 12

    TEMPORAL_NUM_HEADS = 36

    WINDOW_SIZES = [(8, 8, 8), (8, 8, 8), (4, 4, 4)]

    DROP_PATH_RATE = 0.2

    USE_GRADIENT_CHECKPOINTING = True


class SwinJHUConfigSpectralSmallPatch(SwinJHUConfigSpectral):
    """
    JHU DNS128 Spectral 小patch消融实验配置

    Patch Size & Window Size 消融实验:
    - Patch size: (4, 4, 4) → (2, 2, 2)
    - Patched resolution: (32, 32, 32) → (64, 64, 64)
    - Window size: (8, 8, 8) → (4, 4, 4)  ⚠️ 同步缩小以控制显存

    理论分析:
    - Token数量: 32³ = 32,768 → 64³ = 262,144 (8x↑)
    - Tokens per window: 8³ = 512 → 4³ = 64 (8x↓)
    - Total windows: 4³ = 64 → 16³ = 4,096 (64x↑)
    - 更细粒度的patch可以保留更多空间细节
    - 更小的window size降低每个attention block的计算量

    显存分析:
    - Self-attention复杂度: O(N²) where N=tokens_per_window
    - 单个window计算: 64² = 4,096 vs baseline 512² = 262,144 (减少64倍)
    - 总计算量: 4,096 windows × 4,096 ≈ 64 windows × 262,144 (理论相近)

    其他所有配置与 spectral baseline 完全相同:
    - 相同的 spectral weighted loss
    - 相同的 adaptive weight config
    - 相同的模型架构 (embed_dim, depths, heads)
    """

    CHECKPOINT_DIR = 'Checkpoints/swin_jhu_spectral_patch2x2x2'
    EXPERIMENT_NAME = 'swin_jhu_spectral_small_patch'

    PATCH_SIZE = (2, 2, 2)
    PATCHED_RESOLUTION = (64, 64, 64)

    WINDOW_SIZE = (4, 4, 4)


    @property
    def memory_reduction_factor(self):
        """计算相对于原始分辨率的内存缩减因子"""
        original_volume = 128 * 128 * 128
        patched_volume = 64 * 64 * 64
        return original_volume / patched_volume


class SwinJHUConfig4DSpacetime(SwinJHUConfigSpectral):
    """
    JHU DNS128 4D时空联合Attention配置

    实验目的: 对比分离式 vs 联合式时空建模

    关键差异:
    - 原模型: 3D spatial attention (4×4×4=64 tokens) + 1D temporal attention (T=5 tokens)
    - 本模型: 4D spatiotemporal attention (4×4×4×5=320 tokens)

    配置:
    - Patch size: (4, 4, 4) - 与baseline一致
    - Window size (4D): (T=5, Z=4, H=4, W=4)
    - Tokens per window: 320 (vs baseline 64+5=69)
    - Attention complexity: O(320²) = 102,400 (vs baseline 4,096+25 = 4,121)

    预期:
    - ✅ 能直接建模局部时空模式
    - ⚠️ 计算量增加 ~25x
    - ⚠️ 显存需求更高
    """

    CHECKPOINT_DIR = 'Checkpoints/swin_jhu_4d_spacetime'
    EXPERIMENT_NAME = 'swin_jhu_4d_spacetime'

    WINDOW_SIZE = (4, 4, 4)

    WINDOW_SIZE_4D = (5, 4, 4, 4)


    EMBED_DIM = 256
    DEPTHS = [2, 4, 2]
    NUM_HEADS = [4, 8, 16]
    DECODER_DIM = 256
    DECODER_HEADS = 16


class SwinJHUConfigSpectralTiny(SwinJHUConfigSpectral):
    """Spectral config - Tiny (~10M params). For fast prototyping."""
    CHECKPOINT_DIR = 'checkpoints_swin_jhu_spectral_tiny'
    EXPERIMENT_NAME = 'swin_jhu_spectral_tiny'
    EMBED_DIM = 192
    DEPTHS = [2, 2, 2]
    NUM_HEADS = [3, 6, 12]
    DECODER_DIM = 192
    DECODER_NUM_HEADS = 12
    DECODER_DEPTH = 4
    TEMPORAL_NUM_HEADS = 12


class SwinJHUConfigSpectralSmall(SwinJHUConfigSpectral):
    """Spectral config - Small (~50M params). 50% param reduction from base."""
    CHECKPOINT_DIR = 'checkpoints_swin_jhu_spectral_small'
    EXPERIMENT_NAME = 'swin_jhu_spectral_small'
    EMBED_DIM = 272
    DEPTHS = [2, 4, 2]
    NUM_HEADS = [4, 8, 16]
    DECODER_DIM = 272
    DECODER_NUM_HEADS = 16
    DECODER_DEPTH = 6
    TEMPORAL_NUM_HEADS = 16


class SwinJHUConfigSpectralLarge(SwinJHUConfigSpectral):
    """Spectral config - Large (~250M params). For best performance."""
    CHECKPOINT_DIR = 'checkpoints_swin_jhu_spectral_large'
    EXPERIMENT_NAME = 'swin_jhu_spectral_large'
    EMBED_DIM = 512
    DEPTHS = [2, 8, 4]
    NUM_HEADS = [8, 16, 32]
    DECODER_DIM = 512
    DECODER_NUM_HEADS = 32
    DECODER_DEPTH = 10
    TEMPORAL_NUM_HEADS = 32
    DROP_PATH_RATE = 0.15
    USE_GRADIENT_CHECKPOINTING = True


class SwinJHUConfigTiny(SwinJHUConfig):
    """Baseline config - Tiny (~10M params)."""
    CHECKPOINT_DIR = 'checkpoints_swin_jhu_tiny'
    EXPERIMENT_NAME = 'swin_jhu_tiny'
    EMBED_DIM = 192
    DEPTHS = [2, 2, 2]
    NUM_HEADS = [3, 6, 12]
    DECODER_DIM = 192
    DECODER_NUM_HEADS = 12
    DECODER_DEPTH = 4
    TEMPORAL_NUM_HEADS = 12


class SwinJHUConfigSmall(SwinJHUConfig):
    """Baseline config - Small (~50M params)."""
    CHECKPOINT_DIR = 'checkpoints_swin_jhu_small'
    EXPERIMENT_NAME = 'swin_jhu_small'
    EMBED_DIM = 272
    DEPTHS = [2, 4, 2]
    NUM_HEADS = [4, 8, 16]
    DECODER_DIM = 272
    DECODER_NUM_HEADS = 16
    DECODER_DEPTH = 6
    TEMPORAL_NUM_HEADS = 16


JHU_SWIN_CONFIGS = {
    'baseline': SwinJHUConfig,
    'physical': SwinJHUConfigPhysical,
    'lite': SwinJHUConfigLite,
    'full': SwinJHUConfigFull,
    'large': SwinJHUConfigLarge,
    'spectral': SwinJHUConfigSpectral,
    'spectral_high': SwinJHUConfigSpectralHigh,
    'spectral_band': SwinJHUConfigSpectralBand,
    'spectral_progressive': SwinJHUConfigSpectralProgressive,
    'spectral_progressive_large': SwinJHUConfigSpectralProgressiveLarge,
    'spectral_progressive_giant': SwinJHUConfigSpectralProgressiveGiant,
    'spectral_small_patch': SwinJHUConfigSpectralSmallPatch,
    'dual_residual': SwinJHUConfigDualResidual,
    'pde_refiner_large': SwinJHUConfigPDERefinerLarge,
    'pde_refiner_xl': SwinJHUConfigPDERefinerXL,
    'curriculum': SwinJHUConfigCurriculum,
    'ns5x': SwinJHUConfigNS5x,
    '4d_spacetime': SwinJHUConfig4DSpacetime,
    'tiny': SwinJHUConfigTiny,
    'small': SwinJHUConfigSmall,
    'spectral_tiny': SwinJHUConfigSpectralTiny,
    'spectral_small': SwinJHUConfigSpectralSmall,
    'spectral_large': SwinJHUConfigSpectralLarge,
}


def get_jhu_swin_config(config_name: str = 'baseline'):
    """
    获取JHU Swin配置

    Args:
        config_name: 配置名称 ('baseline', 'lite', 'large')

    Returns:
        配置实例
    """
    config_name = config_name.lower()
    if config_name not in JHU_SWIN_CONFIGS:
        raise ValueError(
            f"Unknown JHU config: {config_name}. "
            f"Available: {list(JHU_SWIN_CONFIGS.keys())}"
        )
    return JHU_SWIN_CONFIGS[config_name]()


if __name__ == "__main__":
    print("\n### JHU Baseline Config ###")
    config = SwinJHUConfig()
    config.print_config()

    print("\n\n### JHU Lite Config ###")
    config_lite = SwinJHUConfigLite()
    config_lite.print_config()

    print("\n\n### DNS vs JHU Comparison ###")
    dns_config = SwinPatchedConfigFixed()
    jhu_config = SwinJHUConfig()

    print(f"\n{'Parameter':<25} {'DNS':<20} {'JHU':<20}")
    print("=" * 65)
    print(f"{'INPUT_RESOLUTION':<25} {str(dns_config.INPUT_RESOLUTION):<20} {str(jhu_config.INPUT_RESOLUTION):<20}")
    print(f"{'PATCH_SIZE':<25} {str(dns_config.PATCH_SIZE):<20} {str(jhu_config.PATCH_SIZE):<20}")
    print(f"{'PATCHED_RESOLUTION':<25} {str(dns_config.PATCHED_RESOLUTION):<20} {str(jhu_config.PATCHED_RESOLUTION):<20}")
    print(f"{'NUM_SPARSE_Z_LAYERS':<25} {dns_config.NUM_SPARSE_Z_LAYERS:<20} {jhu_config.NUM_SPARSE_Z_LAYERS:<20}")
    print(f"{'WINDOW_SIZE':<25} {str(dns_config.WINDOW_SIZE):<20} {str(jhu_config.WINDOW_SIZE):<20}")
