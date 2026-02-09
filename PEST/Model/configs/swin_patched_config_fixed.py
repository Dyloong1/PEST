"""
修复后的 Patched Swin Transformer 配置

兼容 LossFactory 和完整训练链路
"""

class SwinPatchedConfigFixed:
    """完整的训练配置 - 兼容所有组件"""

    CHECKPOINT_DIR = 'checkpoints_swin'
    EXPERIMENT_NAME = 'swin_baseline'

    import math
    _DOMAIN_SIZE = 2 * math.pi
    _GRID_Z = 64
    _GRID_HW = 128
    _RE = 1600

    DZ_PHYSICAL = _DOMAIN_SIZE / _GRID_Z
    DX_PHYSICAL = _DOMAIN_SIZE / _GRID_HW
    DY_PHYSICAL = _DOMAIN_SIZE / _GRID_HW
    NU_PHYSICAL = 1.0 / _RE

    IN_CHANNELS = 4
    INPUT_RESOLUTION = (64, 128, 128)

    NUM_SPARSE_Z_LAYERS = 12

    PATCH_SIZE = (2, 4, 4)
    PATCHED_RESOLUTION = (32, 32, 32)

    WINDOW_SIZE = (8, 8, 8)

    EMBED_DIM = 384
    DEPTHS = [2, 6, 2]
    NUM_HEADS = [6, 12, 24]
    MLP_RATIO = 4.0
    ATTENTION_TYPE = 'standard'

    TEMPORAL_NUM_HEADS = 24

    DECODER_DIM = 384
    DECODER_NUM_HEADS = 24
    DECODER_DEPTH = 8

    DROP_RATE = 0.0
    ATTN_DROP_RATE = 0.0
    DROP_PATH_RATE = 0.1
    USE_GRADIENT_CHECKPOINTING = True

    LOSS_CONFIG = {
        'data': {
            'type': 'mse',
            'weight': 1.0,
            'target': 'missing'
        },
        'divergence': {
            'type': 'divergence_3d',
            'weight': 50.0,
            'dx': DX_PHYSICAL,
            'dy': DY_PHYSICAL,
            'dz': DZ_PHYSICAL
        },
        'navier_stokes': {
            'type': 'navier_stokes',
            'weight': 5.0,
            'nu': NU_PHYSICAL,
            'rho': 1.0,
            'dx': DX_PHYSICAL,
            'dy': DY_PHYSICAL,
            'dz': DZ_PHYSICAL,
            'dt': 0.125
        }
    }

    PHYSICS_ON_MISSING_ONLY = False

    ADAPTIVE_WEIGHT_CONFIG = {
        'enabled': False,
        'update_frequency': 100,
        'warmup_steps': 500,
        'min_weight': 0.5,
        'max_weight': 20.0,
        'target_ratio': 0.5,
        'adjustment_rate': 0.1,
        'patience': 3
    }

    COMPLETION_METHOD = 'interpolation'

    FOURIER_CONFIG = {
        'd_model': 256,
        'freq_size': 32,
        'n_heads': 8,
        'n_layers': 4,
        'dropout': 0.1,
    }

    INTERPOLATION_CONFIG = {
        'use_refinement': True,
    }

    FOURIER_INTERMEDIATE_WEIGHT = 0.5

    @property
    def memory_reduction_factor(self):
        """Calculate memory reduction from patching."""
        original_size = self.INPUT_RESOLUTION[0] * self.INPUT_RESOLUTION[1] * self.INPUT_RESOLUTION[2]
        patched_size = self.PATCHED_RESOLUTION[0] * self.PATCHED_RESOLUTION[1] * self.PATCHED_RESOLUTION[2]
        return original_size / patched_size

    def print_config(self):
        """Print full configuration."""
        print("=" * 80)
        print("Patched Swin Transformer - Complete Configuration")
        print("=" * 80)

        print(f"\n[Data]")
        print(f"  Input channels: {self.IN_CHANNELS}")
        print(f"  Resolution: {self.INPUT_RESOLUTION}")
        print(f"  Sparse z-layers (total count): {self.NUM_SPARSE_Z_LAYERS}")

        print(f"\n[Patch]")
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
        if self.COMPLETION_METHOD == 'fourier':
            print(f"  Fourier config: {self.FOURIER_CONFIG}")
            print(f"  Intermediate weight: {self.FOURIER_INTERMEDIATE_WEIGHT}")
        else:
            print(f"  Interpolation config: {self.INTERPOLATION_CONFIG}")

        print("=" * 80)


class SwinPatchedConfigLiteFixed(SwinPatchedConfigFixed):
    """Lite version for extremely limited memory."""

    EMBED_DIM = 64
    DEPTHS = [2]
    NUM_HEADS = [4]
    DECODER_DIM = 96
    TEMPORAL_NUM_HEADS = 4

    LOSS_CONFIG = {
        'data': {
            'type': 'mse',
            'weight': 1.0,
            'target': 'missing'
        },
        'divergence': {
            'type': 'divergence_3d',
            'weight': 20.0,
            'dx': SwinPatchedConfigFixed.DX_PHYSICAL,
            'dy': SwinPatchedConfigFixed.DY_PHYSICAL,
            'dz': SwinPatchedConfigFixed.DZ_PHYSICAL
        }
    }


class SwinPatchedConfigLargeFixed(SwinPatchedConfigFixed):
    """Large version for better performance."""

    EMBED_DIM = 128
    DEPTHS = [2, 2, 4]
    NUM_HEADS = [4, 8, 16]
    DECODER_DIM = 192
    TEMPORAL_NUM_HEADS = 12
    DROP_PATH_RATE = 0.2


class SwinDNSConfigBaselineV2(SwinPatchedConfigFixed):
    """
    DNS Baseline V2 配置 (与 JHU baseline 保持一致的权重)

    使用与 JHU 一致的权重设置:
    - Data loss (MSE): weight=1.0
    - Divergence loss: weight=1.0 (原50.0)
    - Navier-Stokes loss: weight=0.1 (原5.0)
    - Adaptive weights enabled

    物理参数使用真实物理单位 (继承自父类):
    - dx, dy, dz: 物理网格间距
    - nu: 物理运动粘度 1/Re = 0.000625
    """

    CHECKPOINT_DIR = 'checkpoints_dns_baseline_v2'
    EXPERIMENT_NAME = 'dns_baseline_v2'

    LOSS_CONFIG = {
        'data': {
            'type': 'mse',
            'weight': 1.0,
            'target': 'missing'
        },
        'divergence': {
            'type': 'divergence_3d',
            'weight': 1.0,
            'dx': SwinPatchedConfigFixed.DX_PHYSICAL,
            'dy': SwinPatchedConfigFixed.DY_PHYSICAL,
            'dz': SwinPatchedConfigFixed.DZ_PHYSICAL
        },
        'navier_stokes': {
            'type': 'navier_stokes',
            'weight': 0.1,
            'nu': SwinPatchedConfigFixed.NU_PHYSICAL,
            'rho': 1.0,
            'dx': SwinPatchedConfigFixed.DX_PHYSICAL,
            'dy': SwinPatchedConfigFixed.DY_PHYSICAL,
            'dz': SwinPatchedConfigFixed.DZ_PHYSICAL,
            'dt': 0.125
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


class SwinDNSConfigGradient(SwinPatchedConfigFixed):
    """
    DNS Gradient 配置 (与 JHU gradient 保持一致)

    在 baseline_v2 基础上添加 gradient loss:
    - Data loss (MSE): weight=1.0
    - Divergence loss: weight=1.0
    - Navier-Stokes loss: weight=0.1
    - Gradient loss: weight=0.1
    - Adaptive weights enabled

    物理参数使用真实物理单位 (继承自父类)
    """

    CHECKPOINT_DIR = 'checkpoints_dns_gradient'
    EXPERIMENT_NAME = 'dns_gradient'

    LOSS_CONFIG = {
        'data': {
            'type': 'mse',
            'weight': 1.0,
            'target': 'missing'
        },
        'divergence': {
            'type': 'divergence_3d',
            'weight': 1.0,
            'dx': SwinPatchedConfigFixed.DX_PHYSICAL,
            'dy': SwinPatchedConfigFixed.DY_PHYSICAL,
            'dz': SwinPatchedConfigFixed.DZ_PHYSICAL
        },
        'navier_stokes': {
            'type': 'navier_stokes',
            'weight': 0.1,
            'nu': SwinPatchedConfigFixed.NU_PHYSICAL,
            'rho': 1.0,
            'dx': SwinPatchedConfigFixed.DX_PHYSICAL,
            'dy': SwinPatchedConfigFixed.DY_PHYSICAL,
            'dz': SwinPatchedConfigFixed.DZ_PHYSICAL,
            'dt': 0.125
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
        'min_weight': 0.01,
        'max_weight': 10.0,
        'target_ratio': 0.5,
        'adjustment_rate': 0.1,
        'patience': 2
    }


class SwinDNSConfigSpectral(SwinPatchedConfigFixed):
    """
    DNS 频谱加权Loss版本配置

    与 JHU SwinJHUConfigSpectral 保持一致:
    - Spectral-weighted data loss (high_freq_weight=3.0, band weighting)
    - Divergence loss (weight=1.0)
    - Navier-Stokes loss (weight=0.1)
    - Gradient loss (weight=0.1)
    - Adaptive weight enabled

    DNS Taylor-Green 物理参数:
    - Domain: 2π × 2π × 2π
    - Grid: 64 × 128 × 128
    - Re = 1600
    - 使用真实物理单位
    """

    CHECKPOINT_DIR = 'checkpoints_dns_spectral'
    EXPERIMENT_NAME = 'dns_spectral'

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
            'dx': SwinPatchedConfigFixed.DX_PHYSICAL,
            'dy': SwinPatchedConfigFixed.DY_PHYSICAL,
            'dz': SwinPatchedConfigFixed.DZ_PHYSICAL
        },
        'navier_stokes': {
            'type': 'navier_stokes',
            'weight': 0.1,
            'nu': SwinPatchedConfigFixed.NU_PHYSICAL,
            'rho': 1.0,
            'dx': SwinPatchedConfigFixed.DX_PHYSICAL,
            'dy': SwinPatchedConfigFixed.DY_PHYSICAL,
            'dz': SwinPatchedConfigFixed.DZ_PHYSICAL,
            'dt': 0.125
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
        'min_weight': 0.01,
        'max_weight': 10.0,
        'target_ratio': 0.5,
        'adjustment_rate': 0.1,
        'patience': 2
    }


class SwinDNSConfigSpectralV2(SwinPatchedConfigFixed):
    """
    DNS 频谱加权Loss V2 - 真正的自适应权重

    相比 spectral 配置的改进:
    - 真正的自适应权重调整 (根据各频带误差动态调整)
    - band_weights: [1.0, 2.0, 5.0] 作为初始值
    - 自适应将根据误差分布自动调整到更高值
    - target_ratio_mode: 'error_based' - 误差越大，权重越大

    目标: 让模型自动找到最优的频带权重
    使用真实物理单位
    """

    CHECKPOINT_DIR = 'checkpoints_dns_spectral_v2'
    EXPERIMENT_NAME = 'dns_spectral_v2'

    LOSS_CONFIG = {
        'data': {
            'type': 'spectral_weighted',
            'weight': 1.0,
            'high_freq_weight': 5.0,
            'weighting_mode': 'band',
            'adaptive': True,
            'adaptive_config': {
                'adaptation_rate': 0.05,
                'min_band_weight': 0.5,
                'max_band_weight': 50.0,
                'warmup_steps': 200,
                'target_ratio_mode': 'boost_high'
            },
            'band_weights': [1.0, 2.0, 5.0],
            'target': 'full'
        },
        'divergence': {
            'type': 'divergence_3d',
            'weight': 1.0,
            'dx': SwinPatchedConfigFixed.DX_PHYSICAL,
            'dy': SwinPatchedConfigFixed.DY_PHYSICAL,
            'dz': SwinPatchedConfigFixed.DZ_PHYSICAL
        },
        'navier_stokes': {
            'type': 'navier_stokes',
            'weight': 0.1,
            'nu': SwinPatchedConfigFixed.NU_PHYSICAL,
            'rho': 1.0,
            'dx': SwinPatchedConfigFixed.DX_PHYSICAL,
            'dy': SwinPatchedConfigFixed.DY_PHYSICAL,
            'dz': SwinPatchedConfigFixed.DZ_PHYSICAL,
            'dt': 0.125
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
        'min_weight': 0.01,
        'max_weight': 10.0,
        'target_ratio': 0.5,
        'adjustment_rate': 0.1,
        'patience': 2
    }


class SwinDNSConfigSpectralGradient(SwinPatchedConfigFixed):
    """
    DNS 相对频谱Loss + Gradient Loss 组合

    特点:
    - 使用 Relative Spectral Loss 替代 MSE data loss (根据Parseval定理等价)
    - 相对误差自动适应不同样本的频谱特性 (从稳定态到湍流态)
    - 频段划分基于物理尺度: 大尺度(0-3%), 惯性区(3-8%), 耗散区(8-100%)
    - 叠加 Gradient Loss 增强空间细节
    - 加上 NS 和 Divergence 物理约束

    Loss 组成 (使用物理单位，量纲一致):
    - data (spectral_relative): weight=1.0, 相对频谱误差
    - gradient: weight=1.0, 空间梯度损失
    - divergence: weight=0.1, 散度约束 (物理单位下量级放大~415x)
    - navier_stokes: weight=1.0, NS方程约束 (物理单位下量级放大~6x)

    物理参数 (DNS Taylor-Green, Re=1600):
    - dx = dy = 2π/128 ≈ 0.049 (物理网格间距)
    - dz = 2π/64 ≈ 0.098 (物理网格间距)
    - nu = 1/Re = 0.000625 (物理运动粘度)
    - dt = 0.125 (无量纲时间步长)
    """

    CHECKPOINT_DIR = 'checkpoints_dns_spectral_gradient'
    EXPERIMENT_NAME = 'dns_spectral_gradient'

    LOSS_CONFIG = {
        'data': {
            'type': 'spectral_relative',
            'weight': 1.0,
            'band_boundaries': [0.03, 0.08],
            'band_weights': [1.0, 1.0, 1.0],
            'target': 'full'
        },
        'gradient': {
            'type': 'gradient_3d',
            'weight': 1.0
        },
        'divergence': {
            'type': 'divergence_3d',
            'weight': 0.1,
            'dx': SwinPatchedConfigFixed.DX_PHYSICAL,
            'dy': SwinPatchedConfigFixed.DY_PHYSICAL,
            'dz': SwinPatchedConfigFixed.DZ_PHYSICAL
        },
        'navier_stokes': {
            'type': 'navier_stokes',
            'weight': 1.0,
            'nu': SwinPatchedConfigFixed.NU_PHYSICAL,
            'rho': 1.0,
            'dx': SwinPatchedConfigFixed.DX_PHYSICAL,
            'dy': SwinPatchedConfigFixed.DY_PHYSICAL,
            'dz': SwinPatchedConfigFixed.DZ_PHYSICAL,
            'dt': 0.125
        }
    }

    ADAPTIVE_WEIGHT_CONFIG = {
        'enabled': True,
        'update_frequency': 50,
        'warmup_steps': 200,
        'min_weight': 0.01,
        'max_weight': 10.0,
        'target_ratio': 1.0,
        'adjustment_rate': 0.1,
        'patience': 2,
        'gradient_target_ratio': 0.22,
        'gradient_min_weight': 0.1,
        'gradient_max_weight': 5.0
    }


class SwinDNSConfigUNet(SwinDNSConfigSpectralGradient):
    """
    U-Net Style DNS Configuration

    基于 SWIn paper (4D Swin Transformer for Coastal Ocean Circulation) 的架构设计:
    - U-Net encoder-decoder with hierarchical downsampling
    - Patch Merging: (32,32,32) → (16,16,16) → (8,8,8)
    - Progressive window sizes: (8,8,8) → (4,4,4) → (4,4,4)
    - Skip connections for spatial detail preservation

    继承 spectral_gradient 的 loss 配置 (最佳实践):
    - spectral_relative: 相对频谱误差，自动适应不同样本
    - gradient: 独立自适应调整
    - 物理约束: divergence + navier_stokes
    """

    CHECKPOINT_DIR = 'checkpoints_dns_unet'
    EXPERIMENT_NAME = 'dns_unet'

    USE_UNET_ARCHITECTURE = True


    EMBED_DIM = 96
    DEPTHS = [2, 2, 2]
    NUM_HEADS = [3, 6, 12]

    WINDOW_SIZES_PER_STAGE = [(8, 8, 8), (4, 4, 4), (4, 4, 4)]

    WINDOW_SIZE = (8, 8, 8)

    DECODER_DEPTHS = [2, 2]
    DECODER_NUM_HEADS_PER_STAGE = [6, 3]

    TEMPORAL_NUM_HEADS = 12

    def print_config(self):
        """Print U-Net specific configuration."""
        super().print_config()

        print(f"\n[U-Net Architecture]")
        print(f"  Enabled: {self.USE_UNET_ARCHITECTURE}")
        print(f"  Progressive windows: {self.WINDOW_SIZES_PER_STAGE}")
        print(f"  Encoder resolutions: (32,32,32) → (16,16,16) → (8,8,8)")
        print(f"  Encoder dims: [{self.EMBED_DIM}, {self.EMBED_DIM*2}, {self.EMBED_DIM*4}]")
        print(f"  Decoder depths: {self.DECODER_DEPTHS}")


class SwinDNSConfigUNetLarge(SwinDNSConfigUNet):
    """
    U-Net Large Configuration - 更大的模型容量

    增大 EMBED_DIM 以提升模型容量
    """

    CHECKPOINT_DIR = 'checkpoints_dns_unet_large'
    EXPERIMENT_NAME = 'dns_unet_large'

    EMBED_DIM = 128
    NUM_HEADS = [4, 8, 16]
    TEMPORAL_NUM_HEADS = 16
    DECODER_NUM_HEADS_PER_STAGE = [8, 4]


class SwinDNSConfigTiny(SwinPatchedConfigFixed):
    """DNS config - Tiny (~10M params). For fast prototyping."""
    CHECKPOINT_DIR = 'checkpoints_dns_tiny'
    EXPERIMENT_NAME = 'dns_tiny'
    EMBED_DIM = 192
    DEPTHS = [2, 2, 2]
    NUM_HEADS = [3, 6, 12]
    DECODER_DIM = 192
    DECODER_NUM_HEADS = 12
    DECODER_DEPTH = 4
    TEMPORAL_NUM_HEADS = 12


class SwinDNSConfigSmall(SwinPatchedConfigFixed):
    """DNS config - Small (~50M params). 50% param reduction from base."""
    CHECKPOINT_DIR = 'checkpoints_dns_small'
    EXPERIMENT_NAME = 'dns_small'
    EMBED_DIM = 272
    DEPTHS = [2, 4, 2]
    NUM_HEADS = [4, 8, 16]
    DECODER_DIM = 272
    DECODER_NUM_HEADS = 16
    DECODER_DEPTH = 6
    TEMPORAL_NUM_HEADS = 16


class SwinDNSConfigXLarge(SwinPatchedConfigFixed):
    """DNS config - XLarge (~250M params)."""
    CHECKPOINT_DIR = 'checkpoints_dns_xlarge'
    EXPERIMENT_NAME = 'dns_xlarge'
    EMBED_DIM = 512
    DEPTHS = [2, 8, 4]
    NUM_HEADS = [8, 16, 32]
    DECODER_DIM = 512
    DECODER_NUM_HEADS = 32
    DECODER_DEPTH = 10
    TEMPORAL_NUM_HEADS = 32
    DROP_PATH_RATE = 0.15
    USE_GRADIENT_CHECKPOINTING = True


DNS_SWIN_CONFIGS = {
    'baseline': SwinPatchedConfigFixed,
    'baseline_v2': SwinDNSConfigBaselineV2,
    'gradient': SwinDNSConfigGradient,
    'spectral_mid': SwinDNSConfigSpectral,
    'spectral_v2': SwinDNSConfigSpectralV2,
    'spectral_gradient': SwinDNSConfigSpectralGradient,
    'unet': SwinDNSConfigUNet,
    'unet_large': SwinDNSConfigUNetLarge,
    'lite': SwinPatchedConfigLiteFixed,
    'large': SwinPatchedConfigLargeFixed,
    'tiny': SwinDNSConfigTiny,
    'small': SwinDNSConfigSmall,
    'xlarge': SwinDNSConfigXLarge,
}


def get_dns_swin_config(config_name: str = 'baseline'):
    """
    获取DNS Swin配置

    Args:
        config_name: 配置名称 ('baseline', 'lite', 'large', 'spectral_mid')

    Returns:
        配置实例
    """
    config_name = config_name.lower()
    if config_name not in DNS_SWIN_CONFIGS:
        raise ValueError(
            f"Unknown DNS config: {config_name}. "
            f"Available: {list(DNS_SWIN_CONFIGS.keys())}"
        )
    return DNS_SWIN_CONFIGS[config_name]()


if __name__ == "__main__":
    print("\n### Standard Config ###")
    config = SwinPatchedConfigFixed()
    config.print_config()

    print("\n\n### Lite Config ###")
    config_lite = SwinPatchedConfigLiteFixed()
    config_lite.print_config()

    print("\n\n### Large Config ###")
    config_large = SwinPatchedConfigLargeFixed()
    config_large.print_config()
