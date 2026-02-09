"""损失工厂 (修复版本)"""
import torch
import torch.nn as nn
from .data_loss import get_data_loss
from .physics_loss import get_physics_loss
from .spectral_loss import get_spectral_loss
from .spectral_loss_dynamic import get_dynamic_spectral_loss


class LossFactory:
    """
    损失函数工厂

    统一管理多个损失函数，支持：
    1. Data Loss: 只对missing层计算
    2. Physics Loss: 根据空间维度特性智能选择
       - 涉及Z方向导数(层间耦合) → 必须用完整场
       - 仅在层内(XY平面) → 可以只用missing层
    """

    def __init__(self, config, norm_mean=None, norm_std=None):
        # 兼容两种配置格式:
        # 1. config对象 (有LOSS_CONFIG属性)
        # 2. dict (直接就是loss_config)
        if isinstance(config, dict):
            # 如果传入的是dict，包装成一个简单的对象
            class ConfigWrapper:
                pass
            wrapper = ConfigWrapper()
            wrapper.LOSS_CONFIG = config
            self.config = wrapper
        else:
            self.config = config

        self.loss_fns = {}
        self.weights = {}

        # 归一化参数（用于物理损失的反归一化）
        self.norm_mean = norm_mean
        self.norm_std = norm_std

        # 细粒度控制：根据物理损失的空间特性决定计算策略
        self.loss_spatial_mode = {
            # 物理空间（有限差分）
            'divergence_2d': 'in_plane',      # 仅XY平面 ∂u/∂x + ∂v/∂y
            'divergence_3d': 'full_field',    # 有∂w/∂z，需要Z方向连续层
            'navier_stokes': 'full_field',    # 有所有∂/∂z项，需要Z方向连续层
            'vorticity': 'in_plane',          # 仅XY平面涡量
            # 谱域（FFT）- 全场计算
            'spectral_divergence': 'full_field',
            'spectral_divergence_soft': 'full_field',
            'spectral_ns': 'full_field',
            'spectral_weighted': 'full_field',  # 频谱加权损失需要完整场做FFT
            'spectral_dynamic': 'full_field',  # 动态频谱加权损失需要完整场做FFT
            'spectral_relative': 'full_field',  # 相对频谱损失需要完整场做FFT
            # 梯度和平滑损失 - 需要 pred 和 target 比较
            'gradient_3d': 'full_field',
            'tv_3d': 'full_field',
            'edge_smooth': 'full_field',
            'laplacian': 'full_field',
        }

        # 向后兼容：如果配置中有PHYSICS_ON_MISSING_ONLY，用作默认策略
        if hasattr(config, 'PHYSICS_ON_MISSING_ONLY') and config.PHYSICS_ON_MISSING_ONLY:
            # 如果全局标志为True，强制所有物理损失只用missing层
            for key in self.loss_spatial_mode.keys():
                self.loss_spatial_mode[key] = 'in_plane'

        self._build_losses()
    
    def _build_losses(self):
        """构建所有损失函数"""
        for loss_name, loss_config in self.config.LOSS_CONFIG.items():
            loss_type = loss_config['type']
            weight = loss_config.get('weight', 1.0)

            if loss_type in ['mse', 'mae', 'l1', 'huber', 'relative_l2']:
                loss_fn = get_data_loss(loss_type)
            elif loss_type in ['divergence_2d', 'divergence_3d', 'navier_stokes', 'navier_stokes_forced_turb',
                               'vorticity', 'spectral_divergence', 'spectral_divergence_soft', 'spectral_ns',
                               'gradient_3d', 'tv_3d', 'edge_smooth', 'laplacian']:
                # 将归一化参数加入配置，传给物理损失
                physics_config = dict(loss_config)
                physics_config['norm_mean'] = self.norm_mean
                physics_config['norm_std'] = self.norm_std
                loss_fn = get_physics_loss(loss_type, physics_config)
            elif loss_type in ['fourier', 'wavelet', 'energy_spectrum', 'spectral_weighted', 'spectral_relative']:
                # 频谱损失 (包括频谱加权损失和相对频谱损失)
                loss_fn = get_spectral_loss(loss_type, loss_config)
            elif loss_type == 'spectral_dynamic':
                # 动态频谱加权损失
                loss_fn = get_dynamic_spectral_loss(loss_config)
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")

            self.loss_fns[loss_name] = loss_fn
            self.weights[loss_name] = weight
    
    def get_loss_names(self):
        """返回所有注册的损失函数名称"""
        return list(self.loss_fns.keys())

    def set_epoch(self, epoch):
        """设置当前epoch，用于动态损失函数"""
        for loss_fn in self.loss_fns.values():
            if hasattr(loss_fn, 'set_epoch'):
                loss_fn.set_epoch(epoch)

    def get_loss_strategy(self):
        """
        返回当前损失函数的计算策略

        Returns:
            dict: {loss_name: strategy}
                - 'missing_only': 只对missing层计算
                - 'full_field': 对完整场计算
        """
        strategy = {}
        for loss_name in self.loss_fns.keys():
            if loss_name == 'data':
                strategy[loss_name] = 'missing_only'
            else:
                loss_type = self.config.LOSS_CONFIG[loss_name]['type']
                spatial_mode = self.loss_spatial_mode.get(loss_type, 'full_field')
                strategy[loss_name] = spatial_mode
        return strategy
    
    def compute(self, pred, target, missing_indices=None, pred_full=None, target_full=None):
        """
        计算所有损失

        Args:
            pred: (B, T, C, Z, H, W) - 预测 (归一化空间)
            target: (B, T, C, Z, H, W) - 真值 (归一化空间)
            missing_indices: (B, n_missing) - 缺失z层索引
            pred_full: 用于physics loss的完整预测 (可选，归一化空间)
            target_full: 用于physics loss的完整真值 (可选，归一化空间)

        Returns:
            losses: dict - {'data': ..., 'divergence': ..., 'total': ...}
        """
        losses = {}
        total_loss = 0.0

        for loss_name, loss_fn in self.loss_fns.items():
            loss_type = self.config.LOSS_CONFIG[loss_name]['type']
            loss_config = self.config.LOSS_CONFIG[loss_name]

            if loss_name == 'data':
                # Data Loss: 根据类型决定计算方式
                if loss_type in ['spectral_weighted', 'spectral_dynamic', 'spectral_relative']:
                    # 频谱损失需要完整场做FFT (FFT需要完整空间结构才能正确分解频率)
                    # 注: sparse层的pred=GT，所以这些层的误差贡献=0，不影响优化
                    # 检查配置中的target设置
                    target_mode = loss_config.get('target', 'full')
                    if target_mode == 'full' and pred_full is not None:
                        loss_val = loss_fn(pred_full, target_full)
                    else:
                        # 如果没有full数据，回退到pred/target
                        loss_val = loss_fn(pred, target)

                    # 获取频带误差详情 (用于监控)
                    if hasattr(loss_fn, 'get_last_band_errors'):
                        band_errors = loss_fn.get_last_band_errors()
                        losses['data_low'] = band_errors.get('low', 0.0)
                        losses['data_mid'] = band_errors.get('mid', 0.0)
                        losses['data_high'] = band_errors.get('high', 0.0)
                else:
                    # 普通数据损失 (MSE/L1等): 只对missing层计算
                    loss_val = loss_fn(pred, target, missing_indices=missing_indices)
            else:
                # Physics Loss: 根据空间维度特性智能选择
                spatial_mode = self.loss_spatial_mode.get(loss_type, 'full_field')

                if spatial_mode == 'in_plane' and missing_indices is not None:
                    # 层内约束（无Z方向导数）: 只对missing层计算
                    # 理由: 每层独立，sparse层是GT复制会稀释物理约束
                    pred_physics = self._extract_missing(pred_full if pred_full is not None else pred, missing_indices)
                    target_physics = self._extract_missing(target_full if target_full is not None else target, missing_indices)
                    loss_val = loss_fn(pred_physics, target_physics)
                else:
                    # 层间约束（有Z方向导数）: 必须用完整场
                    # 理由: Z方向导数需要连续的层，sparse层之间gap太大
                    if pred_full is not None:
                        loss_val = loss_fn(pred_full, target_full)
                    else:
                        loss_val = loss_fn(pred, target)

            # 加权累加
            weighted_loss = self.weights[loss_name] * loss_val
            losses[loss_name] = loss_val
            total_loss += weighted_loss

        losses['total'] = total_loss
        return losses
    
    def _extract_missing(self, volume, missing_indices):
        """
        提取missing层
        
        Args:
            volume: (B, T, C, Z, H, W)
            missing_indices: (B, n_missing)
        
        Returns:
            missing_volume: (B, T, C, n_missing, H, W)
        """
        B = volume.shape[0]
        missing_slices = []
        
        for b in range(B):
            missing_z = missing_indices[b]  # (n_missing,)
            missing_slices.append(volume[b, :, :, missing_z, :, :])  # (T, C, n_missing, H, W)
        
        return torch.stack(missing_slices, dim=0)  # (B, T, C, n_missing, H, W)


def build_loss_factory(config, norm_mean=None, norm_std=None):
    """
    构建损失工厂

    Args:
        config: 配置对象
        norm_mean: 归一化均值 (C,) - 用于物理损失反归一化
        norm_std: 归一化标准差 (C,) - 用于物理损失反归一化

    Returns:
        LossFactory实例
    """
    return LossFactory(config, norm_mean=norm_mean, norm_std=norm_std)