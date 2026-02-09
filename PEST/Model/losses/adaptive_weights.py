"""
自适应损失权重管理器

根据训练过程中各损失项的相对大小，动态调整权重
目标：
1. 物理损失 (div + ns) 保持与数据损失的指定比例
2. 梯度损失 独立保持与数据损失的指定比例（辅助作用）
"""

import torch
import numpy as np
from collections import deque


class AdaptiveWeightManager:
    """
    自适应权重管理器

    策略：
    1. 监控 physics_loss / data_loss 的比值 → 调整 div/ns 权重
    2. 监控 gradient_loss / data_loss 的比值 → 调整 gradient 权重
    3. 两个比值独立调整，避免相互干扰
    """

    def __init__(self, config):
        self.config = config
        self.adaptive_config = config.ADAPTIVE_WEIGHT_CONFIG

        # 是否启用动态权重
        self.enabled = self.adaptive_config.get('enabled', False)

        # 基本参数
        self.update_frequency = self.adaptive_config.get('update_frequency', 100)
        self.warmup_steps = self.adaptive_config.get('warmup_steps', 500)
        self.min_weight = self.adaptive_config.get('min_weight', 0.5)
        self.max_weight = self.adaptive_config.get('max_weight', 20.0)
        self.adjustment_rate = self.adaptive_config.get('adjustment_rate', 0.1)
        self.patience = self.adaptive_config.get('patience', 5)

        # 目标比值 (相对于 data loss)
        self.physics_target_ratio = self.adaptive_config.get('target_ratio', 0.5)  # div+ns / data
        self.gradient_target_ratio = self.adaptive_config.get('gradient_target_ratio', 0.15)  # gradient / data

        # Gradient 独立的权重范围 (辅助作用，不需要太大)
        self.gradient_min_weight = self.adaptive_config.get('gradient_min_weight', 0.1)
        self.gradient_max_weight = self.adaptive_config.get('gradient_max_weight', 10.0)

        # 状态变量
        self.current_step = 0
        self.current_weights = {}
        self.loss_history = {
            'data': deque(maxlen=self.update_frequency),
            'physics': deque(maxlen=self.update_frequency),
            'gradient': deque(maxlen=self.update_frequency)
        }

        # 调整方向跟踪（避免震荡）- 分别跟踪 physics 和 gradient
        self.physics_direction = 0
        self.physics_direction_count = 0
        self.gradient_direction = 0
        self.gradient_direction_count = 0

        # 初始化权重
        self._initialize_weights()

        print(f"[AdaptiveWeightManager] 初始化:")
        if self.enabled:
            print(f"  模式: 动态权重 (physics 和 gradient 独立调整)")
            print(f"  更新频率: 每{self.update_frequency}步")
            print(f"  预热步数: {self.warmup_steps}")
            print(f"  Physics (div+ns):")
            print(f"    目标比值: {self.physics_target_ratio}")
            print(f"    权重范围: [{self.min_weight}, {self.max_weight}]")
            print(f"  Gradient:")
            print(f"    目标比值: {self.gradient_target_ratio}")
            print(f"    权重范围: [{self.gradient_min_weight}, {self.gradient_max_weight}]")
        else:
            print(f"  模式: 固定权重 (不自动调整)")
            print(f"  divergence: {self.current_weights.get('divergence', 50.0)}")
            print(f"  navier_stokes: {self.current_weights.get('navier_stokes', 5.0)}")
            print(f"  gradient: {self.current_weights.get('gradient', 1.0)}")

    def _initialize_weights(self):
        """初始化当前权重"""
        loss_config = self.config.LOSS_CONFIG

        for loss_name, loss_cfg in loss_config.items():
            if loss_name != 'data':
                self.current_weights[loss_name] = loss_cfg.get('weight', 1.0)

        print(f"  初始权重: {self.current_weights}")

    def step(self, losses):
        """
        在每个训练步骤后调用

        Args:
            losses: dict, 包含各项损失的字典
                   {'data': ..., 'divergence': ..., 'navier_stokes': ..., 'gradient': ...}

        Returns:
            dict: 更新后的权重字典（如果有更新）, 否则返回None
        """
        self.current_step += 1

        # 如果未启用动态权重，直接返回
        if not self.enabled:
            return None

        # 记录损失（确保转换为 Python float，避免 CUDA tensor 问题）
        data_loss = losses.get('data', 0.0)
        if hasattr(data_loss, 'item'):
            data_loss = data_loss.item()

        # 计算物理损失总和 (divergence + navier_stokes)
        physics_loss = 0.0
        for loss_name in ['divergence', 'navier_stokes']:
            if loss_name in losses:
                val = losses[loss_name]
                if hasattr(val, 'item'):
                    val = val.item()
                physics_loss += val

        # 记录 gradient loss
        gradient_loss = losses.get('gradient', 0.0)
        if hasattr(gradient_loss, 'item'):
            gradient_loss = gradient_loss.item()

        self.loss_history['data'].append(data_loss)
        self.loss_history['physics'].append(physics_loss)
        self.loss_history['gradient'].append(gradient_loss)

        # 检查是否需要更新
        if not self._should_update():
            return None

        # 计算平均损失
        avg_data_loss = np.mean(self.loss_history['data'])
        avg_physics_loss = np.mean(self.loss_history['physics'])
        avg_gradient_loss = np.mean(self.loss_history['gradient'])

        if avg_data_loss < 1e-8:
            return None

        # 计算当前比值
        physics_ratio = avg_physics_loss / avg_data_loss
        gradient_ratio = avg_gradient_loss / avg_data_loss

        updated_weights = {}

        # === 调整 Physics 权重 (div + ns) ===
        physics_update = self._compute_physics_weights(physics_ratio)
        if physics_update:
            updated_weights.update(physics_update)

        # === 调整 Gradient 权重 ===
        gradient_update = self._compute_gradient_weight(gradient_ratio)
        if gradient_update:
            updated_weights.update(gradient_update)

        # 打印调整信息
        if updated_weights:
            print(f"\n[AdaptiveWeights] Step {self.current_step}:")
            print(f"  平均 data loss: {avg_data_loss:.6f}")
            print(f"  Physics (div+ns): ratio={physics_ratio:.4f} (目标: {self.physics_target_ratio})")
            print(f"  Gradient: ratio={gradient_ratio:.4f} (目标: {self.gradient_target_ratio})")

            for loss_name, new_weight in updated_weights.items():
                old_weight = self.current_weights[loss_name]
                change = (new_weight - old_weight) / old_weight * 100
                print(f"  {loss_name}: {old_weight:.4f} → {new_weight:.4f} ({change:+.1f}%)")

            # 更新内部权重
            self.current_weights.update(updated_weights)

        return updated_weights if updated_weights else None

    def _should_update(self):
        """判断是否应该更新权重"""
        # 预热期不调整
        if self.current_step < self.warmup_steps:
            return False

        # 按频率更新
        if self.current_step % self.update_frequency == 0:
            # 确保有足够的历史数据
            if len(self.loss_history['data']) >= self.update_frequency // 2:
                return True

        return False

    def _compute_physics_weights(self, current_ratio):
        """
        计算 physics (div + ns) 的新权重
        """
        ratio_error = current_ratio / self.physics_target_ratio

        # 如果接近目标（±20%），不调整
        if 0.8 <= ratio_error <= 1.2:
            return None

        # 确定调整方向
        if ratio_error > 1.2:
            new_direction = -1  # 物理损失过大，降低权重
        else:
            new_direction = 1   # 物理损失过小，增加权重

        # 检查方向是否改变（震荡检测）
        if new_direction == self.physics_direction:
            self.physics_direction_count += 1
        else:
            self.physics_direction_count = 1
            self.physics_direction = new_direction

        # 只有连续相同方向超过patience次才调整
        if self.physics_direction_count < self.patience:
            return None

        # 计算调整幅度
        adjustment_factor = min(abs(ratio_error - 1.0), 0.5)
        adjustment_factor = max(adjustment_factor, self.adjustment_rate)

        updated_weights = {}
        physics_keys = ['divergence', 'navier_stokes']

        for key in physics_keys:
            if key in self.current_weights:
                old_weight = self.current_weights[key]

                if new_direction > 0:
                    new_weight = old_weight * (1 + adjustment_factor)
                else:
                    new_weight = old_weight * (1 - adjustment_factor)

                # 限制在范围内
                new_weight = np.clip(new_weight, self.min_weight, self.max_weight)

                # 只有变化超过5%才更新
                if abs(new_weight - old_weight) / (old_weight + 1e-8) > 0.05:
                    updated_weights[key] = new_weight

        return updated_weights if updated_weights else None

    def _compute_gradient_weight(self, current_ratio):
        """
        计算 gradient 的新权重
        """
        if 'gradient' not in self.current_weights:
            return None

        ratio_error = current_ratio / self.gradient_target_ratio

        # 如果接近目标（±30%），不调整 (gradient 容忍度稍大)
        if 0.7 <= ratio_error <= 1.3:
            return None

        # 确定调整方向
        if ratio_error > 1.3:
            new_direction = -1  # gradient 过大，降低权重
        else:
            new_direction = 1   # gradient 过小，增加权重

        # 检查方向是否改变
        if new_direction == self.gradient_direction:
            self.gradient_direction_count += 1
        else:
            self.gradient_direction_count = 1
            self.gradient_direction = new_direction

        # 只有连续相同方向超过patience次才调整
        if self.gradient_direction_count < self.patience:
            return None

        # 计算调整幅度 (gradient 调整更保守)
        adjustment_factor = min(abs(ratio_error - 1.0), 0.3)
        adjustment_factor = max(adjustment_factor, self.adjustment_rate * 0.5)

        old_weight = self.current_weights['gradient']

        if new_direction > 0:
            new_weight = old_weight * (1 + adjustment_factor)
        else:
            new_weight = old_weight * (1 - adjustment_factor)

        # 使用 gradient 专用的范围
        new_weight = np.clip(new_weight, self.gradient_min_weight, self.gradient_max_weight)

        # 只有变化超过5%才更新
        if abs(new_weight - old_weight) / (old_weight + 1e-8) > 0.05:
            return {'gradient': new_weight}

        return None

    def get_current_weights(self):
        """获取当前权重"""
        return self.current_weights.copy()

    def state_dict(self):
        """保存状态（用于checkpoint）"""
        return {
            'current_step': self.current_step,
            'current_weights': self.current_weights,
            'physics_direction': self.physics_direction,
            'physics_direction_count': self.physics_direction_count,
            'gradient_direction': self.gradient_direction,
            'gradient_direction_count': self.gradient_direction_count,
        }

    def load_state_dict(self, state_dict):
        """加载状态"""
        self.current_step = state_dict['current_step']
        self.current_weights = state_dict['current_weights']
        self.physics_direction = state_dict.get('physics_direction', 0)
        self.physics_direction_count = state_dict.get('physics_direction_count', 0)
        self.gradient_direction = state_dict.get('gradient_direction', 0)
        self.gradient_direction_count = state_dict.get('gradient_direction_count', 0)

        print(f"[AdaptiveWeightManager] 从checkpoint恢复:")
        print(f"  步数: {self.current_step}")
        print(f"  当前权重: {self.current_weights}")
