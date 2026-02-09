"""频域损失函数"""
import torch
import torch.nn as nn
import math


def fourier_loss(pred, target, **kwargs):
    """傅里叶频域损失"""
    pred_fft = torch.fft.rfft2(pred, dim=(-2, -1))
    target_fft = torch.fft.rfft2(target, dim=(-2, -1))

    pred_mag = torch.abs(pred_fft)
    target_mag = torch.abs(target_fft)

    return nn.functional.mse_loss(pred_mag, target_mag)


def energy_spectrum_loss(pred, target, **kwargs):
    """能量谱损失"""
    pred_fft = torch.fft.rfft2(pred, dim=(-2, -1))
    target_fft = torch.fft.rfft2(target, dim=(-2, -1))

    pred_energy = torch.abs(pred_fft) ** 2
    target_energy = torch.abs(target_fft) ** 2

    return nn.functional.mse_loss(pred_energy, target_energy)


class SpectralWeightedLoss(nn.Module):
    """
    频谱加权损失函数

    在频率域计算MSE，并对高频分量给予更大权重。
    基于Parseval定理：空间域MSE = 频率域MSE，
    但通过频率加权可以让优化器更关注高频分量。

    Args:
        high_freq_weight: 高频相对于低频的权重倍数 (默认2.0)
        weighting_mode: 权重计算方式
            - 'linear': 线性加权 w(k) = 1 + (hfw-1) * |k|/|k|_max
            - 'quadratic': 二次加权 w(k) = 1 + (hfw-1) * (|k|/|k|_max)^2
            - 'band': 分频带加权 (低/中/高频分别设置权重)
        adaptive: 是否动态调整权重 (基于当前误差分布)
        band_weights: 分频带权重 [low, mid, high]，仅在weighting_mode='band'时使用
        adaptive_config: 自适应权重配置字典，包含:
            - adaptation_rate: 权重调整速率 (默认0.1)
            - min_band_weight: 最小频带权重 (默认0.5)
            - max_band_weight: 最大频带权重 (默认100.0)
            - warmup_steps: 预热步数，在此之前不调整权重 (默认100)
            - target_ratio_mode: 目标权重计算方式 ('error_based' 或 'equalize')
        curriculum: 是否启用课程学习 (逐渐增加高频权重)
        curriculum_start_weight: 课程学习起始高频权重 (默认1.0，即无加权)
        curriculum_end_weight: 课程学习结束高频权重 (默认使用high_freq_weight)
        total_epochs: 总训练epoch数 (课程学习需要)
        warmup_ratio: 预热阶段比例 (默认0.2，前20% epochs保持起始权重)
        rampup_ratio: 增长阶段比例 (默认0.5，中间50% epochs线性增长)
    """

    def __init__(self, high_freq_weight=2.0, weighting_mode='linear',
                 adaptive=False, band_weights=None, adaptive_config=None,
                 curriculum=False, curriculum_start_weight=1.0,
                 curriculum_end_weight=None, total_epochs=50,
                 warmup_ratio=0.2, rampup_ratio=0.5):
        super().__init__()
        self.high_freq_weight = high_freq_weight
        self.weighting_mode = weighting_mode
        self.adaptive = adaptive
        # 使用list存储以便修改
        self.band_weights = list(band_weights) if band_weights else [1.0, 1.5, 2.0]  # low, mid, high
        self._initial_band_weights = list(self.band_weights)  # 保存初始权重

        # 自适应权重配置
        self.adaptive_config = adaptive_config or {}
        self.adaptation_rate = self.adaptive_config.get('adaptation_rate', 0.1)
        self.min_band_weight = self.adaptive_config.get('min_band_weight', 0.5)
        self.max_band_weight = self.adaptive_config.get('max_band_weight', 100.0)
        self.adaptive_warmup_steps = self.adaptive_config.get('warmup_steps', 100)
        self.target_ratio_mode = self.adaptive_config.get('target_ratio_mode', 'error_based')
        self._adaptive_step_count = 0

        # 课程学习参数
        self.curriculum = curriculum
        self.curriculum_start_weight = curriculum_start_weight
        self.curriculum_end_weight = curriculum_end_weight if curriculum_end_weight is not None else high_freq_weight
        self.total_epochs = total_epochs
        self.warmup_ratio = warmup_ratio
        self.rampup_ratio = rampup_ratio
        self._current_epoch = 0
        self._current_high_freq_weight = curriculum_start_weight if curriculum else high_freq_weight

        # 缓存频率网格 (按shape缓存)
        self._freq_cache = {}
        self._weight_cache = {}

        # 自适应权重的EMA统计
        self.register_buffer('error_ema_low', torch.tensor(0.0))
        self.register_buffer('error_ema_mid', torch.tensor(0.0))
        self.register_buffer('error_ema_high', torch.tensor(0.0))
        self.ema_momentum = 0.99
        self._ema_initialized = False

    def set_epoch(self, epoch):
        """
        设置当前epoch，用于课程学习

        课程学习Schedule:
        - [0, warmup_end): 保持 curriculum_start_weight (专注低频)
        - [warmup_end, rampup_end): 线性增长到 curriculum_end_weight
        - [rampup_end, total_epochs): 保持 curriculum_end_weight (强调高频)
        """
        self._current_epoch = epoch

        if not self.curriculum:
            return

        warmup_end = int(self.total_epochs * self.warmup_ratio)
        rampup_end = int(self.total_epochs * (self.warmup_ratio + self.rampup_ratio))

        if epoch < warmup_end:
            # 预热阶段：保持起始权重
            self._current_high_freq_weight = self.curriculum_start_weight
        elif epoch < rampup_end:
            # 增长阶段：线性插值
            progress = (epoch - warmup_end) / max(1, rampup_end - warmup_end)
            self._current_high_freq_weight = (
                self.curriculum_start_weight +
                (self.curriculum_end_weight - self.curriculum_start_weight) * progress
            )
        else:
            # 稳定阶段：保持结束权重
            self._current_high_freq_weight = self.curriculum_end_weight

        # 清除权重缓存，因为高频权重已改变
        self._weight_cache = {}

    def get_current_high_freq_weight(self):
        """获取当前高频权重 (用于监控)"""
        return self._current_high_freq_weight

    def _get_frequency_grid(self, shape, device):
        """获取或创建频率网格"""
        cache_key = (shape, device)
        if cache_key not in self._freq_cache:
            # shape: (..., Z, H, W) for rfftn
            Z, H, W = shape[-3], shape[-2], shape[-1]

            # 频率坐标
            kz = torch.fft.fftfreq(Z, device=device)
            ky = torch.fft.fftfreq(H, device=device)
            kx = torch.fft.rfftfreq(W, device=device)  # rfft: W//2+1

            # 创建网格
            KZ, KY, KX = torch.meshgrid(kz, ky, kx, indexing='ij')
            freq_mag = torch.sqrt(KX**2 + KY**2 + KZ**2)

            self._freq_cache[cache_key] = freq_mag

        return self._freq_cache[cache_key]

    def _get_weights(self, freq_mag, shape, device):
        """计算频率权重 (使用当前高频权重，支持课程学习)"""
        # 使用当前高频权重作为cache key的一部分
        hfw = self._current_high_freq_weight
        cache_key = (shape, device, self.weighting_mode, hfw)

        if cache_key not in self._weight_cache:
            freq_max = freq_mag.max()
            normalized_freq = freq_mag / (freq_max + 1e-8)

            if self.weighting_mode == 'linear':
                # 线性加权: w = 1 + (hfw-1) * |k|/|k|_max
                weights = 1.0 + (hfw - 1.0) * normalized_freq

            elif self.weighting_mode == 'quadratic':
                # 二次加权: 高频权重增长更快
                weights = 1.0 + (hfw - 1.0) * (normalized_freq ** 2)

            elif self.weighting_mode == 'band':
                # 分频带加权
                weights = torch.ones_like(freq_mag)
                low_mask = normalized_freq < 1/3
                mid_mask = (normalized_freq >= 1/3) & (normalized_freq < 2/3)
                high_mask = normalized_freq >= 2/3

                weights[low_mask] = self.band_weights[0]
                weights[mid_mask] = self.band_weights[1]
                weights[high_mask] = self.band_weights[2]
            else:
                weights = torch.ones_like(freq_mag)

            self._weight_cache[cache_key] = weights

        return self._weight_cache[cache_key]

    def _update_adaptive_weights(self, spectral_error, freq_mag):
        """
        更新自适应权重 (基于当前误差分布)

        自适应策略: 误差更高的频带获得更大的权重，促使模型更关注难学习的频段。
        权重调整公式: new_weight = old_weight + adaptation_rate * (target_weight - old_weight)
        其中 target_weight 基于各频带误差比例计算。
        """
        if not self.training or not self.adaptive:
            return

        freq_max = freq_mag.max()
        normalized_freq = freq_mag / (freq_max + 1e-8)

        # 计算各频带的平均误差
        low_mask = normalized_freq < 1/3
        mid_mask = (normalized_freq >= 1/3) & (normalized_freq < 2/3)
        high_mask = normalized_freq >= 2/3

        error_low = spectral_error[low_mask].mean() if low_mask.any() else torch.tensor(0.0, device=spectral_error.device)
        error_mid = spectral_error[mid_mask].mean() if mid_mask.any() else torch.tensor(0.0, device=spectral_error.device)
        error_high = spectral_error[high_mask].mean() if high_mask.any() else torch.tensor(0.0, device=spectral_error.device)

        # EMA更新
        if not self._ema_initialized:
            # 第一次直接赋值
            self.error_ema_low = error_low.detach()
            self.error_ema_mid = error_mid.detach()
            self.error_ema_high = error_high.detach()
            self._ema_initialized = True
        else:
            self.error_ema_low = self.ema_momentum * self.error_ema_low + (1 - self.ema_momentum) * error_low.detach()
            self.error_ema_mid = self.ema_momentum * self.error_ema_mid + (1 - self.ema_momentum) * error_mid.detach()
            self.error_ema_high = self.ema_momentum * self.error_ema_high + (1 - self.ema_momentum) * error_high.detach()

        # 增加步数计数
        self._adaptive_step_count += 1

        # 预热期间不调整权重
        if self._adaptive_step_count < self.adaptive_warmup_steps:
            return

        # ===== 真正的自适应权重调整 =====
        errors = [self.error_ema_low.item(), self.error_ema_mid.item(), self.error_ema_high.item()]
        total_error = sum(errors) + 1e-8

        # 计算目标权重
        target_total_weight = sum(self._initial_band_weights)  # 8.0 for [1,2,5]

        if self.target_ratio_mode == 'error_based':
            # 相对误差越大的频带获得更大权重
            # 使用 sqrt(error) 来避免极端权重分配
            # 例如: errors = [0.01, 0.001, 0.0001]
            #       sqrt_errors = [0.1, 0.0316, 0.01]
            #       比直接用 errors 更平滑
            sqrt_errors = [max(e, 1e-10) ** 0.5 for e in errors]
            sqrt_total = sum(sqrt_errors) + 1e-8
            error_ratios = [se / sqrt_total for se in sqrt_errors]
            # 目标权重: 保持总权重大致不变，但重新分配
            target_weights = [ratio * target_total_weight for ratio in error_ratios]

        elif self.target_ratio_mode == 'boost_high':
            # 专门增强高频权重
            # 高频误差虽然绝对值小，但相对于其能量占比更大
            # 使用指数衰减：低频权重低，高频权重高
            base_weights = [1.0, 2.0, 4.0]  # 基础比例 1:2:4
            # 根据误差调整：误差越大的频带额外增加
            error_ratios = [e / total_error for e in errors]
            # 高频额外 boost (index 2 获得更多)
            boosted = [base_weights[i] * (1.0 + error_ratios[i]) for i in range(3)]
            boosted_total = sum(boosted)
            target_weights = [(b / boosted_total) * target_total_weight for b in boosted]

        else:  # 'equalize' - 尝试均衡各频带的加权误差
            # 如果要让 w_i * e_i 相等，则 w_i 应与 1/e_i 成正比
            # 这会让误差小的频带获得更大权重
            inv_errors = [1.0 / (e + 1e-8) for e in errors]
            inv_total = sum(inv_errors)
            target_weights = [(inv_e / inv_total) * target_total_weight for inv_e in inv_errors]

        # 渐进式调整权重
        old_weights = list(self.band_weights)
        weights_changed = False

        for i in range(3):
            target_w = max(self.min_band_weight, min(self.max_band_weight, target_weights[i]))
            new_w = self.band_weights[i] + self.adaptation_rate * (target_w - self.band_weights[i])
            new_w = max(self.min_band_weight, min(self.max_band_weight, new_w))

            if abs(new_w - self.band_weights[i]) > 1e-6:
                self.band_weights[i] = new_w
                weights_changed = True

        # 如果权重改变，清除缓存
        if weights_changed:
            self._weight_cache = {}

    def forward(self, pred, target, return_details=False, **kwargs):
        """
        计算频谱加权损失

        Args:
            pred: (B, T, C, Z, H, W) 预测
            target: (B, T, C, Z, H, W) 真值
            return_details: 是否返回详细的频带损失

        Returns:
            loss: 标量损失值
            或 (loss, details_dict) 如果 return_details=True
        """
        # 确保输入是4D以上 (至少有Z, H, W)
        orig_shape = pred.shape
        if pred.dim() == 6:  # (B, T, C, Z, H, W)
            B, T, C, Z, H, W = pred.shape
            pred = pred.reshape(B * T * C, Z, H, W)
            target = target.reshape(B * T * C, Z, H, W)
        elif pred.dim() == 5:  # (B, C, Z, H, W)
            B, C, Z, H, W = pred.shape
            pred = pred.reshape(B * C, Z, H, W)
            target = target.reshape(B * C, Z, H, W)
        elif pred.dim() == 4:  # (B, Z, H, W)
            pass
        else:
            raise ValueError(f"Expected 4-6D input, got {pred.dim()}D")

        # 空间域元素数量 (用于归一化，使loss与MSE可比)
        spatial_size = pred.shape[-3] * pred.shape[-2] * pred.shape[-1]

        # 3D FFT (使用 ortho 归一化使 Parseval 定理精确成立)
        pred_fft = torch.fft.rfftn(pred, dim=(-3, -2, -1), norm='ortho')
        target_fft = torch.fft.rfftn(target, dim=(-3, -2, -1), norm='ortho')

        # 频谱误差 (复数差的模的平方)
        # 使用 ortho norm 后，sum(|FFT|²) = sum(|signal|²)
        spectral_error = torch.abs(pred_fft - target_fft) ** 2

        # 获取频率网格和权重
        freq_mag = self._get_frequency_grid(pred.shape, pred.device)
        weights = self._get_weights(freq_mag, pred.shape, pred.device)

        # 自适应权重更新
        if self.adaptive:
            self._update_adaptive_weights(spectral_error.mean(dim=0), freq_mag)

        # 加权平均
        # 注: rfft 只输出非冗余频率 (实信号的共轭对称性)
        # 这不影响 loss 的有效性: pred 和 target 使用相同的 FFT，相对误差正确
        # 频率权重基于 |k|，对称频率 -k 和 k 权重相同，加权也正确
        weighted_error = weights * spectral_error
        loss = weighted_error.mean()

        # 保存最近一次的频带误差 (用于外部查询)
        freq_max = freq_mag.max()
        normalized_freq = freq_mag / (freq_max + 1e-8)

        low_mask = normalized_freq < 1/3
        mid_mask = (normalized_freq >= 1/3) & (normalized_freq < 2/3)
        high_mask = normalized_freq >= 2/3

        # 计算各频带的未加权误差 (用于监控)
        self._last_band_errors = {
            'low': spectral_error[:, low_mask].mean().item() if low_mask.any() else 0.0,
            'mid': spectral_error[:, mid_mask].mean().item() if mid_mask.any() else 0.0,
            'high': spectral_error[:, high_mask].mean().item() if high_mask.any() else 0.0,
        }

        if return_details:
            return loss, self._last_band_errors
        return loss

    def get_last_band_errors(self):
        """获取最近一次前向传播的频带误差"""
        return getattr(self, '_last_band_errors', {'low': 0.0, 'mid': 0.0, 'high': 0.0})

    def get_current_band_errors(self):
        """获取当前各频带的EMA误差 (用于监控)"""
        return {
            'low': self.error_ema_low.item(),
            'mid': self.error_ema_mid.item(),
            'high': self.error_ema_high.item()
        }

    def get_current_band_weights(self):
        """获取当前各频带的权重 (用于监控自适应权重变化)"""
        return {
            'low': self.band_weights[0],
            'mid': self.band_weights[1],
            'high': self.band_weights[2]
        }

    def get_adaptive_info(self):
        """获取自适应权重的完整信息 (用于日志)"""
        return {
            'band_weights': {
                'low': self.band_weights[0],
                'mid': self.band_weights[1],
                'high': self.band_weights[2]
            },
            'initial_weights': {
                'low': self._initial_band_weights[0],
                'mid': self._initial_band_weights[1],
                'high': self._initial_band_weights[2]
            },
            'error_ema': {
                'low': self.error_ema_low.item(),
                'mid': self.error_ema_mid.item(),
                'high': self.error_ema_high.item()
            },
            'adaptive_step': self._adaptive_step_count,
            'warmup_complete': self._adaptive_step_count >= self.adaptive_warmup_steps
        }


class RelativeSpectralLoss(nn.Module):
    """
    相对频谱损失函数

    与 SpectralWeightedLoss 不同，这里计算的是相对于GT能量的误差比例，
    自动适应不同样本的频谱特性（从稳定态到湍流态）。

    对于每个频段:
        relative_error = |FFT(pred - gt)|² / (|FFT(gt)|² + eps)

    这样不管是低频主导还是高频主导的样本，每个频段的贡献都是相对的。

    Args:
        band_boundaries: 频段边界列表，如 [0.03, 0.08] 表示 [0-3%, 3-8%, 8-100%]
        band_weights: 各频段权重，如 [1.0, 1.0, 1.0] 表示等权重
        eps: 防止除零的小常数
    """

    def __init__(self, band_boundaries=None, band_weights=None, eps=1e-8,
                 max_relative_error=10.0, min_energy_ratio=0.001):
        super().__init__()
        # 默认使用物理尺度划分: 大尺度(0-3%), 惯性区(3-8%), 耗散区(8-100%)
        self.band_boundaries = band_boundaries if band_boundaries else [0.03, 0.08]
        self.band_weights = band_weights if band_weights else [1.0, 1.0, 1.0]
        self.eps = eps
        self.max_relative_error = max_relative_error  # 相对误差上限，防止爆炸
        self.min_energy_ratio = min_energy_ratio  # 最小能量比例，低于此的频段使用绝对误差

        # 缓存
        self._freq_cache = {}
        self._last_band_errors = {}
        self._last_band_energies = {}

    def _get_frequency_grid(self, shape, device):
        """获取归一化的频率网格"""
        cache_key = (shape, device)
        if cache_key not in self._freq_cache:
            Z, H, W = shape[-3], shape[-2], shape[-1]

            kz = torch.fft.fftfreq(Z, device=device)
            ky = torch.fft.fftfreq(H, device=device)
            kx = torch.fft.rfftfreq(W, device=device)

            KZ, KY, KX = torch.meshgrid(kz, ky, kx, indexing='ij')
            freq_mag = torch.sqrt(KX**2 + KY**2 + KZ**2)

            # 归一化到 [0, 1]
            freq_max = freq_mag.max()
            normalized_freq = freq_mag / (freq_max + 1e-8)

            self._freq_cache[cache_key] = normalized_freq

        return self._freq_cache[cache_key]

    def _get_band_masks(self, normalized_freq):
        """获取各频段的mask"""
        masks = []
        boundaries = [0.0] + self.band_boundaries + [1.0]

        for i in range(len(boundaries) - 1):
            low = boundaries[i]
            high = boundaries[i + 1]
            mask = (normalized_freq >= low) & (normalized_freq < high)
            masks.append(mask)

        return masks

    def forward(self, pred, target, return_details=False, **kwargs):
        """
        计算相对频谱损失

        Args:
            pred: (B, T, C, Z, H, W) 或更低维度
            target: 与pred相同shape
            return_details: 是否返回详细信息

        Returns:
            loss: 标量损失
        """
        # 处理输入维度
        if pred.dim() == 6:  # (B, T, C, Z, H, W)
            B, T, C, Z, H, W = pred.shape
            pred = pred.reshape(B * T * C, Z, H, W)
            target = target.reshape(B * T * C, Z, H, W)
        elif pred.dim() == 5:  # (B, C, Z, H, W)
            B, C, Z, H, W = pred.shape
            pred = pred.reshape(B * C, Z, H, W)
            target = target.reshape(B * C, Z, H, W)
        elif pred.dim() == 4:  # (B, Z, H, W)
            pass
        else:
            raise ValueError(f"Expected 4-6D input, got {pred.dim()}D")

        # 3D FFT (ortho归一化保证Parseval定理)
        pred_fft = torch.fft.rfftn(pred, dim=(-3, -2, -1), norm='ortho')
        target_fft = torch.fft.rfftn(target, dim=(-3, -2, -1), norm='ortho')

        # 误差频谱和GT能量频谱
        error_spectrum = torch.abs(pred_fft - target_fft) ** 2  # |FFT(pred - gt)|²
        gt_spectrum = torch.abs(target_fft) ** 2  # |FFT(gt)|²

        # 获取频率网格和频段mask
        normalized_freq = self._get_frequency_grid(pred.shape, pred.device)
        band_masks = self._get_band_masks(normalized_freq)

        # 计算各频段的相对误差
        total_loss = 0.0
        band_names = ['low', 'mid', 'high'] if len(band_masks) == 3 else [f'band_{i}' for i in range(len(band_masks))]

        self._last_band_errors = {}
        self._last_band_energies = {}

        # 计算总GT能量 (用于判断各频段的能量比例)
        total_gt_energy = gt_spectrum.sum(dim=-1).mean() + self.eps

        for i, (mask, weight, name) in enumerate(zip(band_masks, self.band_weights, band_names)):
            if not mask.any():
                continue

            # 该频段的误差和GT能量 (对batch求平均)
            band_error = error_spectrum[:, mask].sum(dim=-1).mean()  # 先sum频率，再mean batch
            band_gt_energy = gt_spectrum[:, mask].sum(dim=-1).mean()

            # 计算该频段的能量比例
            energy_ratio = band_gt_energy / total_gt_energy

            # 如果该频段能量占比太小，使用绝对误差而非相对误差（防止爆炸）
            if energy_ratio < self.min_energy_ratio:
                # 使用绝对误差，但归一化到与其他频段可比的尺度
                relative_error = band_error / (total_gt_energy * self.min_energy_ratio)
            else:
                # 正常的相对误差
                relative_error = band_error / (band_gt_energy + self.eps)

            # 限制相对误差的最大值（额外保护）
            relative_error = torch.clamp(relative_error, max=self.max_relative_error)

            # 加权累加
            total_loss = total_loss + weight * relative_error

            # 保存用于监控
            self._last_band_errors[name] = band_error.item()
            self._last_band_energies[name] = band_gt_energy.item()

        # 归一化 (除以权重总和)
        total_loss = total_loss / sum(self.band_weights)

        if return_details:
            details = {
                'band_errors': self._last_band_errors,
                'band_energies': self._last_band_energies,
                'relative_errors': {
                    k: self._last_band_errors[k] / (self._last_band_energies[k] + self.eps)
                    for k in self._last_band_errors
                }
            }
            return total_loss, details

        return total_loss

    def get_last_band_info(self):
        """获取最近一次的频段信息"""
        return {
            'errors': self._last_band_errors,
            'energies': self._last_band_energies,
            'relative': {
                k: self._last_band_errors.get(k, 0) / (self._last_band_energies.get(k, 1e-8) + self.eps)
                for k in self._last_band_errors
            }
        }


def spectral_weighted_loss(pred, target, high_freq_weight=2.0, weighting_mode='linear', **kwargs):
    """
    函数式接口的频谱加权损失

    Args:
        pred: 预测张量
        target: 真值张量
        high_freq_weight: 高频权重倍数
        weighting_mode: 权重模式 ('linear', 'quadratic', 'band')
    """
    loss_fn = SpectralWeightedLoss(
        high_freq_weight=high_freq_weight,
        weighting_mode=weighting_mode,
        adaptive=False
    )
    return loss_fn(pred, target)


def get_spectral_loss(loss_type, config_dict):
    """获取频谱损失函数"""
    if loss_type == 'fourier':
        return lambda pred, target, **kwargs: fourier_loss(pred, target, **{**config_dict, **kwargs})

    elif loss_type == 'energy_spectrum':
        return lambda pred, target, **kwargs: energy_spectrum_loss(pred, target, **{**config_dict, **kwargs})

    elif loss_type == 'spectral_relative':
        # 相对频谱损失 - 自动适应不同样本的频谱特性
        loss_instance = RelativeSpectralLoss(
            band_boundaries=config_dict.get('band_boundaries', [0.03, 0.08]),
            band_weights=config_dict.get('band_weights', [1.0, 1.0, 1.0]),
            eps=config_dict.get('eps', 1e-8),
            max_relative_error=config_dict.get('max_relative_error', 10.0),  # 防止爆炸
            min_energy_ratio=config_dict.get('min_energy_ratio', 0.001)  # 能量占比<0.1%时使用绝对误差
        )
        return loss_instance

    elif loss_type == 'spectral_weighted':
        # 创建SpectralWeightedLoss实例 (支持课程学习和自适应权重)
        loss_instance = SpectralWeightedLoss(
            high_freq_weight=config_dict.get('high_freq_weight', 2.0),
            weighting_mode=config_dict.get('weighting_mode', 'linear'),
            adaptive=config_dict.get('adaptive', False),
            band_weights=config_dict.get('band_weights', None),
            adaptive_config=config_dict.get('adaptive_config', None),  # 自适应权重配置
            # 课程学习参数
            curriculum=config_dict.get('curriculum', False),
            curriculum_start_weight=config_dict.get('curriculum_start_weight', 1.0),
            curriculum_end_weight=config_dict.get('curriculum_end_weight', None),
            total_epochs=config_dict.get('total_epochs', 50),
            warmup_ratio=config_dict.get('warmup_ratio', 0.2),
            rampup_ratio=config_dict.get('rampup_ratio', 0.5)
        )
        return loss_instance

    else:
        raise ValueError(f"Unknown spectral loss: {loss_type}")
