# Basic imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import json
import csv
import math
from einops import rearrange

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

"""Configuration Classes"""

class SwinPatchedConfigFixed:
    """Base Configuration - Complete training config compatible with all components"""

    CHECKPOINT_DIR = 'checkpoints_swin'
    EXPERIMENT_NAME = 'swin_baseline'

    IN_CHANNELS = 4  # u, v, w, p
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
        'data': {'type': 'mse', 'weight': 1.0, 'target': 'missing'},
        'divergence': {'type': 'divergence_3d', 'weight': 50.0, 'dx': 1.0, 'dy': 1.0, 'dz': 1.0},
        'navier_stokes': {'type': 'navier_stokes', 'weight': 5.0, 'nu': 0.01, 'rho': 1.0,
                         'dx': 1.0, 'dy': 1.0, 'dz': 1.0, 'dt': 0.1}
    }

    PHYSICS_ON_MISSING_ONLY = False

    ADAPTIVE_WEIGHT_CONFIG = {
        'enabled': False, 'update_frequency': 100, 'warmup_steps': 500,
        'min_weight': 0.5, 'max_weight': 20.0, 'target_ratio': 0.5,
        'adjustment_rate': 0.1, 'patience': 3
    }

    COMPLETION_METHOD = 'interpolation'
    FOURIER_CONFIG = {'d_model': 256, 'freq_size': 32, 'n_heads': 8, 'n_layers': 4, 'dropout': 0.1}
    INTERPOLATION_CONFIG = {'use_refinement': True}
    FOURIER_INTERMEDIATE_WEIGHT = 0.5

    @property
    def memory_reduction_factor(self):
        original_size = self.INPUT_RESOLUTION[0] * self.INPUT_RESOLUTION[1] * self.INPUT_RESOLUTION[2]
        patched_size = self.PATCHED_RESOLUTION[0] * self.PATCHED_RESOLUTION[1] * self.PATCHED_RESOLUTION[2]
        return original_size / patched_size


class SwinJHUConfigSpectral(SwinPatchedConfigFixed):
    """JHU DNS128 Spectral Weighted Loss Configuration - Best performing config"""

    CHECKPOINT_DIR = 'checkpoints_swin_jhu_spectral'
    EXPERIMENT_NAME = 'swin_jhu_spectral'

    IN_CHANNELS = 4
    INPUT_RESOLUTION = (128, 128, 128)
    NUM_SPARSE_Z_LAYERS = 16

    PATCH_SIZE = (4, 4, 4)
    PATCHED_RESOLUTION = (32, 32, 32)
    WINDOW_SIZE = (8, 8, 8)

    _DX = 0.0490873852  # 2π/128
    _NU = 0.000185
    _NU_GRID = _NU / (_DX ** 2)  # ≈ 0.077

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
            'dx': 1.0, 'dy': 1.0, 'dz': 1.0
        },
        'navier_stokes': {
            'type': 'navier_stokes',
            'weight': 0.1,
            'nu': 0.077,  # _NU_GRID
            'rho': 1.0,
            'dx': 1.0, 'dy': 1.0, 'dz': 1.0, 'dt': 1.0
        },
        'gradient': {
            'type': 'gradient_3d',
            'weight': 0.1
        }
    }

    ADAPTIVE_WEIGHT_CONFIG = {
        'enabled': True, 'update_frequency': 50, 'warmup_steps': 200,
        'min_weight': 0.01, 'max_weight': 10.0, 'target_ratio': 0.5,
        'adjustment_rate': 0.1, 'patience': 2
    }


# JHU Normalization Statistics
JHU_NORM_STATS = {
    "mean": [0.0008644619867776491, 0.0012105473293069805, 0.0021110948628738713, 0.00037188583444958443],
    "std": [0.6385110885344759, 0.7478282257339498, 0.6766117184108378, 0.41183318615742287],
    "indicators": ["u", "v", "w", "p"]
}

print("Configurations loaded.")

"""Data Loss Functions"""

def mse_loss(pred, target, missing_indices=None, **kwargs):
    if missing_indices is not None:
        return mse_loss_masked(pred, target, missing_indices)
    else:
        return nn.functional.mse_loss(pred, target)

def mse_loss_masked(pred, target, missing_indices):
    B = pred.shape[0]
    losses = []
    for b in range(B):
        missing_z = missing_indices[b]
        pred_missing = pred[b, :, :, missing_z, :, :]
        target_missing = target[b, :, :, missing_z, :, :]
        loss_b = nn.functional.mse_loss(pred_missing, target_missing)
        losses.append(loss_b)
    return torch.stack(losses).mean()

def l1_loss(pred, target, missing_indices=None, **kwargs):
    if missing_indices is not None:
        B = pred.shape[0]
        losses = []
        for b in range(B):
            missing_z = missing_indices[b]
            pred_missing = pred[b, :, :, missing_z, :, :]
            target_missing = target[b, :, :, missing_z, :, :]
            loss_b = nn.functional.l1_loss(pred_missing, target_missing)
            losses.append(loss_b)
        return torch.stack(losses).mean()
    else:
        return nn.functional.l1_loss(pred, target)

def huber_loss(pred, target, missing_indices=None, delta=1.0, **kwargs):
    if missing_indices is not None:
        B = pred.shape[0]
        losses = []
        for b in range(B):
            missing_z = missing_indices[b]
            pred_missing = pred[b, :, :, missing_z, :, :]
            target_missing = target[b, :, :, missing_z, :, :]
            loss_b = nn.functional.smooth_l1_loss(pred_missing, target_missing, beta=delta)
            losses.append(loss_b)
        return torch.stack(losses).mean()
    else:
        return nn.functional.smooth_l1_loss(pred, target, beta=delta)

def relative_l2_loss(pred, target, missing_indices=None, eps=1e-8, **kwargs):
    if missing_indices is not None:
        B = pred.shape[0]
        losses = []
        for b in range(B):
            missing_z = missing_indices[b]
            pred_missing = pred[b, :, :, missing_z, :, :]
            target_missing = target[b, :, :, missing_z, :, :]
            diff = pred_missing - target_missing
            rel_error = torch.norm(diff.reshape(-1)) / (torch.norm(target_missing.reshape(-1)) + eps)
            losses.append(rel_error)
        return torch.stack(losses).mean()
    else:
        diff = pred - target
        rel_error = torch.norm(diff.reshape(diff.shape[0], -1), dim=1) / \
                    (torch.norm(target.reshape(target.shape[0], -1), dim=1) + eps)
        return rel_error.mean()

def get_data_loss(loss_type='mse'):
    losses = {'mse': mse_loss, 'l1': l1_loss, 'huber': huber_loss, 'relative_l2': relative_l2_loss}
    if loss_type not in losses:
        raise ValueError(f"Unknown data loss type: {loss_type}")
    return losses[loss_type]

print("Data loss functions loaded.")

"""Spectral Loss Functions"""

def fourier_loss(pred, target, **kwargs):
    pred_fft = torch.fft.rfft2(pred, dim=(-2, -1))
    target_fft = torch.fft.rfft2(target, dim=(-2, -1))
    pred_mag = torch.abs(pred_fft)
    target_mag = torch.abs(target_fft)
    return nn.functional.mse_loss(pred_mag, target_mag)

def energy_spectrum_loss(pred, target, **kwargs):
    pred_fft = torch.fft.rfft2(pred, dim=(-2, -1))
    target_fft = torch.fft.rfft2(target, dim=(-2, -1))
    pred_energy = torch.abs(pred_fft) ** 2
    target_energy = torch.abs(target_fft) ** 2
    return nn.functional.mse_loss(pred_energy, target_energy)


class SpectralWeightedLoss(nn.Module):
    """
    Spectral Weighted Loss Function
    
    Computes MSE in frequency domain with higher weights for high frequency components.
    Based on Parseval's theorem: spatial MSE = frequency MSE.
    """

    def __init__(self, high_freq_weight=2.0, weighting_mode='linear',
                 adaptive=False, band_weights=None):
        super().__init__()
        self.high_freq_weight = high_freq_weight
        self.weighting_mode = weighting_mode
        self.adaptive = adaptive
        self.band_weights = band_weights or [1.0, 1.5, 2.0]
        self._freq_cache = {}
        self._weight_cache = {}
        self.register_buffer('error_ema_low', torch.tensor(0.0))
        self.register_buffer('error_ema_mid', torch.tensor(0.0))
        self.register_buffer('error_ema_high', torch.tensor(0.0))
        self.ema_momentum = 0.99

    def _get_frequency_grid(self, shape, device):
        cache_key = (shape, device)
        if cache_key not in self._freq_cache:
            Z, H, W = shape[-3], shape[-2], shape[-1]
            kz = torch.fft.fftfreq(Z, device=device)
            ky = torch.fft.fftfreq(H, device=device)
            kx = torch.fft.rfftfreq(W, device=device)
            KZ, KY, KX = torch.meshgrid(kz, ky, kx, indexing='ij')
            freq_mag = torch.sqrt(KX**2 + KY**2 + KZ**2)
            self._freq_cache[cache_key] = freq_mag
        return self._freq_cache[cache_key]

    def _get_weights(self, freq_mag, shape, device):
        cache_key = (shape, device, self.weighting_mode, self.high_freq_weight)
        if cache_key not in self._weight_cache:
            freq_max = freq_mag.max()
            normalized_freq = freq_mag / (freq_max + 1e-8)
            if self.weighting_mode == 'linear':
                weights = 1.0 + (self.high_freq_weight - 1.0) * normalized_freq
            elif self.weighting_mode == 'quadratic':
                weights = 1.0 + (self.high_freq_weight - 1.0) * (normalized_freq ** 2)
            elif self.weighting_mode == 'band':
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

    def forward(self, pred, target, return_details=False, **kwargs):
        orig_shape = pred.shape
        if pred.dim() == 6:
            B, T, C, Z, H, W = pred.shape
            pred = pred.reshape(B * T * C, Z, H, W)
            target = target.reshape(B * T * C, Z, H, W)
        elif pred.dim() == 5:
            B, C, Z, H, W = pred.shape
            pred = pred.reshape(B * C, Z, H, W)
            target = target.reshape(B * C, Z, H, W)
        elif pred.dim() == 4:
            pass
        else:
            raise ValueError(f"Expected 4-6D input, got {pred.dim()}D")

        pred_fft = torch.fft.rfftn(pred, dim=(-3, -2, -1), norm='ortho')
        target_fft = torch.fft.rfftn(target, dim=(-3, -2, -1), norm='ortho')
        spectral_error = torch.abs(pred_fft - target_fft) ** 2

        freq_mag = self._get_frequency_grid(pred.shape, pred.device)
        weights = self._get_weights(freq_mag, pred.shape, pred.device)
        weighted_error = weights * spectral_error
        loss = weighted_error.mean()

        freq_max = freq_mag.max()
        normalized_freq = freq_mag / (freq_max + 1e-8)
        low_mask = normalized_freq < 1/3
        mid_mask = (normalized_freq >= 1/3) & (normalized_freq < 2/3)
        high_mask = normalized_freq >= 2/3

        self._last_band_errors = {
            'low': spectral_error[:, low_mask].mean().item() if low_mask.any() else 0.0,
            'mid': spectral_error[:, mid_mask].mean().item() if mid_mask.any() else 0.0,
            'high': spectral_error[:, high_mask].mean().item() if high_mask.any() else 0.0,
        }

        if return_details:
            return loss, self._last_band_errors
        return loss

    def get_last_band_errors(self):
        return getattr(self, '_last_band_errors', {'low': 0.0, 'mid': 0.0, 'high': 0.0})


def get_spectral_loss(loss_type, config_dict):
    if loss_type == 'fourier':
        return lambda pred, target, **kwargs: fourier_loss(pred, target, **{**config_dict, **kwargs})
    elif loss_type == 'energy_spectrum':
        return lambda pred, target, **kwargs: energy_spectrum_loss(pred, target, **{**config_dict, **kwargs})
    elif loss_type == 'spectral_weighted':
        return SpectralWeightedLoss(
            high_freq_weight=config_dict.get('high_freq_weight', 2.0),
            weighting_mode=config_dict.get('weighting_mode', 'linear'),
            adaptive=config_dict.get('adaptive', False),
            band_weights=config_dict.get('band_weights', None)
        )
    else:
        raise ValueError(f"Unknown spectral loss: {loss_type}")

print("Spectral loss functions loaded.")

"""Physics Loss Functions"""

def denormalize_field(pred, norm_mean=None, norm_std=None):
    """Denormalize physics field"""
    if norm_mean is None or norm_std is None:
        return pred
    pred_real = pred.clone()
    device = pred.device
    if not isinstance(norm_mean, torch.Tensor):
        norm_mean = torch.tensor(norm_mean, device=device)
    if not isinstance(norm_std, torch.Tensor):
        norm_std = torch.tensor(norm_std, device=device)
    norm_mean = norm_mean.to(device)
    norm_std = norm_std.to(device)
    for c_idx in range(pred.shape[2]):
        pred_real[:, :, c_idx] = pred[:, :, c_idx] * norm_std[c_idx] + norm_mean[c_idx]
    return pred_real

def divergence_loss_3d(pred, target=None, dx=1.0, dy=1.0, dz=1.0, norm_mean=None, norm_std=None, **kwargs):
    """3D Divergence Loss"""
    n_channels = pred.shape[2]
    pred_real = denormalize_field(pred, norm_mean, norm_std)
    u = pred_real[:, :, 0, :, :, :]
    v = pred_real[:, :, 1, :, :, :]
    du_dx = (u[:, :, :, :, 2:] - u[:, :, :, :, :-2]) / (2 * dx)
    dv_dy = (v[:, :, :, 2:, :] - v[:, :, :, :-2, :]) / (2 * dy)
    if n_channels >= 4:
        w = pred_real[:, :, 2, :, :, :]
        dw_dz = (w[:, :, 2:, :, :] - w[:, :, :-2, :, :]) / (2 * dz)
        min_z = min(du_dx.shape[2], dv_dy.shape[2], dw_dz.shape[2])
        min_h = min(du_dx.shape[3], dv_dy.shape[3], dw_dz.shape[3])
        min_w = min(du_dx.shape[4], dv_dy.shape[4], dw_dz.shape[4])
        du_dx = du_dx[:, :, :min_z, :min_h, :min_w]
        dv_dy = dv_dy[:, :, :min_z, :min_h, :min_w]
        dw_dz = dw_dz[:, :, :min_z, :min_h, :min_w]
        divergence = du_dx + dv_dy + dw_dz
    else:
        min_h = min(du_dx.shape[3], dv_dy.shape[3])
        min_w = min(du_dx.shape[4], dv_dy.shape[4])
        du_dx = du_dx[:, :, :, :min_h, :min_w]
        dv_dy = dv_dy[:, :, :, :min_h, :min_w]
        divergence = du_dx + dv_dy
    return (divergence ** 2).mean()

def navier_stokes_residual(pred, target=None, nu=0.01, rho=1.0,
                          dx=1.0, dy=1.0, dz=1.0, dt=1.0, norm_mean=None, norm_std=None, **kwargs):
    """Navier-Stokes Residual Loss"""
    B, T, C = pred.shape[0], pred.shape[1], pred.shape[2]
    if C < 4:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    if T < 2:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    pred_real = denormalize_field(pred, norm_mean, norm_std)
    t_mid = T // 2
    u_curr = pred_real[:, t_mid, 0, :, :, :]
    v_curr = pred_real[:, t_mid, 1, :, :, :]
    w_curr = pred_real[:, t_mid, 2, :, :, :]
    p_curr = pred_real[:, t_mid, 3, :, :, :]

    if t_mid > 0 and t_mid < T - 1:
        dudt = (pred_real[:, t_mid+1, 0] - pred_real[:, t_mid-1, 0]) / (2 * dt)
        dvdt = (pred_real[:, t_mid+1, 1] - pred_real[:, t_mid-1, 1]) / (2 * dt)
        dwdt = (pred_real[:, t_mid+1, 2] - pred_real[:, t_mid-1, 2]) / (2 * dt)
    else:
        dudt = (pred_real[:, 1, 0] - pred_real[:, 0, 0]) / dt
        dvdt = (pred_real[:, 1, 1] - pred_real[:, 0, 1]) / dt
        dwdt = (pred_real[:, 1, 2] - pred_real[:, 0, 2]) / dt

    dudx = (u_curr[:, :, :, 2:] - u_curr[:, :, :, :-2]) / (2*dx)
    dudy = (u_curr[:, :, 2:, :] - u_curr[:, :, :-2, :]) / (2*dy)
    dudz = (u_curr[:, 2:, :, :] - u_curr[:, :-2, :, :]) / (2*dz)
    dvdx = (v_curr[:, :, :, 2:] - v_curr[:, :, :, :-2]) / (2*dx)
    dvdy = (v_curr[:, :, 2:, :] - v_curr[:, :, :-2, :]) / (2*dy)
    dvdz = (v_curr[:, 2:, :, :] - v_curr[:, :-2, :, :]) / (2*dz)
    dwdx = (w_curr[:, :, :, 2:] - w_curr[:, :, :, :-2]) / (2*dx)
    dwdy = (w_curr[:, :, 2:, :] - w_curr[:, :, :-2, :]) / (2*dy)
    dwdz = (w_curr[:, 2:, :, :] - w_curr[:, :-2, :, :]) / (2*dz)
    dpdx = (p_curr[:, :, :, 2:] - p_curr[:, :, :, :-2]) / (2*dx)
    dpdy = (p_curr[:, :, 2:, :] - p_curr[:, :, :-2, :]) / (2*dy)
    dpdz = (p_curr[:, 2:, :, :] - p_curr[:, :-2, :, :]) / (2*dz)

    d2udx2 = (u_curr[:, :, :, 2:] - 2*u_curr[:, :, :, 1:-1] + u_curr[:, :, :, :-2]) / (dx**2)
    d2udy2 = (u_curr[:, :, 2:, :] - 2*u_curr[:, :, 1:-1, :] + u_curr[:, :, :-2, :]) / (dy**2)
    d2udz2 = (u_curr[:, 2:, :, :] - 2*u_curr[:, 1:-1, :, :] + u_curr[:, :-2, :, :]) / (dz**2)
    d2vdx2 = (v_curr[:, :, :, 2:] - 2*v_curr[:, :, :, 1:-1] + v_curr[:, :, :, :-2]) / (dx**2)
    d2vdy2 = (v_curr[:, :, 2:, :] - 2*v_curr[:, :, 1:-1, :] + v_curr[:, :, :-2, :]) / (dy**2)
    d2vdz2 = (v_curr[:, 2:, :, :] - 2*v_curr[:, 1:-1, :, :] + v_curr[:, :-2, :, :]) / (dz**2)
    d2wdx2 = (w_curr[:, :, :, 2:] - 2*w_curr[:, :, :, 1:-1] + w_curr[:, :, :, :-2]) / (dx**2)
    d2wdy2 = (w_curr[:, :, 2:, :] - 2*w_curr[:, :, 1:-1, :] + w_curr[:, :, :-2, :]) / (dy**2)
    d2wdz2 = (w_curr[:, 2:, :, :] - 2*w_curr[:, 1:-1, :, :] + w_curr[:, :-2, :, :]) / (dz**2)

    z_sizes = [dudz.shape[1], dvdz.shape[1], dwdz.shape[1], dpdz.shape[1],
               d2udz2.shape[1], d2vdz2.shape[1], d2wdz2.shape[1], dudt.shape[1]]
    h_sizes = [dudy.shape[2], dvdy.shape[2], dwdy.shape[2], dpdy.shape[2],
               d2udy2.shape[2], d2vdy2.shape[2], d2wdy2.shape[2], dudt.shape[2]]
    w_sizes = [dudx.shape[3], dvdx.shape[3], dwdx.shape[3], dpdx.shape[3],
               d2udx2.shape[3], d2vdx2.shape[3], d2wdx2.shape[3], dudt.shape[3]]
    min_z, min_h, min_w = min(z_sizes), min(h_sizes), min(w_sizes)

    def crop(t, z, h, w):
        return t[:, :z, :h, :w]

    u = crop(u_curr[:, 1:-1, 1:-1, 1:-1], min_z, min_h, min_w)
    v = crop(v_curr[:, 1:-1, 1:-1, 1:-1], min_z, min_h, min_w)
    w = crop(w_curr[:, 1:-1, 1:-1, 1:-1], min_z, min_h, min_w)
    dudt = crop(dudt[:, 1:-1, 1:-1, 1:-1], min_z, min_h, min_w)
    dvdt = crop(dvdt[:, 1:-1, 1:-1, 1:-1], min_z, min_h, min_w)
    dwdt = crop(dwdt[:, 1:-1, 1:-1, 1:-1], min_z, min_h, min_w)
    dudx = crop(dudx[:, 1:-1, 1:-1, :], min_z, min_h, min_w)
    dudy = crop(dudy[:, 1:-1, :, 1:-1], min_z, min_h, min_w)
    dudz = crop(dudz[:, :, 1:-1, 1:-1], min_z, min_h, min_w)
    dvdx = crop(dvdx[:, 1:-1, 1:-1, :], min_z, min_h, min_w)
    dvdy = crop(dvdy[:, 1:-1, :, 1:-1], min_z, min_h, min_w)
    dvdz = crop(dvdz[:, :, 1:-1, 1:-1], min_z, min_h, min_w)
    dwdx = crop(dwdx[:, 1:-1, 1:-1, :], min_z, min_h, min_w)
    dwdy = crop(dwdy[:, 1:-1, :, 1:-1], min_z, min_h, min_w)
    dwdz = crop(dwdz[:, :, 1:-1, 1:-1], min_z, min_h, min_w)
    dpdx = crop(dpdx[:, 1:-1, 1:-1, :], min_z, min_h, min_w)
    dpdy = crop(dpdy[:, 1:-1, :, 1:-1], min_z, min_h, min_w)
    dpdz = crop(dpdz[:, :, 1:-1, 1:-1], min_z, min_h, min_w)
    d2udx2 = crop(d2udx2[:, 1:-1, 1:-1, :], min_z, min_h, min_w)
    d2udy2 = crop(d2udy2[:, 1:-1, :, 1:-1], min_z, min_h, min_w)
    d2udz2 = crop(d2udz2[:, :, 1:-1, 1:-1], min_z, min_h, min_w)
    d2vdx2 = crop(d2vdx2[:, 1:-1, 1:-1, :], min_z, min_h, min_w)
    d2vdy2 = crop(d2vdy2[:, 1:-1, :, 1:-1], min_z, min_h, min_w)
    d2vdz2 = crop(d2vdz2[:, :, 1:-1, 1:-1], min_z, min_h, min_w)
    d2wdx2 = crop(d2wdx2[:, 1:-1, 1:-1, :], min_z, min_h, min_w)
    d2wdy2 = crop(d2wdy2[:, 1:-1, :, 1:-1], min_z, min_h, min_w)
    d2wdz2 = crop(d2wdz2[:, :, 1:-1, 1:-1], min_z, min_h, min_w)

    conv_u = u*dudx + v*dudy + w*dudz
    conv_v = u*dvdx + v*dvdy + w*dvdz
    conv_w = u*dwdx + v*dwdy + w*dwdz
    lap_u = nu * (d2udx2 + d2udy2 + d2udz2)
    lap_v = nu * (d2vdx2 + d2vdy2 + d2vdz2)
    lap_w = nu * (d2wdx2 + d2wdy2 + d2wdz2)
    press_u = -(1.0/rho) * dpdx
    press_v = -(1.0/rho) * dpdy
    press_w = -(1.0/rho) * dpdz

    res_u = dudt + conv_u + press_u - lap_u
    res_v = dvdt + conv_v + press_v - lap_v
    res_w = dwdt + conv_w + press_w - lap_w

    loss = (res_u**2).mean() + (res_v**2).mean() + (res_w**2).mean()
    return loss

def gradient_loss_3d(pred, target, norm_mean=None, norm_std=None, **kwargs):
    """3D Gradient Loss (在真实物理空间计算)"""
    # 反归一化到真实物理空间
    pred_real = denormalize_field(pred, norm_mean, norm_std)
    target_real = denormalize_field(target, norm_mean, norm_std)

    pred_dz = pred_real[:, :, :, 1:, :, :] - pred_real[:, :, :, :-1, :, :]
    target_dz = target_real[:, :, :, 1:, :, :] - target_real[:, :, :, :-1, :, :]
    pred_dh = pred_real[:, :, :, :, 1:, :] - pred_real[:, :, :, :, :-1, :]
    target_dh = target_real[:, :, :, :, 1:, :] - target_real[:, :, :, :, :-1, :]
    pred_dw = pred_real[:, :, :, :, :, 1:] - pred_real[:, :, :, :, :, :-1]
    target_dw = target_real[:, :, :, :, :, 1:] - target_real[:, :, :, :, :, :-1]
    loss_z = torch.abs(pred_dz - target_dz).mean()
    loss_h = torch.abs(pred_dh - target_dh).mean()
    loss_w = torch.abs(pred_dw - target_dw).mean()
    return loss_z + loss_h + loss_w

def get_physics_loss(loss_type, config_dict):
    losses = {
        'divergence_3d': divergence_loss_3d,
        'navier_stokes': navier_stokes_residual,
        'gradient_3d': gradient_loss_3d,
    }
    if loss_type not in losses:
        raise ValueError(f"Unknown physics loss type: {loss_type}")
    loss_fn = losses[loss_type]
    def wrapped_loss(pred, target=None, **kwargs):
        merged_kwargs = {**config_dict, **kwargs}
        return loss_fn(pred, target, **merged_kwargs)
    return wrapped_loss

print("Physics loss functions loaded.")

"""Loss Factory"""

class LossFactory:
    """Loss function factory - manages multiple loss functions"""

    def __init__(self, config, norm_mean=None, norm_std=None):
        if isinstance(config, dict):
            class ConfigWrapper:
                pass
            wrapper = ConfigWrapper()
            wrapper.LOSS_CONFIG = config
            self.config = wrapper
        else:
            self.config = config

        self.loss_fns = {}
        self.weights = {}
        self.norm_mean = norm_mean
        self.norm_std = norm_std

        self.loss_spatial_mode = {
            'divergence_2d': 'in_plane', 'divergence_3d': 'full_field',
            'navier_stokes': 'full_field', 'vorticity': 'in_plane',
            'spectral_divergence': 'full_field', 'spectral_divergence_soft': 'full_field',
            'spectral_ns': 'full_field', 'spectral_weighted': 'full_field',
            'spectral_dynamic': 'full_field', 'gradient_3d': 'full_field',
            'tv_3d': 'full_field', 'edge_smooth': 'full_field', 'laplacian': 'full_field',
        }

        if hasattr(config, 'PHYSICS_ON_MISSING_ONLY') and config.PHYSICS_ON_MISSING_ONLY:
            for key in self.loss_spatial_mode.keys():
                self.loss_spatial_mode[key] = 'in_plane'

        self._build_losses()

    def _build_losses(self):
        for loss_name, loss_config in self.config.LOSS_CONFIG.items():
            loss_type = loss_config['type']
            weight = loss_config.get('weight', 1.0)

            if loss_type in ['mse', 'mae', 'l1', 'huber', 'relative_l2']:
                loss_fn = get_data_loss(loss_type)
            elif loss_type in ['divergence_2d', 'divergence_3d', 'navier_stokes', 'vorticity',
                               'gradient_3d', 'tv_3d', 'edge_smooth', 'laplacian']:
                physics_config = dict(loss_config)
                physics_config['norm_mean'] = self.norm_mean
                physics_config['norm_std'] = self.norm_std
                loss_fn = get_physics_loss(loss_type, physics_config)
            elif loss_type in ['fourier', 'wavelet', 'energy_spectrum', 'spectral_weighted']:
                loss_fn = get_spectral_loss(loss_type, loss_config)
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")

            self.loss_fns[loss_name] = loss_fn
            self.weights[loss_name] = weight

    def get_loss_names(self):
        return list(self.loss_fns.keys())

    def compute(self, pred, target, missing_indices=None, pred_full=None, target_full=None):
        losses = {}
        total_loss = 0.0

        for loss_name, loss_fn in self.loss_fns.items():
            loss_type = self.config.LOSS_CONFIG[loss_name]['type']
            loss_config = self.config.LOSS_CONFIG[loss_name]

            if loss_name == 'data':
                if loss_type in ['spectral_weighted', 'spectral_dynamic']:
                    target_mode = loss_config.get('target', 'full')
                    if target_mode == 'full' and pred_full is not None:
                        loss_val = loss_fn(pred_full, target_full)
                    else:
                        loss_val = loss_fn(pred, target)
                    if hasattr(loss_fn, 'get_last_band_errors'):
                        band_errors = loss_fn.get_last_band_errors()
                        losses['data_low'] = band_errors.get('low', 0.0)
                        losses['data_mid'] = band_errors.get('mid', 0.0)
                        losses['data_high'] = band_errors.get('high', 0.0)
                else:
                    loss_val = loss_fn(pred, target, missing_indices=missing_indices)
            else:
                spatial_mode = self.loss_spatial_mode.get(loss_type, 'full_field')
                if spatial_mode == 'in_plane' and missing_indices is not None:
                    pred_physics = self._extract_missing(pred_full if pred_full is not None else pred, missing_indices)
                    target_physics = self._extract_missing(target_full if target_full is not None else target, missing_indices)
                    loss_val = loss_fn(pred_physics, target_physics)
                else:
                    if pred_full is not None:
                        loss_val = loss_fn(pred_full, target_full)
                    else:
                        loss_val = loss_fn(pred, target)

            weighted_loss = self.weights[loss_name] * loss_val
            losses[loss_name] = loss_val
            total_loss += weighted_loss

        losses['total'] = total_loss
        return losses

    def _extract_missing(self, volume, missing_indices):
        B = volume.shape[0]
        missing_slices = []
        for b in range(B):
            missing_z = missing_indices[b]
            missing_slices.append(volume[b, :, :, missing_z, :, :])
        return torch.stack(missing_slices, dim=0)


def build_loss_factory(config, norm_mean=None, norm_std=None):
    return LossFactory(config, norm_mean=norm_mean, norm_std=norm_std)

print("Loss factory loaded.")

"""3D Swin Transformer Blocks"""

from timm.models.layers import DropPath

# Window Partition Utilities
def window_partition_3d(x, window_size):
    """Partition input into 3D windows."""
    B, C, Z, H, W = x.shape
    Wz, Wh, Ww = window_size
    x = x.view(B, C, Z // Wz, Wz, H // Wh, Wh, W // Ww, Ww)
    x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
    windows = x.view(-1, Wz * Wh * Ww, C)
    return windows

def window_reverse_3d(windows, window_size, Z, H, W):
    """Reverse window partition back to full spatial dimensions."""
    Wz, Wh, Ww = window_size
    num_windows_z = Z // Wz
    num_windows_h = H // Wh
    num_windows_w = W // Ww
    B = windows.shape[0] // (num_windows_z * num_windows_h * num_windows_w)
    C = windows.shape[-1]
    x = windows.view(B, num_windows_z, num_windows_h, num_windows_w, Wz, Wh, Ww, C)
    x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
    x = x.view(B, C, Z, H, W)
    return x

def create_shifted_window_mask_3d(Z, H, W, window_size, shift_size, device='cuda'):
    """Create attention mask for shifted windows."""
    Wz, Wh, Ww = window_size
    Sz, Sh, Sw = shift_size
    img_mask = torch.zeros((1, Z, H, W, 1), device=device)
    z_slices = (slice(0, -Wz), slice(-Wz, -Sz), slice(-Sz, None))
    h_slices = (slice(0, -Wh), slice(-Wh, -Sh), slice(-Sh, None))
    w_slices = (slice(0, -Ww), slice(-Ww, -Sw), slice(-Sw, None))
    cnt = 0
    for z in z_slices:
        for h in h_slices:
            for w in w_slices:
                img_mask[:, z, h, w, :] = cnt
                cnt += 1
    img_mask = rearrange(img_mask, 'b z h w c -> b c z h w')
    mask_windows = window_partition_3d(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
    attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


class WindowAttention3D(nn.Module):
    """3D Window-based Multi-Head Self-Attention."""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True,
                 attn_drop=0., proj_drop=0., attention_type='standard'):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.attention_type = attention_type
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads)
        )
        coords_z = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_z, coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, x_kv=None, mask=None):
        """
        Args:
            x: (B*num_windows, Wz*Wh*Ww, C) - query input
            x_kv: (B*num_windows, Wz*Wh*Ww, C) - key/value input for cross-attention
            mask: (num_windows, Wz*Wh*Ww, Wz*Wh*Ww) or None
        Returns:
            output: (B*num_windows, Wz*Wh*Ww, C)
        """
        BW, N, C = x.shape
        is_cross_attn = x_kv is not None

        if self.attention_type == 'standard':
            return self._standard_attention(x, x_kv, mask)
        elif self.attention_type == 'axial':
            return self._axial_attention(x, x_kv, mask)
        else:
            raise ValueError(f"Unknown attention type: {self.attention_type}")

    def _standard_attention(self, x, x_kv, mask):
        BW, N, C = x.shape
        is_cross_attn = x_kv is not None

        if is_cross_attn:
            q = self.q_proj(x).reshape(BW, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            kv = self.kv_proj(x_kv).reshape(BW, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]
        else:
            qkv = self.qkv(x).reshape(BW, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(BW // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(BW, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _axial_attention(self, x, x_kv, mask):
        """
        Axial attention: factorized attention along Z, H, W axes.
        Complexity: O(Wz*N + Wh*N + Ww*N) vs O(N^2) for standard attention
        """
        BW, N, C = x.shape
        Wz, Wh, Ww = self.window_size
        is_cross_attn = x_kv is not None

        # Reshape to spatial: (BW, Wz, Wh, Ww, C)
        x_spatial = x.view(BW, Wz, Wh, Ww, C)

        # QKV projection
        if is_cross_attn:
            q = self.q_proj(x).reshape(BW, N, self.num_heads, C // self.num_heads)
            q = q.view(BW, Wz, Wh, Ww, self.num_heads, C // self.num_heads)
            kv = self.kv_proj(x_kv).reshape(BW, N, 2, self.num_heads, C // self.num_heads)
            kv_spatial = kv.view(BW, Wz, Wh, Ww, 2, self.num_heads, C // self.num_heads)
            k, v = kv_spatial[..., 0, :, :], kv_spatial[..., 1, :, :]
        else:
            qkv = self.qkv(x).reshape(BW, N, 3, self.num_heads, C // self.num_heads)
            qkv_spatial = qkv.view(BW, Wz, Wh, Ww, 3, self.num_heads, C // self.num_heads)
            q, k, v = qkv_spatial[..., 0, :, :], qkv_spatial[..., 1, :, :], qkv_spatial[..., 2, :, :]

        output = torch.zeros_like(x_spatial)

        # Z-axis attention
        q_z = rearrange(q, 'bw z h w nh d -> (bw h w) nh z d')
        k_z = rearrange(k, 'bw z h w nh d -> (bw h w) nh z d')
        v_z = rearrange(v, 'bw z h w nh d -> (bw h w) nh z d')
        attn_z = (q_z * self.scale) @ k_z.transpose(-2, -1)
        attn_z = F.softmax(attn_z, dim=-1)
        attn_z = self.attn_drop(attn_z)
        out_z = attn_z @ v_z
        out_z = rearrange(out_z, '(bw h w) nh z d -> bw z h w (nh d)', bw=BW, h=Wh, w=Ww)
        output = output + out_z

        # H-axis attention
        q_h = rearrange(q, 'bw z h w nh d -> (bw z w) nh h d')
        k_h = rearrange(k, 'bw z h w nh d -> (bw z w) nh h d')
        v_h = rearrange(v, 'bw z h w nh d -> (bw z w) nh h d')
        attn_h = (q_h * self.scale) @ k_h.transpose(-2, -1)
        attn_h = F.softmax(attn_h, dim=-1)
        attn_h = self.attn_drop(attn_h)
        out_h = attn_h @ v_h
        out_h = rearrange(out_h, '(bw z w) nh h d -> bw z h w (nh d)', bw=BW, z=Wz, w=Ww)
        output = output + out_h

        # W-axis attention
        q_w = rearrange(q, 'bw z h w nh d -> (bw z h) nh w d')
        k_w = rearrange(k, 'bw z h w nh d -> (bw z h) nh w d')
        v_w = rearrange(v, 'bw z h w nh d -> (bw z h) nh w d')
        attn_w = (q_w * self.scale) @ k_w.transpose(-2, -1)
        attn_w = F.softmax(attn_w, dim=-1)
        attn_w = self.attn_drop(attn_w)
        out_w = attn_w @ v_w
        out_w = rearrange(out_w, '(bw z h) nh w d -> bw z h w (nh d)', bw=BW, z=Wz, h=Wh)
        output = output + out_w

        # Average the three axes
        output = output / 3.0

        # Project and reshape back
        output = output.view(BW, N, C)
        output = self.proj(output)
        output = self.proj_drop(output)

        return output



class Mlp(nn.Module):
    """MLP module with GELU activation."""
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwinTransformerBlock3D(nn.Module):
    """Swin Transformer Block with Window-based and Shifted-Window attention."""

    def __init__(self, dim, input_resolution, num_heads, window_size=(5, 8, 8),
                 shift_size=(0, 0, 0), mlp_ratio=4., qkv_bias=True, drop=0.,
                 attn_drop=0., drop_path=0., attention_type='standard',
                 use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.attention_type = attention_type
        self.use_checkpoint = use_checkpoint

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)

        self.attn_regular = WindowAttention3D(dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, attention_type=attention_type)
        self.attn_shifted = WindowAttention3D(dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, attention_type=attention_type)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if any(s > 0 for s in self.shift_size):
            Z, H, W = self.input_resolution
            self.register_buffer("attn_mask", create_shifted_window_mask_3d(Z, H, W, self.window_size, self.shift_size))
        else:
            self.attn_mask = None

    def _forward_part1(self, x):
        B, C, Z, H, W = x.shape
        shortcut = x
        x = rearrange(x, 'b c z h w -> b z h w c')
        x_flat = x.view(B, Z * H * W, C)
        x_flat = self.norm1(x_flat)
        x = x_flat.view(B, Z, H, W, C)
        x = rearrange(x, 'b z h w c -> b c z h w')
        x_windows = window_partition_3d(x, self.window_size)
        attn_windows = self.attn_regular(x_windows, mask=None)
        x = window_reverse_3d(attn_windows, self.window_size, Z, H, W)
        x = shortcut + self.drop_path(x)
        shortcut = x
        x = rearrange(x, 'b c z h w -> b z h w c')
        x_flat = x.view(B, Z * H * W, C)
        x_flat = self.norm2(x_flat)
        x_flat = self.mlp1(x_flat)
        x = x_flat.view(B, Z, H, W, C)
        x = rearrange(x, 'b z h w c -> b c z h w')
        x = shortcut + self.drop_path(x)
        return x

    def _forward_part2(self, x):
        B, C, Z, H, W = x.shape
        shortcut = x
        x = rearrange(x, 'b c z h w -> b z h w c')
        x_flat = x.view(B, Z * H * W, C)
        x_flat = self.norm3(x_flat)
        x = x_flat.view(B, Z, H, W, C)
        x = rearrange(x, 'b z h w c -> b c z h w')
        if any(s > 0 for s in self.shift_size):
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(2, 3, 4))
        else:
            shifted_x = x
        x_windows = window_partition_3d(shifted_x, self.window_size)
        attn_windows = self.attn_shifted(x_windows, mask=self.attn_mask)
        shifted_x = window_reverse_3d(attn_windows, self.window_size, Z, H, W)
        if any(s > 0 for s in self.shift_size):
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(2, 3, 4))
        else:
            x = shifted_x
        x = shortcut + self.drop_path(x)
        shortcut = x
        x = rearrange(x, 'b c z h w -> b z h w c')
        x_flat = x.view(B, Z * H * W, C)
        x_flat = self.norm4(x_flat)
        x_flat = self.mlp2(x_flat)
        x = x_flat.view(B, Z, H, W, C)
        x = rearrange(x, 'b z h w c -> b c z h w')
        x = shortcut + self.drop_path(x)
        return x

    def forward(self, x):
        if self.use_checkpoint and self.training:
            x = torch.utils.checkpoint.checkpoint(self._forward_part1, x, use_reentrant=False)
            x = torch.utils.checkpoint.checkpoint(self._forward_part2, x, use_reentrant=False)
        else:
            x = self._forward_part1(x)
            x = self._forward_part2(x)
        return x


class SwinStage3D(nn.Module):
    """A stage of Swin Transformer consisting of multiple Swin blocks."""

    def __init__(self, dim, input_resolution, depth, num_heads, window_size=(5, 8, 8),
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 attention_type='standard', use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else (window_size[0] // 2, window_size[1] // 2, window_size[2] // 2),
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                attention_type=attention_type, use_checkpoint=use_checkpoint
            )
            for i in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

print("Swin Transformer blocks loaded.")

"""Complete Swin Transformer Model with Patch Embedding"""

# Patch Embedding & Recovery
class PatchEmbed3D(nn.Module):
    """3D Patch Embedding using strided convolution."""
    def __init__(self, patch_size=(2, 4, 4), in_channels=4, embed_dim=96):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        B, C, Z, H, W = x.shape
        x = rearrange(x, 'b c z h w -> b z h w c')
        x = self.norm(x)
        x = rearrange(x, 'b z h w c -> b c z h w')
        return x


class PatchEmbedSparse(nn.Module):
    """Patch Embedding for sparse observations with position-aware z-interpolation."""
    def __init__(self, patch_size_hw=(4, 4), in_channels=4, embed_dim=96,
                 num_sparse_z_layers=12, patched_z=32, total_z=64):
        super().__init__()
        self.num_sparse_z_layers = num_sparse_z_layers
        self.patched_z = patched_z
        self.total_z = total_z
        self.patch_size_z = total_z // patched_z
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=(1, patch_size_hw[0], patch_size_hw[1]),
                              stride=(1, patch_size_hw[0], patch_size_hw[1]))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, sparse_indices=None):
        x = self.proj(x)
        B, C, Z_sparse, H, W = x.shape
        x = rearrange(x, 'b c z h w -> b z h w c')
        x = self.norm(x)
        x = rearrange(x, 'b z h w c -> b c z h w')
        if Z_sparse != self.patched_z:
            if sparse_indices is not None:
                x = self._position_aware_interpolate(x, sparse_indices)
            else:
                x = F.interpolate(x, size=(self.patched_z, H, W), mode='trilinear', align_corners=False)
        return x

    def _position_aware_interpolate(self, x, sparse_indices):
        B, C, num_sparse, H, W = x.shape
        device = x.device
        sparse_patch_positions = torch.tensor([idx / self.patch_size_z for idx in sparse_indices],
                                               device=device, dtype=torch.float32)
        output = torch.zeros(B, C, self.patched_z, H, W, device=device, dtype=x.dtype)
        for pz in range(self.patched_z):
            pz_float = float(pz)
            right_idx = 0
            for i, pos in enumerate(sparse_patch_positions):
                if pos >= pz_float:
                    right_idx = i
                    break
                right_idx = i + 1
            if right_idx == 0:
                output[:, :, pz, :, :] = x[:, :, 0, :, :]
            elif right_idx >= num_sparse:
                output[:, :, pz, :, :] = x[:, :, -1, :, :]
            else:
                left_idx = right_idx - 1
                left_pos = sparse_patch_positions[left_idx].item()
                right_pos = sparse_patch_positions[right_idx].item()
                weight = 0.5 if right_pos == left_pos else (pz_float - left_pos) / (right_pos - left_pos)
                output[:, :, pz, :, :] = (1 - weight) * x[:, :, left_idx, :, :] + weight * x[:, :, right_idx, :, :]
        return output


class PatchRecovery3D(nn.Module):
    """Recover from patch space back to full resolution."""
    def __init__(self, patch_size=(2, 4, 4), embed_dim=96, out_channels=4):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.ConvTranspose3d(embed_dim, out_channels, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x)


def interpolate_sparse_to_full(sparse_obs, sparse_indices, total_z=64):
    """Z-direction linear interpolation at original resolution."""
    B, C, num_sparse, H, W = sparse_obs.shape
    device = sparse_obs.device
    dtype = sparse_obs.dtype
    output = torch.zeros(B, C, total_z, H, W, device=device, dtype=dtype)
    sparse_z = torch.as_tensor(sparse_indices, device=device, dtype=torch.float32)
    for z in range(total_z):
        z_float = float(z)
        right_idx = 0
        for i, pos in enumerate(sparse_z):
            if pos >= z_float:
                right_idx = i
                break
            right_idx = i + 1
        if right_idx == 0:
            output[:, :, z, :, :] = sparse_obs[:, :, 0, :, :]
        elif right_idx >= num_sparse:
            output[:, :, z, :, :] = sparse_obs[:, :, -1, :, :]
        else:
            left_idx = right_idx - 1
            left_pos = sparse_z[left_idx].item()
            right_pos = sparse_z[right_idx].item()
            weight = 0.5 if right_pos == left_pos else (z_float - left_pos) / (right_pos - left_pos)
            output[:, :, z, :, :] = (1 - weight) * sparse_obs[:, :, left_idx, :, :] + weight * sparse_obs[:, :, right_idx, :, :]
    return output


# Position Encoding
class PositionEncoding3D(nn.Module):
    def __init__(self, dim, max_z=32, max_h=32, max_w=32):
        super().__init__()
        dim_z = dim // 3
        dim_h = dim // 3
        dim_w = dim - dim_z - dim_h
        self.z_embed = nn.Parameter(torch.randn(1, dim_z, max_z, 1, 1) * 0.02)
        self.h_embed = nn.Parameter(torch.randn(1, dim_h, 1, max_h, 1) * 0.02)
        self.w_embed = nn.Parameter(torch.randn(1, dim_w, 1, 1, max_w) * 0.02)

    def forward(self, x):
        B, C, Z, H, W = x.shape
        z_enc = self.z_embed[:, :, :Z, :, :]
        h_enc = self.h_embed[:, :, :, :H, :]
        w_enc = self.w_embed[:, :, :, :, :W]
        pos_enc = torch.cat([z_enc.expand(B, -1, Z, H, W), h_enc.expand(B, -1, Z, H, W), w_enc.expand(B, -1, Z, H, W)], dim=1)
        return x + pos_enc


class TemporalPositionEncoding(nn.Module):
    def __init__(self, dim, max_t=10):
        super().__init__()
        self.t_embed = nn.Parameter(torch.randn(1, max_t, dim) * 0.02)

    def forward(self, x):
        B, T, D = x.shape
        return x + self.t_embed[:, :T, :D]


# Encoder
class SwinEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        in_channels = config.IN_CHANNELS
        embed_dim = config.EMBED_DIM
        depths = config.DEPTHS
        num_heads = config.NUM_HEADS
        window_size = config.WINDOW_SIZE
        mlp_ratio = config.MLP_RATIO
        drop_rate = config.DROP_RATE
        attn_drop_rate = config.ATTN_DROP_RATE
        drop_path_rate = config.DROP_PATH_RATE
        attention_type = config.ATTENTION_TYPE
        patch_size = config.PATCH_SIZE
        self.patch_embed = PatchEmbed3D(patch_size, in_channels, embed_dim)
        patched_resolution = config.PATCHED_RESOLUTION
        self.pos_embed = PositionEncoding3D(embed_dim, max_z=patched_resolution[0],
                                            max_h=patched_resolution[1], max_w=patched_resolution[2])
        self.pos_drop = nn.Dropout(drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        use_checkpoint = getattr(config, 'USE_GRADIENT_CHECKPOINTING', False)
        self.stages = nn.ModuleList()
        for i_stage in range(len(depths)):
            stage = SwinStage3D(dim=embed_dim, input_resolution=patched_resolution, depth=depths[i_stage],
                               num_heads=num_heads[i_stage], window_size=window_size, mlp_ratio=mlp_ratio,
                               qkv_bias=True, drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_stage]):sum(depths[:i_stage + 1])],
                               attention_type=attention_type, use_checkpoint=use_checkpoint)
            self.stages.append(stage)
        self.num_features = embed_dim
        self.norm = nn.LayerNorm(self.num_features)

    def forward(self, x):
        B, T, C, Z, H, W = x.shape
        features_list = []
        for t in range(T):
            x_t = x[:, t]
            x_t = self.patch_embed(x_t)
            x_t = self.pos_embed(x_t)
            x_t = self.pos_drop(x_t)
            for stage in self.stages:
                x_t = stage(x_t)
            x_t = rearrange(x_t, 'b c z h w -> b z h w c')
            x_t = self.norm(x_t)
            x_t = rearrange(x_t, 'b z h w c -> b c z h w')
            features_list.append(x_t)
        features = torch.stack(features_list, dim=1)
        return features


# Temporal Attention
class TemporalAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm = nn.LayerNorm(dim)
        self.temporal_pos = TemporalPositionEncoding(dim, max_t=10)

    def forward(self, x):
        B, T, D, Z, H, W = x.shape
        x = rearrange(x, 'b t d z h w -> (b z h w) t d')
        x = self.temporal_pos(x)
        x = self.norm(x)
        N = x.shape[0]
        qkv = self.qkv(x).reshape(N, T, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(N, T, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = rearrange(x, '(b z h w) t d -> b t d z h w', b=B, z=Z, h=H, w=W)
        return x


# Decoder
class SwinDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dual_residual = getattr(config, 'DUAL_RESIDUAL', False)
        dim = config.DECODER_DIM
        num_heads = config.DECODER_NUM_HEADS
        window_size = config.WINDOW_SIZE
        attention_type = config.ATTENTION_TYPE
        patch_size = config.PATCH_SIZE
        patched_resolution = config.PATCHED_RESOLUTION
        self.sparse_patch_embed = PatchEmbedSparse(patch_size_hw=(4, 4), in_channels=config.IN_CHANNELS,
            embed_dim=dim, num_sparse_z_layers=config.NUM_SPARSE_Z_LAYERS,
            patched_z=patched_resolution[0], total_z=config.INPUT_RESOLUTION[0])
        self.encoder_proj = nn.Conv3d(config.EMBED_DIM, dim, kernel_size=1)
        if self.dual_residual:
            self.temporal_patch_embed = PatchEmbed3D(patch_size, config.IN_CHANNELS, dim)
            self.dual_fusion = nn.Sequential(nn.Conv3d(dim * 3, dim, kernel_size=1), nn.GroupNorm(8, dim), nn.GELU())
            delta_scale_init = getattr(config, 'DELTA_SCALE_INIT', 0.1)
            self.delta_scale = nn.Parameter(torch.ones(1) * delta_scale_init)
        self.cross_attn = WindowAttention3D(dim=dim, window_size=window_size, num_heads=num_heads,
                                            qkv_bias=True, attention_type=attention_type)
        self.cross_attn_norm = nn.LayerNorm(dim)
        self.refinement = nn.Sequential(nn.Conv3d(dim, dim, kernel_size=3, padding=1), nn.GroupNorm(8, dim), nn.GELU(),
                                        nn.Conv3d(dim, dim, kernel_size=3, padding=1), nn.GroupNorm(8, dim), nn.GELU())
        self.patch_recovery = PatchRecovery3D(patch_size, dim, config.IN_CHANNELS)

    def forward(self, sparse_obs, encoder_features, sparse_indices, input_last=None):
        B, T_out, C, num_sparse_z_layers, H, W = sparse_obs.shape
        total_z = self.config.INPUT_RESOLUTION[0]
        encoder_context = encoder_features.mean(dim=1)
        encoder_context = self.encoder_proj(encoder_context)
        outputs = []
        deltas = []
        for t in range(T_out):
            sparse_t = sparse_obs[:, t]
            interpolated_baseline = interpolate_sparse_to_full(sparse_t, sparse_indices, total_z)
            if self.dual_residual and input_last is not None:
                temporal_base = input_last
                temporal_feat = self.temporal_patch_embed(temporal_base)
                sparse_feat = self.sparse_patch_embed(sparse_t, sparse_indices)
                hint_feat = sparse_feat - temporal_feat
                fused = torch.cat([sparse_feat, hint_feat, encoder_context], dim=1)
                full_field = self.dual_fusion(fused)
            else:
                sparse_t_patched = self.sparse_patch_embed(sparse_t, sparse_indices)
                full_field = sparse_t_patched + encoder_context
            query_windows = window_partition_3d(full_field, self.config.WINDOW_SIZE)
            kv_windows = window_partition_3d(encoder_context, self.config.WINDOW_SIZE)
            query_windows = self.cross_attn_norm(query_windows)
            kv_windows = self.cross_attn_norm(kv_windows)
            attn_out = self.cross_attn(query_windows, kv_windows)
            attn_out = window_reverse_3d(attn_out, self.config.WINDOW_SIZE,
                                         self.config.PATCHED_RESOLUTION[0],
                                         self.config.PATCHED_RESOLUTION[1],
                                         self.config.PATCHED_RESOLUTION[2])
            full_field = full_field + attn_out
            full_field = full_field + self.refinement(full_field)
            delta = self.patch_recovery(full_field)
            if self.dual_residual and input_last is not None:
                delta = delta * self.delta_scale
            output_t = interpolated_baseline + delta
            outputs.append(output_t)
            deltas.append(delta)
        output = torch.stack(outputs, dim=1)
        delta_stack = torch.stack(deltas, dim=1)
        return output, delta_stack


# Complete Model
class SwinTransformerPhysics(nn.Module):
    """Complete 3D Swin Transformer with Patch Embedding."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = SwinEncoder(config)
        self.temporal_attn = TemporalAttention(dim=config.EMBED_DIM, num_heads=config.TEMPORAL_NUM_HEADS,
                                               attn_drop=config.ATTN_DROP_RATE, proj_drop=config.DROP_RATE)
        self.decoder = SwinDecoder(config)

    def forward(self, batch):
        input_dense = batch['input_dense']
        output_sparse = batch['output_sparse_z']
        sparse_indices = batch['sparse_indices']['z']
        if sparse_indices.dim() > 1:
            sparse_indices = sparse_indices[0]
        encoder_features = self.encoder(input_dense)
        temporal_features = self.temporal_attn(encoder_features)
        input_last = input_dense[:, -1] if self.decoder.dual_residual else None
        output_full, delta = self.decoder(output_sparse, temporal_features, sparse_indices, input_last)
        for idx, z_idx in enumerate(sparse_indices):
            output_full[:, :, :, z_idx, :, :] = output_sparse[:, :, :, idx, :, :]
        result = {'output_full': output_full, 'delta': delta,
                  'encoder_features': encoder_features, 'temporal_features': temporal_features}
        if self.decoder.dual_residual:
            result['delta_scale'] = self.decoder.delta_scale.item()
        return result

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

print("Swin Transformer model loaded.")

"""JHU DNS128 Dataset"""

class SparseJHUDataset(Dataset):
    """
    JHU DNS128 Sparse Dataset
    
    Supports:
    - Cubic grid (128, 128, 128)
    - File naming format: {indicator}_{timestep}.npy
    - 200 timesteps
    """

    def __init__(self,
                 data_dir,
                 split='train',
                 split_config_path=None,
                 indicators=['u', 'v', 'w', 'p'],
                 sparse_z_interval=None,
                 normalize=True,
                 norm_stats=None,
                 dense_z_size=128):
        """
        Args:
            data_dir: JHU data directory
            split: 'train', 'val', 'test'
            split_config_path: Split config JSON path
            indicators: Physical quantities
            sparse_z_interval: z-axis sparse sampling interval (e.g., 8 means [0, 8, 16, ...])
            normalize: Whether to normalize
            norm_stats: Normalization statistics dict
            dense_z_size: Total Z layers (128 for JHU)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.indicators = indicators
        self.dense_z_size = dense_z_size

        if sparse_z_interval is None:
            raise ValueError("sparse_z_interval must be explicitly provided!")
        self.sparse_z_interval = sparse_z_interval

        self.normalize = normalize

        if split_config_path is None:
            raise ValueError("split_config_path must be provided")

        with open(split_config_path, 'r') as f:
            config = json.load(f)

        self.sample_indices = config['splits'][split]
        self.metadata = config['metadata']
        self.input_length = self.metadata['input_length']
        self.output_length = self.metadata['output_length']

        # Compute sparse and missing layer indices
        self.sparse_z_indices = list(range(0, self.dense_z_size, sparse_z_interval))
        self.missing_z_indices = [i for i in range(self.dense_z_size)
                                  if i not in self.sparse_z_indices]

        print(f"\n{split.upper()} JHU Dataset initialized:")
        print(f"  Data directory: {data_dir}")
        print(f"  Samples: {len(self.sample_indices)}")
        print(f"  Indicators: {indicators}")
        print(f"  Grid size: ({self.dense_z_size}, 128, 128) - cubic")
        print(f"  Sparse z layers: {len(self.sparse_z_indices)} / {self.dense_z_size}")
        print(f"  Sparse z indices: {self.sparse_z_indices[:5]}...{self.sparse_z_indices[-3:]}")
        print(f"  Missing z layers: {len(self.missing_z_indices)}")

        # Normalization stats
        if normalize:
            if norm_stats is None:
                self.norm_stats = {
                    'u': {'mean': 0.0, 'std': 1.0},
                    'v': {'mean': 0.0, 'std': 1.0},
                    'w': {'mean': 0.0, 'std': 1.0},
                    'p': {'mean': 0.0, 'std': 1.0}
                }
                print(f"  Normalization: enabled (using default stats)")
            else:
                self.norm_stats = norm_stats
                print(f"  Normalization: enabled")

            for ind in indicators:
                if ind in self.norm_stats:
                    print(f"    {ind}: mean={self.norm_stats[ind]['mean']:.4f}, "
                          f"std={self.norm_stats[ind]['std']:.4f}")
        else:
            print(f"  Normalization: disabled")

    def __len__(self):
        return len(self.sample_indices)

    def _load_frame(self, indicator, timestamp):
        """Load single frame data"""
        ts_float = float(timestamp)
        ts_formatted = f"{ts_float:.3f}"
        filename = f"{indicator}_{ts_formatted}.npy"
        filepath = self.data_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        data = np.load(filepath).astype(np.float32)

        if self.normalize and indicator in self.norm_stats:
            mean = self.norm_stats[indicator]['mean']
            std = self.norm_stats[indicator]['std']
            data = (data - mean) / (std + 1e-8)

        return data

    def __getitem__(self, idx):
        """
        Returns:
            dict: {
                'input_dense': (input_length, C, 128, 128, 128)
                'output_sparse': (output_length, C, n_sparse, 128, 128)
                'output_dense': (output_length, C, 128, 128, 128)
                'sparse_z_indices': sparse z layer indices
                'missing_z_indices': missing z layer indices
                'sample_start': sample start index
                'global_timesteps': (input_length + output_length,) global timestep sequence
            }
        """
        sample_timestamps = self.sample_indices[idx]
        input_timestamps = sample_timestamps[:self.input_length]
        output_timestamps = sample_timestamps[self.input_length:]

        # Load input data
        input_data = []
        for indicator in self.indicators:
            indicator_frames = []
            for t in input_timestamps:
                frame = self._load_frame(indicator, t)
                indicator_frames.append(frame)
            input_data.append(np.stack(indicator_frames, axis=0))

        input_dense = np.stack(input_data, axis=1)

        # Load output data
        output_data = []
        for indicator in self.indicators:
            indicator_frames = []
            for t in output_timestamps:
                frame = self._load_frame(indicator, t)
                indicator_frames.append(frame)
            output_data.append(np.stack(indicator_frames, axis=0))

        output_dense = np.stack(output_data, axis=1)
        output_sparse = output_dense[:, :, self.sparse_z_indices, :, :]

        timestamps_float = [float(t) for t in sample_timestamps]

        return {
            'input_dense': torch.from_numpy(input_dense),
            'output_sparse': torch.from_numpy(output_sparse),
            'output_dense': torch.from_numpy(output_dense),
            'sparse_z_indices': torch.tensor(self.sparse_z_indices),
            'missing_z_indices': torch.tensor(self.missing_z_indices),
            'sample_start': idx,
            'global_timesteps': torch.tensor(timestamps_float, dtype=torch.float32)
        }


def create_jhu_dataloaders(data_dir, split_config_path, indicators=['u', 'v', 'w', 'p'],
                           sparse_z_interval=None, batch_size=1, num_workers=2,
                           normalize=True, norm_stats=None, dense_z_size=128):
    """Create JHU data train/val/test dataloaders"""
    if sparse_z_interval is None:
        raise ValueError("sparse_z_interval must be explicitly provided!")

    train_dataset = SparseJHUDataset(
        data_dir=data_dir, split='train', split_config_path=split_config_path,
        indicators=indicators, sparse_z_interval=sparse_z_interval,
        normalize=normalize, norm_stats=norm_stats, dense_z_size=dense_z_size
    )

    val_dataset = SparseJHUDataset(
        data_dir=data_dir, split='val', split_config_path=split_config_path,
        indicators=indicators, sparse_z_interval=sparse_z_interval,
        normalize=normalize, norm_stats=norm_stats, dense_z_size=dense_z_size
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader

print("JHU Dataset loaded.")

"""Training Functions"""

def compute_missing_indices(sparse_indices_z, total_z, device):
    """Compute missing z layer indices"""
    if torch.is_tensor(sparse_indices_z):
        if sparse_indices_z.dim() > 1:
            sparse_indices_z = sparse_indices_z[0]
        sparse_list = sparse_indices_z.tolist()
    else:
        if isinstance(sparse_indices_z, (list, tuple)) and len(sparse_indices_z) > 0:
            if torch.is_tensor(sparse_indices_z[0]):
                sparse_list = sparse_indices_z[0].tolist()
            elif isinstance(sparse_indices_z[0], (list, tuple)):
                sparse_list = list(sparse_indices_z[0])
            else:
                sparse_list = list(sparse_indices_z)
        else:
            sparse_list = list(sparse_indices_z)

    sparse_set = set(sparse_list)
    missing = [z for z in range(total_z) if z not in sparse_set]
    return torch.tensor(missing, device=device)


def train_epoch(model, loader, criterion, optimizer, scaler, device, epoch, total_z):
    """
    Train one epoch
    
    Args:
        model: SwinTransformerPhysics model
        loader: Training dataloader
        criterion: LossFactory
        optimizer: AdamW optimizer
        scaler: AMP GradScaler
        device: Compute device
        epoch: Current epoch
        total_z: Total z layers (128)
    """
    model.train()
    total_loss = 0.0
    loss_components = {}

    pbar = tqdm(loader, desc=f'Epoch {epoch} [Train]')

    for batch in pbar:
        input_dense = batch['input_dense'].to(device)
        output_dense = batch['output_dense'].to(device)
        output_sparse = batch['output_sparse'].to(device)
        sparse_z = batch['sparse_z_indices']

        B = input_dense.shape[0]

        # Prepare sparse indices for model
        sparse_z_tensor = sparse_z[0] if isinstance(sparse_z, list) else sparse_z
        if torch.is_tensor(sparse_z_tensor):
            sparse_z_tensor = sparse_z_tensor.to(device)

        model_batch = {
            'input_dense': input_dense,
            'output_sparse_z': output_sparse,
            'sparse_indices': {'z': sparse_z_tensor}
        }

        missing_indices = compute_missing_indices(sparse_z_tensor, total_z, device)
        missing_indices = missing_indices.unsqueeze(0).expand(B, -1)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            predictions = model(model_batch)
            pred_full = predictions['output_full']

            loss_dict = criterion.compute(
                pred=pred_full,
                target=output_dense,
                missing_indices=missing_indices,
                pred_full=pred_full,
                target_full=output_dense
            )

            loss = loss_dict['total']

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        for key, val in loss_dict.items():
            if key != 'total':
                if key not in loss_components:
                    loss_components[key] = 0.0
                loss_components[key] += val.item() if torch.is_tensor(val) else val

        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    avg_loss = total_loss / len(loader)
    for key in loss_components:
        loss_components[key] /= len(loader)

    return avg_loss, loss_components


@torch.no_grad()
def validate(model, loader, criterion, device, epoch, total_z):
    """Validate"""
    model.eval()
    total_loss = 0.0
    loss_components = {}

    for batch in tqdm(loader, desc=f'Epoch {epoch} [Val]'):
        input_dense = batch['input_dense'].to(device)
        output_dense = batch['output_dense'].to(device)
        output_sparse = batch['output_sparse'].to(device)
        sparse_z = batch['sparse_z_indices']

        B = input_dense.shape[0]

        sparse_z_tensor = sparse_z[0] if isinstance(sparse_z, list) else sparse_z
        if torch.is_tensor(sparse_z_tensor):
            sparse_z_tensor = sparse_z_tensor.to(device)

        model_batch = {
            'input_dense': input_dense,
            'output_sparse_z': output_sparse,
            'sparse_indices': {'z': sparse_z_tensor}
        }

        missing_indices = compute_missing_indices(sparse_z_tensor, total_z, device)
        missing_indices = missing_indices.unsqueeze(0).expand(B, -1)

        with torch.amp.autocast('cuda'):
            predictions = model(model_batch)
            pred_full = predictions['output_full']

            loss_dict = criterion.compute(
                pred=pred_full,
                target=output_dense,
                missing_indices=missing_indices,
                pred_full=pred_full,
                target_full=output_dense
            )

            loss = loss_dict['total']

        total_loss += loss.item()

        for key, val in loss_dict.items():
            if key != 'total':
                if key not in loss_components:
                    loss_components[key] = 0.0
                loss_components[key] += val.item() if torch.is_tensor(val) else val

    avg_loss = total_loss / len(loader)
    for key in loss_components:
        loss_components[key] /= len(loader)

    return avg_loss, loss_components


print("Training functions loaded.")

"""Main Training Script - Configure paths and run training"""

# ================================================================================
# ENVIRONMENT SWITCH - Set to 'local' or 'colab'
# ================================================================================
ENVIRONMENT = 'local'  # 'local' or 'colab'

# ================================================================================
# PATH CONFIGURATION
# ================================================================================
if ENVIRONMENT == 'local':
    # Local machine paths
    DATA_DIR = '/media/ydai17/T7 Shield/JHU_data/DNS128'
    SPLIT_CONFIG_PATH = '/home/ydai17/Turbo/Code_autoregressive/Dataset/jhu_data_splits_ar.json'
    CHECKPOINT_DIR = '/home/ydai17/Turbo/Code_autoregressive/Checkpoints/swin_spectral_test'
    NUM_WORKERS = 2
else:
    # Google Colab paths (Google Drive)
    DATA_DIR = '/content/drive/MyDrive/JHU_data/DNS128'
    SPLIT_CONFIG_PATH = '/content/drive/MyDrive/JHU_data/jhu_data_splits.json'
    CHECKPOINT_DIR = '/content/drive/MyDrive/checkpoints_swin_spectral'
    NUM_WORKERS = 2  # Colab has limited workers

# ================================================================================
# TRAINING HYPERPARAMETERS
# ================================================================================
EPOCHS = 2
BATCH_SIZE = 1
LEARNING_RATE = 2e-4
SPARSE_Z_INTERVAL = 8  # Every 8 layers = 16 sparse layers for 128³ grid

# ================================================================================
# NORMALIZATION STATS (Pre-computed from JHU data)
# ================================================================================
NORM_STATS = {
    'u': {'mean': 0.0008644619867776491, 'std': 0.6385110885344759},
    'v': {'mean': 0.0012105473293069805, 'std': 0.7478282257339498},
    'w': {'mean': 0.0021110948628738713, 'std': 0.6766117184108378},
    'p': {'mean': 0.00037188583444958443, 'std': 0.41183318615742287}
}

print(f"Environment: {ENVIRONMENT}")
print(f"Data directory: {DATA_DIR}")
print(f"Split config: {SPLIT_CONFIG_PATH}")
print(f"Checkpoint directory: {CHECKPOINT_DIR}")

# ================================================================================
# MAIN TRAINING FUNCTION
# ================================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "=" * 80)
    print("Swin Transformer Spectral Training")
    print("=" * 80)
    print(f"  Device: {device}")
    print(f"  Data directory: {DATA_DIR}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Sparse z interval: {SPARSE_Z_INTERVAL}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(CHECKPOINT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Use best spectral configuration
    config = SwinJHUConfigSpectral()
    total_z = 128
    channels = ['u', 'v', 'w', 'p']
    
    # Create model
    print("\nCreating model...")
    model = SwinTransformerPhysics(config).to(device)
    print(f"  Parameters: {model.get_num_params() / 1e6:.2f}M")
    print(f"  Memory reduction factor: {config.memory_reduction_factor}x")
    
    # Create dataloaders
    print("\nCreating datasets...")
    train_loader, val_loader = create_jhu_dataloaders(
        data_dir=DATA_DIR,
        split_config_path=SPLIT_CONFIG_PATH,
        indicators=channels,
        sparse_z_interval=SPARSE_Z_INTERVAL,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        normalize=True,
        norm_stats=NORM_STATS,
        dense_z_size=total_z
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Create loss factory with normalization params for physics loss
    norm_mean = [NORM_STATS[ch]['mean'] for ch in channels]
    norm_std = [NORM_STATS[ch]['std'] for ch in channels]
    
    print(f"\nPhysics loss normalization:")
    print(f"  norm_mean: {norm_mean}")
    print(f"  norm_std:  {norm_std}")
    
    criterion = build_loss_factory(config, norm_mean=norm_mean, norm_std=norm_std)
    
    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')
    
    # Training log
    log_file = open(checkpoint_dir / 'training_log.csv', 'w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        'epoch', 'train_loss', 'val_loss', 
        'train_data', 'train_div', 'train_ns', 'train_grad',
        'val_data', 'val_div', 'val_ns', 'val_grad',
        'train_data_low', 'train_data_mid', 'train_data_high',
        'val_data_low', 'val_data_mid', 'val_data_high',
        'lr', 'best_val_loss'
    ])
    
    # Save config
    config_dict = {
        'data_dir': DATA_DIR,
        'split_config': SPLIT_CONFIG_PATH,
        'checkpoint_dir': CHECKPOINT_DIR,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'lr': LEARNING_RATE,
        'sparse_z_interval': SPARSE_Z_INTERVAL,
        'model_params': model.get_num_params(),
        'loss_config': config.LOSS_CONFIG,
        'norm_stats': NORM_STATS
    }
    with open(checkpoint_dir / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    best_val_loss = float('inf')
    best_epoch = 0
    
    print("\nStarting training...")
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        
        # Train
        train_loss, train_components = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, total_z
        )
        
        # Validate
        val_loss, val_components = validate(
            model, val_loader, criterion, device, epoch, total_z
        )
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Print results
        print(f"  Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        print(f"  Train - data: {train_components.get('data', 0):.4f}, "
              f"div: {train_components.get('divergence', 0):.4f}, "
              f"ns: {train_components.get('navier_stokes', 0):.4f}, "
              f"grad: {train_components.get('gradient', 0):.4f}")
        
        # Print frequency band loss
        if train_components.get('data_low', 0) > 0:
            print(f"  Train freq - low: {train_components.get('data_low', 0):.4f}, "
                  f"mid: {train_components.get('data_mid', 0):.4f}, "
                  f"high: {train_components.get('data_high', 0):.4f}")
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
        
        # Save latest
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
        }, checkpoint_dir / 'latest_model.pt')
        
        # Save best
        if is_best:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_val_loss': best_val_loss,
            }, checkpoint_dir / 'best_model.pt')
        
        # Write log
        log_row = [
            epoch, f'{train_loss:.6f}', f'{val_loss:.6f}',
            f'{train_components.get("data", 0):.6f}',
            f'{train_components.get("divergence", 0):.6f}',
            f'{train_components.get("navier_stokes", 0):.6f}',
            f'{train_components.get("gradient", 0):.6f}',
            f'{val_components.get("data", 0):.6f}',
            f'{val_components.get("divergence", 0):.6f}',
            f'{val_components.get("navier_stokes", 0):.6f}',
            f'{val_components.get("gradient", 0):.6f}',
            f'{train_components.get("data_low", 0):.6f}',
            f'{train_components.get("data_mid", 0):.6f}',
            f'{train_components.get("data_high", 0):.6f}',
            f'{val_components.get("data_low", 0):.6f}',
            f'{val_components.get("data_mid", 0):.6f}',
            f'{val_components.get("data_high", 0):.6f}',
            f'{current_lr:.2e}', f'{best_val_loss:.6f}'
        ]
        log_writer.writerow(log_row)
        log_file.flush()
    
    log_file.close()
    
    print("\n" + "=" * 80)
    print("Training Completed!")
    print("=" * 80)
    print(f"  Best Val Loss: {best_val_loss:.6f} (Epoch {best_epoch})")
    print(f"  Checkpoints saved to: {checkpoint_dir}")


# Run training
if __name__ == "__main__":
    main()

# Run training
main()