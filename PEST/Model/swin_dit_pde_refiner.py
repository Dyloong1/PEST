"""
Swin-DiT with PDE-Refiner Configuration

Combines:
- Swin Transformer backbone (from Swin-DiT)
- PDE-Refiner's diffusion setup:
  - Exponential sigma schedule (K=3 steps)
  - Sparse observation interpolation as prior
  - v-prediction objective

This addresses the performance gap by using PDE-Refiner's proven diffusion strategy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange
import math
from typing import Optional, Tuple, List


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timestep."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class TimestepEmbedder(nn.Module):
    """Embed diffusion timestep into a vector."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = SinusoidalPosEmb(self.frequency_embedding_size)(t)
        t_emb = self.mlp(t_freq)
        return t_emb


def window_partition_3d(x: torch.Tensor, window_size: Tuple[int, int, int]) -> torch.Tensor:
    """Partition 3D tensor into windows."""
    B, D, H, W, C = x.shape
    wd, wh, ww = window_size
    x = x.view(B, D // wd, wd, H // wh, wh, W // ww, ww, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    windows = windows.view(-1, wd * wh * ww, C)
    return windows


def window_reverse_3d(windows: torch.Tensor, window_size: Tuple[int, int, int],
                      D: int, H: int, W: int) -> torch.Tensor:
    """Reverse window partition."""
    wd, wh, ww = window_size
    B = int(windows.shape[0] / (D * H * W / (wd * wh * ww)))
    x = windows.view(B, D // wd, H // wh, W // ww, wd, wh, ww, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    x = x.view(B, D, H, W, -1)
    return x


class WindowAttention3D(nn.Module):
    """3D Window-based Multi-head Self-Attention."""

    def __init__(self, dim: int, window_size: Tuple[int, int, int], num_heads: int,
                 qkv_bias: bool = True, attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads)
        )

        coords_d = torch.arange(window_size[0])
        coords_h = torch.arange(window_size[1])
        coords_w = torch.arange(window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing='ij'))
        coords_flatten = coords.flatten(1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 2] += window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * window_size[2] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """MLP with GELU activation."""

    def __init__(self, in_features: int, hidden_features: int = None, out_features: int = None, drop: float = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwinDiTBlock(nn.Module):
    """Swin Transformer block with DiT-style adaLN-Zero conditioning."""

    def __init__(self, dim: int, num_heads: int, window_size: Tuple[int, int, int],
                 shift_size: Tuple[int, int, int] = (0, 0, 0), mlp_ratio: float = 4.,
                 drop: float = 0., attn_drop: float = 0., time_dim: int = None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim or dim, 6 * dim)
        )

        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = WindowAttention3D(dim, window_size, num_heads, attn_drop=attn_drop, proj_drop=drop)

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, D: int, H: int, W: int) -> torch.Tensor:
        B, L, C = x.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(t_emb).chunk(6, dim=-1)

        x_3d = x.view(B, D, H, W, C)

        if any(s > 0 for s in self.shift_size):
            shifted_x = torch.roll(x_3d, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]),
                                   dims=(1, 2, 3))
        else:
            shifted_x = x_3d

        x_windows = window_partition_3d(shifted_x, self.window_size)

        x_norm = self.norm1(x_windows)
        x_norm = x_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out = self.attn(x_norm)
        x_windows = x_windows + gate_msa.unsqueeze(1) * attn_out

        shifted_x = window_reverse_3d(x_windows, self.window_size, D, H, W)

        if any(s > 0 for s in self.shift_size):
            x_3d = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]),
                              dims=(1, 2, 3))
        else:
            x_3d = shifted_x

        x = x_3d.view(B, L, C)

        x_norm = self.norm2(x)
        x_norm = x_norm * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)

        return x


class PatchEmbed3D(nn.Module):
    """3D Patch Embedding using Conv3d."""

    def __init__(self, patch_size: Tuple[int, int, int] = (4, 4, 4),
                 in_chans: int = 4, embed_dim: int = 96):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int, int]:
        B, C, D, H, W = x.shape
        x = self.proj(x)
        D, H, W = x.shape[2], x.shape[3], x.shape[4]
        x = x.flatten(2).transpose(1, 2)
        return x, D, H, W


class PatchUnEmbed3D(nn.Module):
    """3D Patch Un-embedding."""

    def __init__(self, patch_size: Tuple[int, int, int] = (4, 4, 4),
                 embed_dim: int = 96, out_chans: int = 4):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.ConvTranspose3d(embed_dim, out_chans, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor, D: int, H: int, W: int) -> torch.Tensor:
        B, L, C = x.shape
        x = x.transpose(1, 2).view(B, C, D, H, W)
        x = self.proj(x)
        return x


class PDERefinerScheduler:
    """
    PDE-Refiner's exponential sigma schedule.

    Key features:
    - Only K=3 refinement steps (not 1000!)
    - Exponential sigma decay: σ_k = σ_min^(k/K)
    - Very small final noise (σ_min = 4e-7)
    """

    def __init__(self, num_refine_steps: int = 3, min_noise_std: float = 4e-7,
                 prediction_type: str = "v_prediction"):
        self.num_train_timesteps = num_refine_steps + 1
        self.num_refine_steps = num_refine_steps
        self.min_noise_std = min_noise_std
        self.prediction_type = prediction_type

        self.sigmas = torch.tensor([
            min_noise_std ** (k / num_refine_steps)
            for k in range(num_refine_steps + 1)
        ], dtype=torch.float32)

        self.alphas_cumprod = 1.0 - self.sigmas ** 2
        self.alphas_cumprod = torch.clamp(self.alphas_cumprod, min=1e-10, max=1.0)

        self.timesteps = torch.arange(0, num_refine_steps)

        self.config = type('Config', (), {
            'num_train_timesteps': self.num_train_timesteps,
            'prediction_type': prediction_type
        })()

    def to(self, device):
        self.sigmas = self.sigmas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.timesteps = self.timesteps.to(device)
        return self

    def add_noise(self, original: torch.Tensor, noise: torch.Tensor,
                  timesteps: torch.Tensor) -> torch.Tensor:
        """Add noise according to the schedule."""
        device = original.device
        alphas_cumprod = self.alphas_cumprod.to(device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5

        while len(sqrt_alpha_prod.shape) < len(original.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        return sqrt_alpha_prod * original + sqrt_one_minus_alpha_prod * noise

    def get_v_target(self, original: torch.Tensor, noise: torch.Tensor,
                     timesteps: torch.Tensor) -> torch.Tensor:
        """Get v-prediction target: v = sqrt(α) * noise - sqrt(1-α) * x"""
        device = original.device
        alphas_cumprod = self.alphas_cumprod.to(device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5

        while len(sqrt_alpha_prod.shape) < len(original.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        return sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * original

    def step(self, model_output: torch.Tensor, timestep: int,
             sample: torch.Tensor) -> torch.Tensor:
        """Perform one denoising step."""
        device = sample.device
        t = timestep

        alpha_prod_t = self.alphas_cumprod[t].to(device)
        alpha_prod_t_prev = self.alphas_cumprod[t + 1].to(device) if t < self.num_refine_steps else torch.tensor(1.0, device=device)

        sqrt_alpha_prod = alpha_prod_t ** 0.5
        sqrt_one_minus_alpha_prod = (1 - alpha_prod_t) ** 0.5

        pred_original = sqrt_alpha_prod * sample - sqrt_one_minus_alpha_prod * model_output

        sqrt_alpha_prod_prev = alpha_prod_t_prev ** 0.5
        pred_prev = sqrt_alpha_prod_prev * pred_original

        return pred_prev

    def get_original_from_v(self, v_pred: torch.Tensor, noised_sample: torch.Tensor,
                             timesteps: torch.Tensor) -> torch.Tensor:
        """
        Recover clean signal from v-prediction.

        v = sqrt(α) * noise - sqrt(1-α) * x_0
        noised = sqrt(α) * x_0 + sqrt(1-α) * noise

        From these two equations:
        x_0 = sqrt(α) * noised - sqrt(1-α) * v
        """
        device = noised_sample.device
        alphas_cumprod = self.alphas_cumprod.to(device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5

        while len(sqrt_alpha_prod.shape) < len(noised_sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        pred_original = sqrt_alpha_prod * noised_sample - sqrt_one_minus_alpha_prod * v_pred
        return pred_original


class SwinDiTPDERefiner(nn.Module):
    """
    Swin-DiT backbone with PDE-Refiner's diffusion configuration.

    Key changes from original Swin-DiT:
    1. Uses PDERefinerScheduler (K=3 steps, exponential sigma)
    2. Sparse observation interpolation as conditioning prior
    3. Condition = input_history + sparse_interpolated (like PDE-Refiner)
    4. Full-field loss (not just missing layers)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.t_in = getattr(config, 'INPUT_TIMESTEPS', 5)
        self.t_out = getattr(config, 'OUTPUT_TIMESTEPS', 5)
        self.in_channels = len(getattr(config, 'INDICATORS', ['u', 'v', 'w', 'p']))
        self.total_z = getattr(config, 'TOTAL_Z_LAYERS', 128)

        self.num_refine_steps = getattr(config, 'REFINER_STEPS', 3)
        self.min_noise_std = getattr(config, 'MIN_NOISE_STD', 4e-7)

        self.time_multiplier = 1000.0 / (self.num_refine_steps + 1)

        self.cond_channels = self.t_in * self.in_channels
        self.out_channels = self.t_out * self.in_channels

        self.scheduler = PDERefinerScheduler(
            num_refine_steps=self.num_refine_steps,
            min_noise_std=self.min_noise_std,
            prediction_type="v_prediction"
        )

        embed_dim = getattr(config, 'EMBED_DIM', 96)
        depths = getattr(config, 'DEPTHS', [2, 2, 6, 2])
        num_heads = getattr(config, 'NUM_HEADS', [3, 6, 12, 24])
        window_size = getattr(config, 'WINDOW_SIZE', (4, 4, 4))
        patch_size = getattr(config, 'PATCH_SIZE', (4, 4, 4))
        mlp_ratio = getattr(config, 'MLP_RATIO', 4.0)
        drop_rate = getattr(config, 'DROP_RATE', 0.0)
        attn_drop_rate = getattr(config, 'ATTN_DROP_RATE', 0.0)

        self.embed_dim = embed_dim
        self.total_depth = sum(depths)
        self.num_heads_first = num_heads[0] if isinstance(num_heads, list) else num_heads

        self.patch_embed = PatchEmbed3D(patch_size, self.out_channels, embed_dim)

        self.cond_embed = PatchEmbed3D(patch_size, self.cond_channels * 2, embed_dim)

        self.combine = nn.Linear(embed_dim * 2, embed_dim)

        self.time_embed = TimestepEmbedder(embed_dim)

        input_size = (self.total_z, 128, 128)
        self.patch_grid = (
            input_size[0] // patch_size[0],
            input_size[1] // patch_size[1],
            input_size[2] // patch_size[2]
        )

        self.blocks = nn.ModuleList()
        for i_block in range(self.total_depth):
            shift_size = (0, 0, 0) if (i_block % 2 == 0) else (
                window_size[0] // 2, window_size[1] // 2, window_size[2] // 2
            )
            block = SwinDiTBlock(
                dim=embed_dim,
                num_heads=self.num_heads_first,
                window_size=window_size,
                shift_size=shift_size,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                time_dim=embed_dim,
            )
            self.blocks.append(block)

        self.final_norm = nn.LayerNorm(embed_dim)
        self.patch_unembed = PatchUnEmbed3D(patch_size, embed_dim, self.out_channels)

        hidden_channels = getattr(config, 'REFINER_HIDDEN', 64)
        self.sparse_encoder = nn.Sequential(
            nn.Conv3d(self.in_channels, hidden_channels // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden_channels // 2, self.cond_channels, kernel_size=1)
        )

        self._init_weights()
        self._print_info()

    def _init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
        self.apply(_init)

    def _print_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n[Swin-DiT + PDE-Refiner Config] Model initialized:")
        print(f"  Input: ({self.t_in}, {self.in_channels}, Z, H, W)")
        print(f"  Output: ({self.t_out}, {self.in_channels}, Z, H, W)")
        print(f"  Refinement steps: {self.num_refine_steps}")
        print(f"  Min noise std: {self.min_noise_std}")
        print(f"  Embed dim: {self.embed_dim}")
        print(f"  Total depth: {self.total_depth}")
        print(f"  Total parameters: {total_params:,}")

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing to save memory."""
        self._use_gradient_checkpointing = True
        print(f"  Gradient checkpointing enabled for {len(self.blocks)} blocks")

    def _interpolate_sparse_to_full(self, sparse_obs: torch.Tensor,
                                     sparse_indices: torch.Tensor,
                                     total_z: int) -> torch.Tensor:
        """
        Interpolate sparse observations to full Z resolution.
        Same as PDE-Refiner's implementation.
        """
        B, T, C, n_sparse, H, W = sparse_obs.shape
        device = sparse_obs.device
        dtype = sparse_obs.dtype

        output = torch.zeros(B, T, C, total_z, H, W, device=device, dtype=dtype)

        if isinstance(sparse_indices, torch.Tensor):
            sparse_z = sparse_indices[0].float()
        else:
            sparse_z = torch.tensor(sparse_indices, device=device, dtype=torch.float32)

        for z in range(total_z):
            z_float = float(z)
            right_idx = 0
            for i, pos in enumerate(sparse_z):
                if pos >= z_float:
                    right_idx = i
                    break
                right_idx = i + 1

            if right_idx == 0:
                output[:, :, :, z, :, :] = sparse_obs[:, :, :, 0, :, :]
            elif right_idx >= n_sparse:
                output[:, :, :, z, :, :] = sparse_obs[:, :, :, -1, :, :]
            else:
                left_idx = right_idx - 1
                left_pos = sparse_z[left_idx].item()
                right_pos = sparse_z[right_idx].item()
                weight = (z_float - left_pos) / (right_pos - left_pos) if right_pos != left_pos else 0.5
                output[:, :, :, z, :, :] = (1 - weight) * sparse_obs[:, :, :, left_idx, :, :] + \
                                           weight * sparse_obs[:, :, :, right_idx, :, :]

        return output

    def _forward_model(self, y_noised: torch.Tensor, condition: torch.Tensor,
                       timestep: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Swin Transformer.

        Args:
            y_noised: (B, T*C, Z, H, W) noised output
            condition: (B, 2*T*C, Z, H, W) condition (history + sparse_interp)
            timestep: (B,) scaled timestep

        Returns:
            v_pred: (B, T*C, Z, H, W)
        """
        B = y_noised.shape[0]

        t_emb = self.time_embed(timestep)

        x, D, H, W = self.patch_embed(y_noised)
        cond, _, _, _ = self.cond_embed(condition)

        x = self.combine(torch.cat([x, cond], dim=-1))

        use_ckpt = getattr(self, '_use_gradient_checkpointing', False) and self.training
        for block in self.blocks:
            if use_ckpt:
                x = checkpoint(block, x, t_emb, D, H, W, use_reentrant=False)
            else:
                x = block(x, t_emb, D, H, W)

        x = self.final_norm(x)
        v_pred = self.patch_unembed(x, D, H, W)

        return v_pred

    def get_training_loss(self, input_dense: torch.Tensor, output_dense: torch.Tensor,
                          output_sparse: torch.Tensor, sparse_z_indices: torch.Tensor,
                          use_physics_loss: bool = False, physics_config: dict = None) -> torch.Tensor:
        """
        Compute training loss with v-prediction (same as PDE-Refiner).

        With optional physics loss at low noise stages (k >= 2).

        Args:
            input_dense: (B, T_in, C, Z, H, W) input history
            output_dense: (B, T_out, C, Z, H, W) ground truth output
            output_sparse: (B, T_out, C, n_sparse, H, W) sparse observations
            sparse_z_indices: (B, n_sparse) sparse layer indices
            use_physics_loss: Whether to add physics loss at low noise stages
            physics_config: Dict with physics loss configuration:
                - div_weight: Weight for divergence loss (default: 1e-4)
                - ns_weight: Weight for NS residual loss (default: 1e-5)
                - physics_threshold: Minimum k to apply physics loss (default: 2)
                - norm_mean, norm_std: Normalization stats for physics loss

        Returns:
            loss: scalar loss value
        """
        B, T_in, C, Z, H, W = input_dense.shape
        T_out = output_dense.shape[1]
        device = input_dense.device

        x = rearrange(input_dense, 'b t c z h w -> b (t c) z h w')
        y = rearrange(output_dense, 'b t c z h w -> b (t c) z h w')

        sparse_full = self._interpolate_sparse_to_full(output_sparse, sparse_z_indices, Z)
        sparse_interp = rearrange(sparse_full, 'b t c z h w -> b (t c) z h w')

        condition = torch.cat([x, sparse_interp], dim=1)

        k = torch.randint(0, self.num_refine_steps, (B,), device=device)

        noise = torch.randn_like(y)
        y_noised = self.scheduler.add_noise(y, noise, k)

        v_pred = self._forward_model(y_noised, condition, k.float() * self.time_multiplier)

        v_target = self.scheduler.get_v_target(y, noise, k)

        mse_loss = F.mse_loss(v_pred, v_target)

        if use_physics_loss and physics_config is not None:
            physics_threshold = physics_config.get('physics_threshold', 2)
            div_weight = physics_config.get('div_weight', 1e-4)
            ns_weight = physics_config.get('ns_weight', 1e-5)
            norm_mean = physics_config.get('norm_mean', None)
            norm_std = physics_config.get('norm_std', None)

            low_noise_mask = (k >= physics_threshold)

            if low_noise_mask.any():
                pred_original = self.scheduler.get_original_from_v(v_pred, y_noised, k)

                pred_6d = rearrange(pred_original, 'b (t c) z h w -> b t c z h w', t=T_out, c=C)

                pred_6d_filtered = pred_6d[low_noise_mask]
                k_filtered = k[low_noise_mask]

                physics_loss = torch.tensor(0.0, device=device)

                from Model.losses.physics_loss import (
                    spectral_divergence_3d_filtered,
                    navier_stokes_forced_turb
                )

                if div_weight > 0:
                    try:
                        div_loss = spectral_divergence_3d_filtered(
                            pred_6d_filtered, k_cutoff=12, normalize_output=True
                        )
                        k_factor = k_filtered.float() / self.num_refine_steps
                        div_loss_weighted = div_loss * k_factor.mean()
                        physics_loss = physics_loss + div_weight * div_loss_weighted
                    except Exception as e:
                        pass

                if ns_weight > 0 and norm_mean is not None and norm_std is not None:
                    try:
                        domain_size = 2 * 3.141592653589793
                        ns_loss = navier_stokes_forced_turb(
                            pred_6d_filtered,
                            nu=0.000185,
                            dx=domain_size / 128,
                            dy=domain_size / 128,
                            dz=domain_size / 128,
                            dt=0.0065,
                            forcing_k_cutoff=2.0,
                            norm_mean=norm_mean,
                            norm_std=norm_std
                        )
                        k_factor = k_filtered.float() / self.num_refine_steps
                        ns_loss_weighted = ns_loss * k_factor.mean()
                        physics_loss = physics_loss + ns_weight * ns_loss_weighted
                    except Exception as e:
                        pass

                return mse_loss + physics_loss

        return mse_loss

    def forward(self, input_dense: torch.Tensor, output_sparse: torch.Tensor,
                sparse_z_indices: torch.Tensor, missing_z_indices: torch.Tensor,
                global_timesteps: torch.Tensor = None) -> torch.Tensor:
        """
        Inference: K-step iterative denoising (same as PDE-Refiner).

        Args:
            input_dense: (B, T_in, C, Z, H, W)
            output_sparse: (B, T_out, C, n_sparse, H, W)
            sparse_z_indices: (B, n_sparse)
            missing_z_indices: (B, n_missing)

        Returns:
            output: (B, T_out, C, Z, H, W)
        """
        B, T_in, C, Z, H, W = input_dense.shape
        T_out = output_sparse.shape[1]
        device = input_dense.device

        x = rearrange(input_dense, 'b t c z h w -> b (t c) z h w')

        sparse_full = self._interpolate_sparse_to_full(output_sparse, sparse_z_indices, Z)
        sparse_interp = rearrange(sparse_full, 'b t c z h w -> b (t c) z h w')

        condition = torch.cat([x, sparse_interp], dim=1)

        y_noised = torch.randn(
            size=(B, self.out_channels, Z, H, W),
            dtype=input_dense.dtype,
            device=device
        )

        for k in self.scheduler.timesteps:
            k_tensor = torch.full((B,), k.item(), device=device, dtype=torch.long)

            v_pred = self._forward_model(y_noised, condition, k_tensor.float() * self.time_multiplier)

            y_noised = self.scheduler.step(v_pred, k.item(), y_noised)

        output = rearrange(y_noised, 'b (t c) z h w -> b t c z h w', t=T_out, c=C)

        for b in range(B):
            for i, z_idx in enumerate(sparse_z_indices[b]):
                output[b, :, :, z_idx, :, :] = output_sparse[b, :, :, i, :, :]

        return output


class SwinDiTPDERefinerConfig:
    """Configuration for Swin-DiT with PDE-Refiner setup."""

    def __init__(self):
        self.INDICATORS = ['u', 'v', 'w', 'p']
        self.INPUT_TIMESTEPS = 5
        self.OUTPUT_TIMESTEPS = 5
        self.TOTAL_Z_LAYERS = 128
        self.SPARSE_Z_LAYERS = 16

        self.REFINER_STEPS = 3
        self.MIN_NOISE_STD = 4e-7
        self.REFINER_HIDDEN = 64

        self.EMBED_DIM = 224
        self.DEPTHS = [2, 2, 8, 2]
        self.NUM_HEADS = [7, 7, 7, 7]
        self.WINDOW_SIZE = (4, 4, 4)
        self.PATCH_SIZE = (4, 4, 4)
        self.MLP_RATIO = 4.0
        self.DROP_RATE = 0.0
        self.ATTN_DROP_RATE = 0.0

        self.USE_PHYSICS_LOSS = False
        self.PHYSICS_THRESHOLD = 2
        self.DIV_WEIGHT = 1e-4
        self.NS_WEIGHT = 1e-5


def get_swin_dit_pde_refiner_config(name: str = 'default', data_source: str = 'jhu') -> SwinDiTPDERefinerConfig:
    """Get configuration by name and data source.

    Args:
        name: Model size ('small', 'default', 'large', 'xlarge', 'giant')
        data_source: 'jhu' (128x128x128) or 'dns' (64x128x128)
    """
    config = SwinDiTPDERefinerConfig()

    if data_source == 'dns':
        config.TOTAL_Z_LAYERS = 64
        config.SPARSE_Z_LAYERS = 8

    if name == 'small':
        config.EMBED_DIM = 160
        config.DEPTHS = [2, 2, 4, 2]
        config.NUM_HEADS = [5, 5, 5, 5]
        config.REFINER_HIDDEN = 48
    elif name == 'default':
        pass
    elif name == 'large':
        config.EMBED_DIM = 320
        config.DEPTHS = [2, 4, 12, 2]
        config.NUM_HEADS = [10, 10, 10, 10]
        config.REFINER_HIDDEN = 128
    elif name == 'xlarge':
        config.EMBED_DIM = 448
        config.DEPTHS = [2, 6, 18, 2]
        config.NUM_HEADS = [14, 14, 14, 14]
        config.REFINER_HIDDEN = 192
    elif name == 'giant':
        config.EMBED_DIM = 1152
        config.DEPTHS = [2, 11, 34, 2]
        config.NUM_HEADS = [36, 36, 36, 36]
        config.REFINER_HIDDEN = 1152

    return config


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = get_swin_dit_pde_refiner_config('default')
    model = SwinDiTPDERefiner(config).to(device)

    B = 1
    input_dense = torch.randn(B, 5, 4, 128, 128, 128).to(device)
    output_sparse = torch.randn(B, 5, 4, 16, 128, 128).to(device)
    sparse_z_indices = torch.arange(0, 128, 8).unsqueeze(0).expand(B, -1).to(device)
    missing_z_indices = torch.tensor([i for i in range(128) if i % 8 != 0]).unsqueeze(0).expand(B, -1).to(device)

    with torch.no_grad():
        output = model(input_dense, output_sparse, sparse_z_indices, missing_z_indices)

    print(f"Input shape: {input_dense.shape}")
    print(f"Output shape: {output.shape}")
    print("Forward pass successful!")
