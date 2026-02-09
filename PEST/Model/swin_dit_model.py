"""
Swin-DiT: 3D Swin Transformer for Diffusion-based Sparse-to-Dense Reconstruction

Combines:
- 3D Swin Transformer (window attention for efficiency)
- DiT-style time conditioning (adaLN-Zero)
- Inpainting-style diffusion for sparse GT preservation

This is an independent implementation, not dependent on existing Swin or PDE-Refiner code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math
from typing import Optional, Tuple, List


# ============================================================================
# Time Embedding
# ============================================================================

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


# ============================================================================
# 3D Window Attention
# ============================================================================

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

        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads)
        )

        # Get relative position index
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


# ============================================================================
# Swin-DiT Block with adaLN-Zero
# ============================================================================

class Mlp(nn.Module):
    """MLP as used in Vision Transformer."""

    def __init__(self, in_features: int, hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None, drop: float = 0.):
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
    """
    Swin Transformer Block with DiT-style adaLN-Zero conditioning.

    Key features:
    - 3D window attention (local)
    - Shifted window for cross-window connection
    - adaLN-Zero for timestep conditioning
    """

    def __init__(self, dim: int, num_heads: int, window_size: Tuple[int, int, int] = (4, 4, 4),
                 shift_size: Tuple[int, int, int] = (0, 0, 0), mlp_ratio: float = 4.,
                 qkv_bias: bool = True, drop: float = 0., attn_drop: float = 0.,
                 time_dim: int = None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.time_dim = time_dim if time_dim is not None else dim

        # Layer norms (will be modulated by adaLN)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

        # Window attention
        self.attn = WindowAttention3D(
            dim, window_size=window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )

        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        # adaLN-Zero modulation: 6 * dim for (gamma1, beta1, alpha1, gamma2, beta2, alpha2)
        # Input is time_dim (fixed), output is 6 * dim (varies by stage)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.time_dim, 6 * dim)
        )

        # Initialize alpha to zero for residual gates
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor,
                D: int, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: (B, D*H*W, C) flattened 3D features
            t_emb: (B, C) timestep embedding
            D, H, W: spatial dimensions
        """
        B, L, C = x.shape

        # adaLN modulation
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(t_emb).chunk(6, dim=-1)

        # Reshape to 3D
        x_3d = x.view(B, D, H, W, C)

        # Cyclic shift for shifted window attention
        if any(s > 0 for s in self.shift_size):
            shifted_x = torch.roll(x_3d, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))
        else:
            shifted_x = x_3d

        # Partition windows
        x_windows = window_partition_3d(shifted_x, self.window_size)

        # Modulate and apply attention
        x_norm = self.norm1(x_windows)
        x_norm = x_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_windows = self.attn(x_norm)

        # Merge windows
        attn_windows = window_reverse_3d(attn_windows, self.window_size, D, H, W)

        # Reverse cyclic shift
        if any(s > 0 for s in self.shift_size):
            shifted_x = torch.roll(attn_windows, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3))
        else:
            shifted_x = attn_windows

        # Flatten back
        x_flat = shifted_x.view(B, L, C)

        # Residual with gate
        x = x + gate_msa.unsqueeze(1) * x_flat

        # MLP block with modulation
        x_norm2 = self.norm2(x)
        x_norm2 = x_norm2 * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm2)

        return x


# ============================================================================
# Patch Embedding and Unpatch
# ============================================================================

class PatchEmbed3D(nn.Module):
    """3D Patch Embedding."""

    def __init__(self, patch_size: Tuple[int, int, int] = (4, 4, 4),
                 in_chans: int = 4, embed_dim: int = 96):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int, int]:
        """
        Args:
            x: (B, C, D, H, W)
        Returns:
            x: (B, D'*H'*W', embed_dim)
            D', H', W': patch grid dimensions
        """
        x = self.proj(x)  # (B, embed_dim, D', H', W')
        D, H, W = x.shape[2], x.shape[3], x.shape[4]
        x = x.flatten(2).transpose(1, 2)  # (B, D'*H'*W', embed_dim)
        x = self.norm(x)
        return x, D, H, W


class PatchUnEmbed3D(nn.Module):
    """3D Patch Un-embedding (reverse of patch embed)."""

    def __init__(self, patch_size: Tuple[int, int, int] = (4, 4, 4),
                 embed_dim: int = 96, out_chans: int = 4):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.ConvTranspose3d(embed_dim, out_chans, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor, D: int, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: (B, D*H*W, embed_dim)
            D, H, W: patch grid dimensions
        Returns:
            x: (B, out_chans, D*patch_size, H*patch_size, W*patch_size)
        """
        B, L, C = x.shape
        x = x.transpose(1, 2).view(B, C, D, H, W)
        x = self.proj(x)
        return x


# ============================================================================
# Main Swin-DiT Model
# ============================================================================

class SwinDiT3D(nn.Module):
    """
    Swin-DiT for 3D Sparse-to-Dense Turbulence Reconstruction.

    Architecture (Flat DiT style - no downsampling):
    - 3D Patch Embedding
    - Stack of Swin-DiT blocks with alternating window shift
    - Timestep conditioning via adaLN-Zero
    - Final projection to output channels

    For diffusion, predicts v (v-prediction) or noise (epsilon-prediction).
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        input_size: Tuple[int, int, int] = (128, 128, 128),
        patch_size: Tuple[int, int, int] = (4, 4, 4),
        embed_dim: int = 96,
        depths: List[int] = [2, 2, 6, 2],
        num_heads: List[int] = [3, 6, 12, 24],
        window_size: Tuple[int, int, int] = (4, 4, 4),
        mlp_ratio: float = 4.,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        cond_channels: int = 4,  # Channels for sparse condition
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Total depth = sum of all stage depths (flat architecture)
        self.total_depth = sum(depths)
        # Use first stage's num_heads for all blocks
        self.num_heads = num_heads[0] if isinstance(num_heads, list) else num_heads

        # Patch embedding for noisy input
        self.patch_embed = PatchEmbed3D(patch_size, in_channels, embed_dim)

        # Patch embedding for condition (sparse input)
        self.cond_embed = PatchEmbed3D(patch_size, cond_channels, embed_dim)

        # Combine noisy input and condition
        self.combine = nn.Linear(embed_dim * 2, embed_dim)

        # Timestep embedding
        self.time_embed = TimestepEmbedder(embed_dim)

        # Calculate patch grid size
        self.patch_grid = (
            input_size[0] // patch_size[0],
            input_size[1] // patch_size[1],
            input_size[2] // patch_size[2]
        )

        # Build flat stack of Swin-DiT blocks
        self.blocks = nn.ModuleList()
        for i_block in range(self.total_depth):
            # Alternate between regular and shifted window
            shift_size = (0, 0, 0) if (i_block % 2 == 0) else (
                window_size[0] // 2, window_size[1] // 2, window_size[2] // 2
            )
            block = SwinDiTBlock(
                dim=embed_dim,
                num_heads=self.num_heads,
                window_size=window_size,
                shift_size=shift_size,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                time_dim=embed_dim,
            )
            self.blocks.append(block)

        # Final norm and projection
        self.final_norm = nn.LayerNorm(embed_dim)
        self.patch_unembed = PatchUnEmbed3D(patch_size, embed_dim, out_channels)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x_noisy: torch.Tensor, t: torch.Tensor,
                condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_noisy: (B, C, D, H, W) noisy input (or current denoised estimate)
            t: (B,) diffusion timestep
            condition: (B, C, D, H, W) sparse condition (sparse GT + mask info)

        Returns:
            v_pred: (B, C, D, H, W) v-prediction for diffusion
        """
        B = x_noisy.shape[0]

        # Embed timestep
        t_emb = self.time_embed(t)  # (B, embed_dim)

        # Patch embed noisy input and condition
        x, D, H, W = self.patch_embed(x_noisy)  # (B, L, embed_dim)
        cond, _, _, _ = self.cond_embed(condition)  # (B, L, embed_dim)

        # Combine noisy input and condition
        x = self.combine(torch.cat([x, cond], dim=-1))  # (B, L, embed_dim)

        # Apply all Swin-DiT blocks
        for block in self.blocks:
            x = block(x, t_emb, D, H, W)

        # Final norm and projection
        x = self.final_norm(x)
        v_pred = self.patch_unembed(x, D, H, W)

        return v_pred


# ============================================================================
# Diffusion Scheduler and Sampler
# ============================================================================

class DDPMScheduler:
    """Simple DDPM scheduler for training and sampling."""

    def __init__(self, num_train_timesteps: int = 1000, beta_start: float = 0.0001,
                 beta_end: float = 0.02, beta_schedule: str = "linear"):
        self.num_train_timesteps = num_train_timesteps

        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        elif beta_schedule == "cosine":
            # Cosine schedule from improved DDPM
            steps = num_train_timesteps + 1
            s = 0.008
            t = torch.linspace(0, num_train_timesteps, steps) / num_train_timesteps
            alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # For v-prediction
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        return self

    def add_noise(self, original: torch.Tensor, noise: torch.Tensor,
                  timesteps: torch.Tensor) -> torch.Tensor:
        """Add noise to data according to the diffusion schedule."""
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]

        # Reshape for broadcasting
        while len(sqrt_alpha_prod.shape) < len(original.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy = sqrt_alpha_prod * original + sqrt_one_minus_alpha_prod * noise
        return noisy

    def get_v_target(self, original: torch.Tensor, noise: torch.Tensor,
                     timesteps: torch.Tensor) -> torch.Tensor:
        """Get v-prediction target: v = sqrt(alpha) * noise - sqrt(1-alpha) * x"""
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]

        while len(sqrt_alpha_prod.shape) < len(original.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        v_target = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * original
        return v_target

    def predict_original(self, x_t: torch.Tensor, v_pred: torch.Tensor,
                        timesteps: torch.Tensor) -> torch.Tensor:
        """Predict x_0 from x_t and v prediction."""
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]

        while len(sqrt_alpha_prod.shape) < len(x_t.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        x_0_pred = sqrt_alpha_prod * x_t - sqrt_one_minus_alpha_prod * v_pred
        return x_0_pred


class SwinDiTSampler:
    """
    Sampler for Swin-DiT with inpainting support.

    Uses DDPM sampling with sparse GT injection at each step.
    """

    def __init__(self, model: SwinDiT3D, scheduler: DDPMScheduler,
                 num_inference_steps: int = 50):
        self.model = model
        self.scheduler = scheduler
        self.num_inference_steps = num_inference_steps

        # Create inference timesteps (evenly spaced)
        step_ratio = scheduler.num_train_timesteps // num_inference_steps
        self.timesteps = torch.arange(0, scheduler.num_train_timesteps, step_ratio)
        self.timesteps = torch.flip(self.timesteps, [0])  # Reverse for denoising

    @torch.no_grad()
    def sample(self, sparse_gt: torch.Tensor, sparse_mask: torch.Tensor,
               condition: torch.Tensor = None) -> torch.Tensor:
        """
        Sample dense field from sparse GT using inpainting-style diffusion.

        Args:
            sparse_gt: (B, C, D, H, W) sparse ground truth values
            sparse_mask: (B, 1, D, H, W) mask where 1 = known, 0 = unknown
            condition: (B, C, D, H, W) optional additional condition

        Returns:
            sample: (B, C, D, H, W) denoised dense field
        """
        device = sparse_gt.device
        B, C, D, H, W = sparse_gt.shape

        # Prepare condition (sparse GT concatenated with mask)
        if condition is None:
            condition = torch.cat([sparse_gt, sparse_mask.expand(-1, C, -1, -1, -1)], dim=1)
            # Adjust model if needed, or just use sparse_gt as condition
            condition = sparse_gt  # Simplified: use sparse_gt directly

        # Start from random noise
        x_t = torch.randn(B, C, D, H, W, device=device)

        # Inject sparse GT into initial noise
        x_t = sparse_mask * sparse_gt + (1 - sparse_mask) * x_t

        # Denoising loop
        for i, t in enumerate(self.timesteps):
            t_batch = torch.full((B,), t.item(), device=device, dtype=torch.long)

            # Model prediction
            v_pred = self.model(x_t, t_batch, condition)

            # Predict x_0
            x_0_pred = self.scheduler.predict_original(x_t, v_pred, t_batch)

            # DDPM step: x_{t-1} = ...
            if i < len(self.timesteps) - 1:
                t_prev = self.timesteps[i + 1]
                t_prev_batch = torch.full((B,), t_prev.item(), device=device, dtype=torch.long)

                # Get noise for stochastic sampling
                noise = torch.randn_like(x_t)

                # Compute x_{t-1}
                alpha_t = self.scheduler.alphas_cumprod[t]
                alpha_t_prev = self.scheduler.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)

                # Simplified DDPM step
                beta_t = 1 - alpha_t / alpha_t_prev
                x_t_prev = torch.sqrt(alpha_t_prev) * x_0_pred + torch.sqrt(1 - alpha_t_prev) * noise

                # Inpainting: inject sparse GT (with noise at level t_prev, not t!)
                sparse_noisy = self.scheduler.add_noise(sparse_gt, noise, t_prev_batch)
                x_t = sparse_mask * sparse_noisy + (1 - sparse_mask) * x_t_prev
            else:
                # Final step: use predicted x_0 directly
                x_t = x_0_pred

        # Final injection of sparse GT
        x_t = sparse_mask * sparse_gt + (1 - sparse_mask) * x_t

        return x_t


# ============================================================================
# Model Configurations
# ============================================================================

def swin_dit_tiny(**kwargs) -> SwinDiT3D:
    """Tiny Swin-DiT (~15M params)."""
    return SwinDiT3D(
        embed_dim=64,
        depths=[2, 2, 4, 2],
        num_heads=[2, 4, 8, 16],
        window_size=(4, 4, 4),
        **kwargs
    )


def swin_dit_small(**kwargs) -> SwinDiT3D:
    """Small Swin-DiT (~50M params)."""
    return SwinDiT3D(
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=(4, 4, 4),
        **kwargs
    )


def swin_dit_base(**kwargs) -> SwinDiT3D:
    """Base Swin-DiT (~100M params)."""
    return SwinDiT3D(
        embed_dim=128,
        depths=[2, 2, 8, 2],
        num_heads=[4, 8, 16, 32],
        window_size=(4, 4, 4),
        **kwargs
    )


if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    model = swin_dit_tiny(
        in_channels=4,
        out_channels=4,
        input_size=(64, 64, 64),  # Smaller for testing
        patch_size=(4, 4, 4),
    ).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params / 1e6:.2f}M")

    # Test forward pass
    B = 2
    x_noisy = torch.randn(B, 4, 64, 64, 64).to(device)
    t = torch.randint(0, 1000, (B,)).to(device)
    condition = torch.randn(B, 4, 64, 64, 64).to(device)

    with torch.no_grad():
        v_pred = model(x_noisy, t, condition)

    print(f"Input shape: {x_noisy.shape}")
    print(f"Output shape: {v_pred.shape}")
    print("Forward pass successful!")
