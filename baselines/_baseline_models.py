
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from collections import defaultdict

DATA_SOURCE = 'JHU'

SPATIAL_SIZE = (128, 128, 128)
FNO_MODES = {'z': 12, 'h': 12, 'w': 12}
PATCH_TRANS = (8, 8, 8)
TFNO_MODES_Z = 16
PATCH_DPOT = (4, 8, 8)
class SpectralConv3d(nn.Module):
    """3D Spectral Convolution Layer."""

    def __init__(self, in_channels, out_channels, modes_z, modes_h, modes_w):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_z = modes_z
        self.modes_h = modes_h
        self.modes_w = modes_w

        self.scale = 1 / (in_channels * out_channels)

        # Complex weights for 4 corners of the frequency domain
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes_z, modes_h, modes_w, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes_z, modes_h, modes_w, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes_z, modes_h, modes_w, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes_z, modes_h, modes_w, dtype=torch.cfloat))

    def compl_mul3d(self, input, weights):
        # (B, C_in, Z, H, W), (C_in, C_out, Z, H, W) -> (B, C_out, Z, H, W)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        B = x.shape[0]

        # FFT
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # Allocate output
        out_ft = torch.zeros(B, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1,
                            dtype=torch.cfloat, device=x.device)

        # Multiply relevant modes
        out_ft[:, :, :self.modes_z, :self.modes_h, :self.modes_w] = \
            self.compl_mul3d(x_ft[:, :, :self.modes_z, :self.modes_h, :self.modes_w], self.weights1)
        out_ft[:, :, -self.modes_z:, :self.modes_h, :self.modes_w] = \
            self.compl_mul3d(x_ft[:, :, -self.modes_z:, :self.modes_h, :self.modes_w], self.weights2)
        out_ft[:, :, :self.modes_z, -self.modes_h:, :self.modes_w] = \
            self.compl_mul3d(x_ft[:, :, :self.modes_z, -self.modes_h:, :self.modes_w], self.weights3)
        out_ft[:, :, -self.modes_z:, -self.modes_h:, :self.modes_w] = \
            self.compl_mul3d(x_ft[:, :, -self.modes_z:, -self.modes_h:, :self.modes_w], self.weights4)

        # IFFT
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


class FNOBlock3d(nn.Module):
    """Single FNO Block: Spectral Conv + Skip Connection."""

    def __init__(self, width, modes_z, modes_h, modes_w):
        super().__init__()
        self.spectral_conv = SpectralConv3d(width, width, modes_z, modes_h, modes_w)
        self.conv = nn.Conv3d(width, width, 1)
        self.norm = nn.InstanceNorm3d(width)

    def forward(self, x):
        x1 = self.spectral_conv(x)
        x2 = self.conv(x)
        x = self.norm(x1 + x2)
        return F.gelu(x)


class FNO3D(nn.Module):
    """
    3D Fourier Neural Operator for temporal prediction.

    Input: (B, T_in, C, Z, H, W)
    Output: (B, T_out, C, Z, H, W)
    """

    def __init__(self, in_channels=4, out_channels=4,
                 input_timesteps=5, output_timesteps=5,
                 width=32, modes_z=16, modes_h=16, modes_w=16,
                 num_layers=4):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_timesteps = input_timesteps
        self.output_timesteps = output_timesteps
        self.width = width

        # Lifting: (T_in * C) -> width
        self.lift = nn.Conv3d(input_timesteps * in_channels, width, 1)

        # FNO Blocks
        self.blocks = nn.ModuleList([
            FNOBlock3d(width, modes_z, modes_h, modes_w)
            for _ in range(num_layers)
        ])

        # Projection: width -> (T_out * C)
        self.proj1 = nn.Conv3d(width, width * 2, 1)
        self.proj2 = nn.Conv3d(width * 2, output_timesteps * out_channels, 1)

    def forward(self, x):
        # x: (B, T_in, C, Z, H, W)
        B, T, C, Z, H, W = x.shape

        # Reshape: (B, T*C, Z, H, W)
        x = x.view(B, T * C, Z, H, W)

        # Lifting
        x = self.lift(x)

        # FNO Blocks
        for block in self.blocks:
            x = block(x)

        # Projection
        x = F.gelu(self.proj1(x))
        x = self.proj2(x)

        # Reshape: (B, T_out, C, Z, H, W)
        x = x.view(B, self.output_timesteps, self.out_channels, Z, H, W)

        return x

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# Test FNO3D

# Cell 14
# ============================================================
# U-FNO - U-Net Enhanced Fourier Neural Operator
# ============================================================

class UFNOEncoder(nn.Module):
    """Encoder block with spectral conv + downsampling."""

    def __init__(self, in_ch, out_ch, modes_z, modes_h, modes_w):
        super().__init__()
        self.spectral = SpectralConv3d(in_ch, out_ch, modes_z, modes_h, modes_w)
        self.conv = nn.Conv3d(in_ch, out_ch, 1)
        self.norm = nn.InstanceNorm3d(out_ch)
        self.down = nn.Conv3d(out_ch, out_ch, 2, stride=2)

    def forward(self, x):
        x1 = self.spectral(x)
        x2 = self.conv(x)
        x = F.gelu(self.norm(x1 + x2))
        skip = x
        x = self.down(x)
        return x, skip


class UFNODecoder(nn.Module):
    """Decoder block with spectral conv + upsampling."""

    def __init__(self, in_ch, out_ch, skip_ch, modes_z, modes_h, modes_w):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)
        # After concat: out_ch + skip_ch channels
        concat_ch = out_ch + skip_ch
        self.spectral = SpectralConv3d(concat_ch, out_ch, modes_z, modes_h, modes_w)
        self.conv = nn.Conv3d(concat_ch, out_ch, 1)
        self.norm = nn.InstanceNorm3d(out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x1 = self.spectral(x)
        x2 = self.conv(x)
        x = F.gelu(self.norm(x1 + x2))
        return x


class UFNO3D(nn.Module):
    """
    U-Net Enhanced Fourier Neural Operator.

    Input: (B, T_in, C, Z, H, W)
    Output: (B, T_out, C, Z, H, W)
    """

    def __init__(self, in_channels=4, out_channels=4,
                 input_timesteps=5, output_timesteps=5,
                 base_width=32, modes=12):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_timesteps = input_timesteps
        self.output_timesteps = output_timesteps

        w = base_width

        # Lifting
        self.lift = nn.Conv3d(input_timesteps * in_channels, w, 1)

        # Encoder (3 levels)
        # enc1: w -> w (skip1 has w channels)
        # enc2: w -> w*2 (skip2 has w*2 channels)
        # enc3: w*2 -> w*4 (skip3 has w*4 channels)
        self.enc1 = UFNOEncoder(w, w, modes, modes, modes)
        self.enc2 = UFNOEncoder(w, w*2, modes//2, modes//2, modes//2)
        self.enc3 = UFNOEncoder(w*2, w*4, modes//4, modes//4, modes//4)

        # Bottleneck
        self.bottleneck = FNOBlock3d(w*4, modes//4, modes//4, modes//4)

        # Decoder (skip_ch must match encoder output channels)
        # dec3: w*4 -> w*2, skip3 has w*4 channels
        # dec2: w*2 -> w, skip2 has w*2 channels
        # dec1: w -> w, skip1 has w channels
        self.dec3 = UFNODecoder(w*4, w*2, w*4, modes//4, modes//4, modes//4)
        self.dec2 = UFNODecoder(w*2, w, w*2, modes//2, modes//2, modes//2)
        self.dec1 = UFNODecoder(w, w, w, modes, modes, modes)

        # Projection
        self.proj = nn.Sequential(
            nn.Conv3d(w, w*2, 1),
            nn.GELU(),
            nn.Conv3d(w*2, output_timesteps * out_channels, 1)
        )

    def forward(self, x):
        # x: (B, T_in, C, Z, H, W)
        B, T, C, Z, H, W = x.shape

        # Reshape
        x = x.view(B, T * C, Z, H, W)

        # Lifting
        x = self.lift(x)

        # Encoder
        x, skip1 = self.enc1(x)   # skip1: w channels
        x, skip2 = self.enc2(x)   # skip2: w*2 channels
        x, skip3 = self.enc3(x)   # skip3: w*4 channels

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.dec3(x, skip3)   # w*4 + skip(w*4) -> w*2
        x = self.dec2(x, skip2)   # w*2 + skip(w*2) -> w
        x = self.dec1(x, skip1)   # w + skip(w) -> w

        # Projection
        x = self.proj(x)

        # Reshape
        x = x.view(B, self.output_timesteps, self.out_channels, Z, H, W)

        return x

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# Test U-FNO

# Cell 16
# ============================================================
# Transolver - Physics-Attention Transformer (ICML 2024)
# ============================================================

class PhysicsAttention(nn.Module):
    """
    Physics-Attention: Groups points into K slices and performs cross-attention.
    Instead of O(N²) attention, uses O(N*K) where K << N.
    """

    def __init__(self, dim, num_slices=16, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_slices = num_slices
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Learnable slice queries
        self.slice_tokens = nn.Parameter(torch.randn(num_slices, dim))

        # Projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        # Slice-to-point attention
        self.slice_q = nn.Linear(dim, dim)
        self.slice_k = nn.Linear(dim, dim)
        self.slice_v = nn.Linear(dim, dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (B, N, D) where N = Z*H*W
        B, N, D = x.shape

        # Point-to-slice attention: aggregate points into slices
        q = self.q_proj(self.slice_tokens).unsqueeze(0).expand(B, -1, -1)  # (B, K, D)
        k = self.k_proj(x)  # (B, N, D)
        v = self.v_proj(x)  # (B, N, D)

        # Multi-head attention
        q = q.view(B, self.num_slices, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention: (B, H, K, D/H) @ (B, H, D/H, N) -> (B, H, K, N)
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)

        # (B, H, K, N) @ (B, H, N, D/H) -> (B, H, K, D/H)
        slices = torch.matmul(attn, v)
        slices = slices.transpose(1, 2).reshape(B, self.num_slices, D)

        # Slice-to-point attention: broadcast slices back to points
        q2 = self.slice_q(self.norm1(x))  # (B, N, D)
        k2 = self.slice_k(slices)  # (B, K, D)
        v2 = self.slice_v(slices)  # (B, K, D)

        q2 = q2.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k2 = k2.view(B, self.num_slices, self.num_heads, self.head_dim).transpose(1, 2)
        v2 = v2.view(B, self.num_slices, self.num_heads, self.head_dim).transpose(1, 2)

        attn2 = torch.matmul(q2, k2.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn2 = F.softmax(attn2, dim=-1)

        out = torch.matmul(attn2, v2)
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.out_proj(out)

        return self.norm2(x + out)


class TransolverBlock(nn.Module):
    """Transolver block: Physics-Attention + FFN."""

    def __init__(self, dim, num_slices=16, num_heads=8, mlp_ratio=4):
        super().__init__()
        self.attn = PhysicsAttention(dim, num_slices, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.attn(x)
        x = x + self.ffn(self.norm(x))
        return x


class Transolver3D(nn.Module):
    """
    Transolver for 3D temporal prediction.

    Input: (B, T_in, C, Z, H, W)
    Output: (B, T_out, C, Z, H, W)
    """

    def __init__(self, in_channels=4, out_channels=4,
                 input_timesteps=5, output_timesteps=5,
                 dim=256, depth=6, num_slices=16, num_heads=8,
                 patch_size=(8, 8, 8)):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_timesteps = input_timesteps
        self.output_timesteps = output_timesteps
        self.patch_size = patch_size

        # Patch embedding
        p_z, p_h, p_w = patch_size
        self.patch_embed = nn.Conv3d(
            input_timesteps * in_channels, dim,
            kernel_size=patch_size, stride=patch_size
        )

        # Transolver blocks
        self.blocks = nn.ModuleList([
            TransolverBlock(dim, num_slices, num_heads)
            for _ in range(depth)
        ])

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, p_z * p_h * p_w * output_timesteps * out_channels)
        )

        # Store grid size
        self.num_patches = None

    def forward(self, x):
        # x: (B, T_in, C, Z, H, W)
        B, T, C, Z, H, W = x.shape

        # Reshape
        x = x.view(B, T * C, Z, H, W)

        # Patch embedding: (B, D, nZ, nH, nW)
        x = self.patch_embed(x)
        nZ, nH, nW = x.shape[2], x.shape[3], x.shape[4]
        self.num_patches = (nZ, nH, nW)

        # Reshape to sequence: (B, N, D) where N = nZ*nH*nW
        x = x.flatten(2).transpose(1, 2)

        # Transolver blocks
        for block in self.blocks:
            x = block(x)

        # Decode: (B, N, D) -> (B, N, p_z*p_h*p_w*T_out*C)
        x = self.decoder(x)

        # Reshape: (B, nZ, nH, nW, p_z, p_h, p_w, T_out, C)
        p_z, p_h, p_w = self.patch_size
        x = x.view(B, nZ, nH, nW, p_z, p_h, p_w, self.output_timesteps, self.out_channels)

        # Rearrange: (B, T_out, C, Z, H, W)
        x = x.permute(0, 7, 8, 1, 4, 2, 5, 3, 6)  # (B, T, C, nZ, p_z, nH, p_h, nW, p_w)
        x = x.reshape(B, self.output_timesteps, self.out_channels, Z, H, W)

        return x

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# Test Transolver

# Cell 17
# ============================================================
# DPOT - Denoising Pretraining Operator Transformer (ICML 2024)
# ============================================================

class SpectralAttention(nn.Module):
    """
    Spectral Attention: Attention in Fourier space.
    Key insight: operate on frequency coefficients rather than spatial points.
    """

    def __init__(self, dim, modes_z=8, modes_h=8, modes_w=8, num_heads=8):
        super().__init__()
        self.dim = dim
        self.modes_z = modes_z
        self.modes_h = modes_h
        self.modes_w = modes_w
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Learnable spectral weights
        self.spectral_weight = nn.Parameter(
            torch.randn(num_heads, modes_z, modes_h, modes_w//2+1, dtype=torch.cfloat) * 0.02
        )

        # Projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.norm = nn.LayerNorm(dim)

    def forward(self, x, spatial_shape):
        # x: (B, N, D) where N = Z*H*W
        B, N, D = x.shape
        Z, H, W = spatial_shape

        # Reshape to spatial
        x_spatial = x.view(B, Z, H, W, D).permute(0, 4, 1, 2, 3)  # (B, D, Z, H, W)

        # FFT
        x_ft = torch.fft.rfftn(x_spatial, dim=(-3, -2, -1))

        # Apply spectral weights
        mz, mh, mw = min(self.modes_z, Z), min(self.modes_h, H), min(self.modes_w//2+1, W//2+1)

        # Simple spectral filtering (element-wise multiply with learnable weights)
        out_ft = torch.zeros_like(x_ft)

        # Reshape for multi-head processing
        x_ft_heads = x_ft.view(B, self.num_heads, self.head_dim, Z, H, W//2+1)

        # Apply spectral weights to low frequencies
        for h in range(self.num_heads):
            weight = self.spectral_weight[h, :mz, :mh, :mw]
            out_ft[:, h*self.head_dim:(h+1)*self.head_dim, :mz, :mh, :mw] = \
                x_ft[:, h*self.head_dim:(h+1)*self.head_dim, :mz, :mh, :mw] * weight.unsqueeze(0).unsqueeze(0)

        # IFFT
        x_spatial = torch.fft.irfftn(out_ft, s=(Z, H, W))

        # Back to sequence
        x_out = x_spatial.permute(0, 2, 3, 4, 1).reshape(B, N, D)

        # Residual with projection
        x_out = self.out_proj(x_out)
        return self.norm(x + x_out)


class DPOTBlock(nn.Module):
    """DPOT block: Spectral Attention + FFN."""

    def __init__(self, dim, modes_z=8, modes_h=8, modes_w=8, num_heads=8, mlp_ratio=4):
        super().__init__()
        self.spectral_attn = SpectralAttention(dim, modes_z, modes_h, modes_w, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, spatial_shape):
        x = self.spectral_attn(x, spatial_shape)
        x = x + self.ffn(self.norm(x))
        return x


class DPOT3D(nn.Module):
    """
    DPOT for 3D temporal prediction.

    Input: (B, T_in, C, Z, H, W)
    Output: (B, T_out, C, Z, H, W)
    """

    def __init__(self, in_channels=4, out_channels=4,
                 input_timesteps=5, output_timesteps=5,
                 dim=256, depth=8, modes=16, num_heads=8,
                 patch_size=(4, 8, 8)):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_timesteps = input_timesteps
        self.output_timesteps = output_timesteps
        self.patch_size = patch_size

        # Patch embedding
        p_z, p_h, p_w = patch_size
        self.patch_embed = nn.Conv3d(
            input_timesteps * in_channels, dim,
            kernel_size=patch_size, stride=patch_size
        )

        # DPOT blocks
        self.blocks = nn.ModuleList([
            DPOTBlock(dim, modes, modes, modes, num_heads)
            for _ in range(depth)
        ])

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, p_z * p_h * p_w * output_timesteps * out_channels)
        )

    def forward(self, x):
        # x: (B, T_in, C, Z, H, W)
        B, T, C, Z, H, W = x.shape

        # Reshape
        x = x.view(B, T * C, Z, H, W)

        # Patch embedding
        x = self.patch_embed(x)
        nZ, nH, nW = x.shape[2], x.shape[3], x.shape[4]
        spatial_shape = (nZ, nH, nW)

        # Reshape to sequence
        x = x.flatten(2).transpose(1, 2)

        # DPOT blocks
        for block in self.blocks:
            x = block(x, spatial_shape)

        # Decode
        x = self.decoder(x)

        # Reshape
        p_z, p_h, p_w = self.patch_size
        x = x.view(B, nZ, nH, nW, p_z, p_h, p_w, self.output_timesteps, self.out_channels)
        x = x.permute(0, 7, 8, 1, 4, 2, 5, 3, 6)
        x = x.reshape(B, self.output_timesteps, self.out_channels, Z, H, W)

        return x

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# Test DPOT

# Cell 18
# ============================================================
# Factformer - Factorized Attention Transformer (NeurIPS 2023)
# ============================================================

class AxialAttention(nn.Module):
    """Axial attention along a single dimension."""

    def __init__(self, dim, num_heads=8, axis=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.axis = axis

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, shape):
        # x: (B, N, D) where N = Z*H*W
        B, N, D = x.shape
        Z, H, W = shape

        # Reshape to 3D
        x = x.view(B, Z, H, W, D)

        # Transpose based on axis
        if self.axis == 0:  # Z axis
            x = x.permute(0, 2, 3, 1, 4)  # (B, H, W, Z, D)
            seq_len = Z
        elif self.axis == 1:  # H axis
            x = x.permute(0, 1, 3, 2, 4)  # (B, Z, W, H, D)
            seq_len = H
        else:  # W axis
            x = x  # (B, Z, H, W, D)
            seq_len = W

        # Reshape for attention: (B*other_dims, seq_len, D)
        batch_size = x.shape[0] * x.shape[1] * x.shape[2]
        x = x.reshape(batch_size, seq_len, D)

        # QKV
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, D)
        out = self.proj(out)

        # Reshape back
        if self.axis == 0:
            out = out.view(B, H, W, Z, D).permute(0, 3, 1, 2, 4)
        elif self.axis == 1:
            out = out.view(B, Z, W, H, D).permute(0, 1, 3, 2, 4)
        else:
            out = out.view(B, Z, H, W, D)

        return out.reshape(B, N, D)


class FactorizedAttention(nn.Module):
    """Factorized attention: Z -> H -> W axial attention."""

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.attn_z = AxialAttention(dim, num_heads, axis=0)
        self.attn_h = AxialAttention(dim, num_heads, axis=1)
        self.attn_w = AxialAttention(dim, num_heads, axis=2)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, shape):
        x = x + self.attn_z(self.norm1(x), shape)
        x = x + self.attn_h(self.norm2(x), shape)
        x = x + self.attn_w(self.norm3(x), shape)
        return x


class FactformerBlock(nn.Module):
    """Factformer block: Factorized Attention + FFN."""

    def __init__(self, dim, num_heads=8, mlp_ratio=4):
        super().__init__()
        self.attn = FactorizedAttention(dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, shape):
        x = self.attn(x, shape)
        x = x + self.ffn(self.norm(x))
        return x


class Factformer3D(nn.Module):
    """
    Factformer for 3D temporal prediction.

    Input: (B, T_in, C, Z, H, W)
    Output: (B, T_out, C, Z, H, W)
    """

    def __init__(self, in_channels=4, out_channels=4,
                 input_timesteps=5, output_timesteps=5,
                 dim=128, depth=6, num_heads=8, downsample=4):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_timesteps = input_timesteps
        self.output_timesteps = output_timesteps
        self.downsample = downsample

        # Downsampling via 3D conv (reduce spatial resolution)
        self.encoder = nn.Sequential(
            nn.Conv3d(input_timesteps * in_channels, dim,
                      kernel_size=downsample, stride=downsample),
            nn.GELU()
        )

        # Factformer blocks
        self.blocks = nn.ModuleList([
            FactformerBlock(dim, num_heads)
            for _ in range(depth)
        ])

        # Upsampling decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(dim, dim, kernel_size=downsample, stride=downsample),
            nn.GELU(),
            nn.Conv3d(dim, output_timesteps * out_channels, 1)
        )

    def forward(self, x):
        # x: (B, T_in, C, Z, H, W)
        B, T, C, Z, H, W = x.shape

        # Reshape
        x = x.view(B, T * C, Z, H, W)

        # Encode (downsample)
        x = self.encoder(x)  # (B, D, Z/ds, H/ds, W/ds)
        nZ, nH, nW = x.shape[2], x.shape[3], x.shape[4]
        shape = (nZ, nH, nW)

        # Reshape to sequence
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)

        # Factformer blocks
        for block in self.blocks:
            x = block(x, shape)

        # Reshape back to spatial
        x = x.transpose(1, 2).view(B, -1, nZ, nH, nW)

        # Decode (upsample)
        x = self.decoder(x)

        # Reshape
        x = x.view(B, self.output_timesteps, self.out_channels, Z, H, W)

        return x

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# Test Factformer

# Cell 19
# ============================================================
# PINO - Physics-Informed Neural Operator
# ============================================================

class PINO3D(nn.Module):
    """
    Physics-Informed Neural Operator.
    Same architecture as FNO but with physics loss computation.

    Input: (B, T_in, C, Z, H, W)
    Output: (B, T_out, C, Z, H, W)
    """

    def __init__(self, in_channels=4, out_channels=4,
                 input_timesteps=5, output_timesteps=5,
                 width=32, modes_z=16, modes_h=16, modes_w=16,
                 num_layers=4):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_timesteps = input_timesteps
        self.output_timesteps = output_timesteps
        self.width = width

        # FNO architecture
        self.lift = nn.Conv3d(input_timesteps * in_channels, width, 1)

        self.blocks = nn.ModuleList([
            FNOBlock3d(width, modes_z, modes_h, modes_w)
            for _ in range(num_layers)
        ])

        self.proj1 = nn.Conv3d(width, width * 2, 1)
        self.proj2 = nn.Conv3d(width * 2, output_timesteps * out_channels, 1)

    def forward(self, x):
        B, T, C, Z, H, W = x.shape
        x = x.view(B, T * C, Z, H, W)

        x = self.lift(x)
        for block in self.blocks:
            x = block(x)
        x = F.gelu(self.proj1(x))
        x = self.proj2(x)

        x = x.view(B, self.output_timesteps, self.out_channels, Z, H, W)
        return x

    def compute_physics_loss(self, pred, grid_spacing=None):
        """
        Compute physics losses: divergence constraint for incompressible flow.

        Args:
            pred: (B, T, C, Z, H, W) where C = [u, v, w, p], in PHYSICAL units
            grid_spacing: tuple (dz, dy, dx) or scalar. If None, uses 1.0.

        Returns:
            Divergence loss (should be ~0 for incompressible flow)
        """
        # Parse grid spacing
        if grid_spacing is None:
            dz, dy, dx = 1.0, 1.0, 1.0
        elif isinstance(grid_spacing, (tuple, list)):
            dz, dy, dx = grid_spacing
        else:
            dz = dy = dx = float(grid_spacing)

        u = pred[:, :, 0]  # (B, T, Z, H, W)
        v = pred[:, :, 1]
        w = pred[:, :, 2]

        # Compute divergence using central differences
        # ∂u/∂x + ∂v/∂y + ∂w/∂z ≈ 0
        du_dx = (u[:, :, :, :, 2:] - u[:, :, :, :, :-2]) / (2 * dx)
        dv_dy = (v[:, :, :, 2:, :] - v[:, :, :, :-2, :]) / (2 * dy)
        dw_dz = (w[:, :, 2:, :, :] - w[:, :, :-2, :, :]) / (2 * dz)

        # Crop to same size
        min_z = min(du_dx.shape[2], dv_dy.shape[2], dw_dz.shape[2])
        min_h = min(du_dx.shape[3], dv_dy.shape[3], dw_dz.shape[3])
        min_w = min(du_dx.shape[4], dv_dy.shape[4], dw_dz.shape[4])

        du_dx = du_dx[:, :, :min_z, :min_h, :min_w]
        dv_dy = dv_dy[:, :, :min_z, :min_h, :min_w]
        dw_dz = dw_dz[:, :, :min_z, :min_h, :min_w]

        divergence = du_dx + dv_dy + dw_dz
        div_loss = torch.mean(divergence ** 2)

        return div_loss

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# Test PINO

# Cell 20
# ============================================================
# DeepONet - Deep Operator Network
# ============================================================

class FourierPositionalEncoding(nn.Module):
    """
    Fourier positional encoding for coordinate inputs.
    Following NeRF/DeepONet best practices for encoding spatial coordinates.
    """

    def __init__(self, num_freqs=10, include_input=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input

        # Frequencies: 2^0, 2^1, ..., 2^(L-1)
        freqs = 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs)
        self.register_buffer('freqs', freqs)

    def forward(self, x):
        """
        Args:
            x: (..., D) coordinates in [0, 1]
        Returns:
            (..., D * (2 * num_freqs + include_input)) encoded features
        """
        out = []
        if self.include_input:
            out.append(x)

        for freq in self.freqs:
            out.append(torch.sin(freq * np.pi * x))
            out.append(torch.cos(freq * np.pi * x))

        return torch.cat(out, dim=-1)

    def output_dim(self, input_dim):
        """Calculate output dimension given input dimension."""
        return input_dim * (2 * self.num_freqs + int(self.include_input))


class BranchNet(nn.Module):
    """
    Branch network: encodes the input function.
    Uses 3D CNN with Instance Normalization for input-sensitive encoding.

    FIXED: Added normalization layers to prevent mode collapse and
    preserve input-dependent variations in the output.
    """

    def __init__(self, in_channels, hidden_dim, out_dim):
        super().__init__()

        # Encoder with Instance Normalization for better gradient flow
        # and input-dependent behavior
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, 4, stride=2, padding=1),
            nn.InstanceNorm3d(hidden_dim, affine=True),
            nn.GELU(),
            nn.Conv3d(hidden_dim, hidden_dim*2, 4, stride=2, padding=1),
            nn.InstanceNorm3d(hidden_dim*2, affine=True),
            nn.GELU(),
            nn.Conv3d(hidden_dim*2, hidden_dim*4, 4, stride=2, padding=1),
            nn.InstanceNorm3d(hidden_dim*4, affine=True),
            nn.GELU(),
            nn.Conv3d(hidden_dim*4, hidden_dim*4, 4, stride=2, padding=1),
            nn.InstanceNorm3d(hidden_dim*4, affine=True),
            nn.GELU(),
        )

        # Use 4x4x4 pooling to preserve more spatial information
        self.pool = nn.AdaptiveAvgPool3d(4)
        self.max_pool = nn.AdaptiveMaxPool3d(4)

        # FC layers with 4x4x4 pooling
        # pool_features = hidden_dim * 4 * 64 * 2 = hidden_dim * 512
        # For hidden_dim=48: 48 * 512 = 24576
        pool_features = hidden_dim * 4 * 64 * 2  # 4x4x4 * 2 pools * hidden_dim*4 channels
        # Use gradual expansion
        fc_dim1 = min(1536, max(768, out_dim // 2))  # 768-1536
        fc_dim2 = min(1536, max(768, out_dim // 2))  # 768-1536
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(pool_features, fc_dim1),
            nn.LayerNorm(fc_dim1),
            nn.GELU(),
            nn.Linear(fc_dim1, fc_dim2),
            nn.LayerNorm(fc_dim2),
            nn.GELU(),
            nn.Linear(fc_dim2, out_dim)
        )

    def forward(self, x):
        # x: (B, C, Z, H, W)
        x = self.encoder(x)
        # Combine avg and max pooling for richer features
        x_avg = self.pool(x)
        x_max = self.max_pool(x)
        x = torch.cat([x_avg, x_max], dim=1)
        x = self.fc(x)  # (B, out_dim)
        return x


class TrunkNet(nn.Module):
    """
    Trunk network: encodes spatial coordinates.
    Uses fixed 3D grid coordinates with Fourier positional encoding.
    """

    def __init__(self, coord_dim, hidden_dim, out_dim, num_points, spatial_size=None,
                 num_freqs=10, use_positional_encoding=True):
        super().__init__()

        self.use_positional_encoding = use_positional_encoding

        # Fourier positional encoding
        if use_positional_encoding:
            self.pos_encoding = FourierPositionalEncoding(num_freqs=num_freqs, include_input=True)
            input_dim = self.pos_encoding.output_dim(coord_dim)
        else:
            self.pos_encoding = None
            input_dim = coord_dim

        # Create 3D grid coordinates
        if spatial_size is not None:
            Z, H, W = spatial_size
            assert Z * H * W == num_points, f"Spatial size {spatial_size} doesn't match num_points {num_points}"

            # Create normalized 3D grid coordinates [0, 1]
            z = torch.linspace(0, 1, Z)
            h = torch.linspace(0, 1, H)
            w = torch.linspace(0, 1, W)
            zz, hh, ww = torch.meshgrid(z, h, w, indexing='ij')
            coords = torch.stack([zz.flatten(), hh.flatten(), ww.flatten()], dim=1)
        else:
            # Fallback: use learnable embedding (for backward compatibility)
            coords = torch.randn(num_points, coord_dim) * 0.02

        # Register as buffer (not trainable parameter)
        self.register_buffer('coords', coords)

        # MLP with positional encoding
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, batch_size):
        # Apply positional encoding if enabled
        if self.use_positional_encoding and self.pos_encoding is not None:
            encoded_coords = self.pos_encoding(self.coords)  # (N, input_dim)
        else:
            encoded_coords = self.coords  # (N, coord_dim)

        # MLP forward
        x = self.net(encoded_coords)  # (N, out_dim)
        return x.unsqueeze(0).expand(batch_size, -1, -1)  # (B, N, out_dim)


class DeepONet3D(nn.Module):
    """
    DeepONet for 3D temporal prediction (FIXED: no downsampling).

    Input: (B, T_in, C, Z, H, W)
    Output: (B, T_out, C, Z, H, W)
    """

    def __init__(self, in_channels=4, out_channels=4,
                 input_timesteps=5, output_timesteps=5,
                 spatial_size=(128, 128, 128),
                 branch_dim=32, trunk_dim=128, basis_dim=64):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_timesteps = input_timesteps
        self.output_timesteps = output_timesteps
        self.spatial_size = spatial_size

        Z, H, W = spatial_size
        self.num_points = Z * H * W

        # NO DOWNSAMPLING - work at full resolution
        # This is the correct DeepONet approach

        # Branch: encode input (multiple branches for output channels and timesteps)
        num_outputs = output_timesteps * out_channels
        self.branch = BranchNet(
            input_timesteps * in_channels,
            branch_dim,
            basis_dim * num_outputs
        )

        # Trunk: encode positions at FULL resolution
        self.trunk = TrunkNet(
            coord_dim=3,
            hidden_dim=trunk_dim,
            out_dim=basis_dim,
            num_points=self.num_points,
            spatial_size=spatial_size  # FIXED: pass actual spatial size
        )

        self.num_outputs = num_outputs
        self.basis_dim = basis_dim

    def forward(self, x):
        # x: (B, T_in, C, Z, H, W)
        B, T, C, Z, H, W = x.shape

        # Reshape input
        x_in = x.view(B, T * C, Z, H, W)

        # Branch: encode input function
        branch_out = self.branch(x_in)  # (B, basis_dim * num_outputs)
        branch_out = branch_out.view(B, self.num_outputs, self.basis_dim)

        # Trunk: encode positions at FULL resolution
        trunk_out = self.trunk(B)  # (B, N, basis_dim) where N = Z*H*W

        # Dot product: (B, num_outputs, basis_dim) @ (B, basis_dim, N)
        # -> (B, num_outputs, N)
        out = torch.bmm(branch_out, trunk_out.transpose(1, 2))

        # Reshape to spatial (full resolution, no upsampling needed)
        out = out.view(B, self.num_outputs, Z, H, W)

        # Reshape to (B, T_out, C, Z, H, W)
        out = out.view(B, self.output_timesteps, self.out_channels, Z, H, W)

        return out

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# Test DeepONet

# Cell 21
# ============================================================
# PI-DeepONet - Physics-Informed DeepONet
# ============================================================

class PIDeepONet3D(nn.Module):
    """
    Physics-Informed DeepONet (FIXED: no downsampling).
    Same as DeepONet but with physics loss computation.

    Input: (B, T_in, C, Z, H, W)
    Output: (B, T_out, C, Z, H, W)
    """

    def __init__(self, in_channels=4, out_channels=4,
                 input_timesteps=5, output_timesteps=5,
                 spatial_size=(128, 128, 128),
                 branch_dim=32, trunk_dim=128, basis_dim=64):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_timesteps = input_timesteps
        self.output_timesteps = output_timesteps
        self.spatial_size = spatial_size

        Z, H, W = spatial_size
        self.num_points = Z * H * W

        # NO DOWNSAMPLING - work at full resolution

        num_outputs = output_timesteps * out_channels
        self.branch = BranchNet(
            input_timesteps * in_channels,
            branch_dim,
            basis_dim * num_outputs
        )

        self.trunk = TrunkNet(
            coord_dim=3,
            hidden_dim=trunk_dim,
            out_dim=basis_dim,
            num_points=self.num_points,
            spatial_size=spatial_size  # FIXED: pass actual spatial size
        )

        self.num_outputs = num_outputs
        self.basis_dim = basis_dim

    def forward(self, x):
        B, T, C, Z, H, W = x.shape
        x_in = x.view(B, T * C, Z, H, W)

        branch_out = self.branch(x_in)
        branch_out = branch_out.view(B, self.num_outputs, self.basis_dim)

        trunk_out = self.trunk(B)

        out = torch.bmm(branch_out, trunk_out.transpose(1, 2))

        # Full resolution, no upsampling needed
        out = out.view(B, self.num_outputs, Z, H, W)
        out = out.view(B, self.output_timesteps, self.out_channels, Z, H, W)

        return out

    def compute_physics_loss(self, pred, grid_spacing=None):
        """
        Compute physics loss (divergence constraint for incompressible flow).

        Args:
            pred: (B, T, C, Z, H, W) where C = [u, v, w, p], in PHYSICAL units
            grid_spacing: tuple (dz, dy, dx) or scalar. If None, uses 1.0.

        Returns:
            Divergence loss (should be ~0 for incompressible flow)
        """
        # Parse grid spacing
        if grid_spacing is None:
            dz, dy, dx = 1.0, 1.0, 1.0
        elif isinstance(grid_spacing, (tuple, list)):
            dz, dy, dx = grid_spacing
        else:
            dz = dy = dx = float(grid_spacing)

        u = pred[:, :, 0]
        v = pred[:, :, 1]
        w = pred[:, :, 2]

        # Compute divergence using central differences
        du_dx = (u[:, :, :, :, 2:] - u[:, :, :, :, :-2]) / (2 * dx)
        dv_dy = (v[:, :, :, 2:, :] - v[:, :, :, :-2, :]) / (2 * dy)
        dw_dz = (w[:, :, 2:, :, :] - w[:, :, :-2, :, :]) / (2 * dz)

        min_z = min(du_dx.shape[2], dv_dy.shape[2], dw_dz.shape[2])
        min_h = min(du_dx.shape[3], dv_dy.shape[3], dw_dz.shape[3])
        min_w = min(du_dx.shape[4], dv_dy.shape[4], dw_dz.shape[4])

        du_dx = du_dx[:, :, :min_z, :min_h, :min_w]
        dv_dy = dv_dy[:, :, :min_z, :min_h, :min_w]
        dw_dz = dw_dz[:, :, :min_z, :min_h, :min_w]

        divergence = du_dx + dv_dy + dw_dz
        return torch.mean(divergence ** 2)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# Test PI-DeepONet

# Cell 22
# ============================================================
# TFNO - Tensorized Fourier Neural Operator (ICLR 2023)
# ============================================================

class TuckerSpectralConv3d(nn.Module):
    """
    Tucker-decomposed 3D Spectral Convolution.
    Instead of full weight tensor (in, out, z, h, w),
    uses Tucker decomposition: core × (U_in, U_out, U_z, U_h, U_w)
    """

    def __init__(self, in_channels, out_channels, modes_z, modes_h, modes_w, rank=16):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_z = modes_z
        self.modes_h = modes_h
        self.modes_w = modes_w
        self.rank = rank

        # Factor matrices for Tucker decomposition
        self.U_in = nn.Parameter(torch.randn(in_channels, rank) * 0.02)
        self.U_out = nn.Parameter(torch.randn(out_channels, rank) * 0.02)
        self.U_z = nn.Parameter(torch.randn(modes_z, rank) * 0.02)
        self.U_h = nn.Parameter(torch.randn(modes_h, rank) * 0.02)
        self.U_w = nn.Parameter(torch.randn(modes_w, rank) * 0.02)

        # Core tensor (much smaller than full weight)
        self.core = nn.Parameter(torch.randn(rank, rank, rank, rank, rank, dtype=torch.cfloat) * 0.02)

        # Second set for negative frequencies
        self.U_z2 = nn.Parameter(torch.randn(modes_z, rank) * 0.02)
        self.U_h2 = nn.Parameter(torch.randn(modes_h, rank) * 0.02)
        self.core2 = nn.Parameter(torch.randn(rank, rank, rank, rank, rank, dtype=torch.cfloat) * 0.02)

    def reconstruct_weight(self, core, U_in, U_out, U_z, U_h, U_w):
        """Reconstruct full weight from Tucker factors."""
        # core: (r, r, r, r, r)
        # Result: (in, out, z, h, w)
        weight = torch.einsum('ir,or,zr,hr,wr,abcde->iozwh',
                             U_in.cfloat(), U_out.cfloat(),
                             U_z.cfloat(), U_h.cfloat(), U_w.cfloat(),
                             core)
        return weight

    def forward(self, x):
        B = x.shape[0]

        # FFT
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # Allocate output
        out_ft = torch.zeros(B, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1,
                            dtype=torch.cfloat, device=x.device)

        # Reconstruct weights from Tucker decomposition
        weight1 = self.reconstruct_weight(self.core, self.U_in, self.U_out,
                                          self.U_z, self.U_h, self.U_w)
        weight2 = self.reconstruct_weight(self.core2, self.U_in, self.U_out,
                                          self.U_z2, self.U_h2, self.U_w)

        # Apply to low frequency modes
        out_ft[:, :, :self.modes_z, :self.modes_h, :self.modes_w] = \
            torch.einsum("bixyz,ioxyz->boxyz",
                        x_ft[:, :, :self.modes_z, :self.modes_h, :self.modes_w],
                        weight1)

        out_ft[:, :, -self.modes_z:, :self.modes_h, :self.modes_w] = \
            torch.einsum("bixyz,ioxyz->boxyz",
                        x_ft[:, :, -self.modes_z:, :self.modes_h, :self.modes_w],
                        weight2)

        # IFFT
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


class TFNOBlock3d(nn.Module):
    """TFNO Block with Tucker-decomposed spectral conv."""

    def __init__(self, width, modes_z, modes_h, modes_w, rank=16):
        super().__init__()
        self.spectral_conv = TuckerSpectralConv3d(width, width, modes_z, modes_h, modes_w, rank)
        self.conv = nn.Conv3d(width, width, 1)
        self.norm = nn.InstanceNorm3d(width)

    def forward(self, x):
        x1 = self.spectral_conv(x)
        x2 = self.conv(x)
        x = self.norm(x1 + x2)
        return F.gelu(x)


class TFNO3D(nn.Module):
    """
    Tensorized FNO for 3D temporal prediction.

    Input: (B, T_in, C, Z, H, W)
    Output: (B, T_out, C, Z, H, W)
    """

    def __init__(self, in_channels=4, out_channels=4,
                 input_timesteps=5, output_timesteps=5,
                 width=32, modes_z=16, modes_h=16, modes_w=16,
                 num_layers=4, rank=16):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_timesteps = input_timesteps
        self.output_timesteps = output_timesteps
        self.width = width

        self.lift = nn.Conv3d(input_timesteps * in_channels, width, 1)

        self.blocks = nn.ModuleList([
            TFNOBlock3d(width, modes_z, modes_h, modes_w, rank)
            for _ in range(num_layers)
        ])

        self.proj1 = nn.Conv3d(width, width * 2, 1)
        self.proj2 = nn.Conv3d(width * 2, output_timesteps * out_channels, 1)

    def forward(self, x):
        B, T, C, Z, H, W = x.shape
        x = x.view(B, T * C, Z, H, W)

        x = self.lift(x)
        for block in self.blocks:
            x = block(x)
        x = F.gelu(self.proj1(x))
        x = self.proj2(x)

        x = x.view(B, self.output_timesteps, self.out_channels, Z, H, W)
        return x

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# Test TFNO

# Cell 23
# ============================================================
# Model Configurations (adapts to data source: DNS or JHU)
# ============================================================
# DNS: 64x128x128 (z truncated from 65), smaller z-modes/patch
# JHU: 128x128x128, symmetric configuration

# Data source specific parameters
if DATA_SOURCE == 'DNS':
    # DNS: asymmetric - z is half the size of h/w
    FNO_MODES = {'z': 8, 'h': 12, 'w': 12}  # Fewer z modes for 64 layers
    PATCH_TRANS = (2, 8, 8)  # Smaller z patch for 64 layers
    PATCH_DPOT = (2, 8, 8)
else:
    # JHU: symmetric 128^3
    FNO_MODES = {'z': 12, 'h': 12, 'w': 12}
    PATCH_TRANS = (8, 8, 8)
    PATCH_DPOT = (4, 8, 8)

print(f"Model configs for {DATA_SOURCE}:")
print(f"  FNO modes: z={FNO_MODES['z']}, h={FNO_MODES['h']}, w={FNO_MODES['w']}")
print(f"  Transolver patch: {PATCH_TRANS}")
print(f"  DPOT patch: {PATCH_DPOT}")

BASELINE_CONFIGS = {
    # --- Spectral Methods ---
    'fno': {
        'class': FNO3D,
        'params': {
            'width': 32,
            'modes_z': FNO_MODES['z'],
            'modes_h': FNO_MODES['h'],
            'modes_w': FNO_MODES['w'],
            'num_layers': 4
        },
        'use_amp': False,
    },
    'ufno': {
        'class': UFNO3D,
        'params': {
            'base_width': 32,
            'modes': min(FNO_MODES.values())
        },
        'use_amp': False,
    },

    # --- Transformer Methods ---
    'transolver': {
        'class': Transolver3D,
        'params': {
            'dim': 384,
            'depth': 8,
            'num_slices': 64,
            'num_heads': 12,
            'patch_size': PATCH_TRANS
        },
        'use_amp': True,
    },
    'dpot': {
        'class': DPOT3D,
        'params': {
            'dim': 384,
            'depth': 12,
            'modes': 16,
            'num_heads': 12,
            'patch_size': PATCH_DPOT
        },
        'use_amp': False,
    },
    'factformer': {
        'class': Factformer3D,
        'params': {
            'dim': 320,
            'depth': 10,
            'num_heads': 10,
            'downsample': 4
        },
        'use_amp': True,
    },

    # --- Physics-Informed Methods ---
    'pino': {
        'class': PINO3D,
        'params': {
            'width': 32,
            'modes_z': FNO_MODES['z'],
            'modes_h': FNO_MODES['h'],
            'modes_w': FNO_MODES['w'],
            'num_layers': 4
        },
        'use_amp': False,
        'use_physics_loss': True,
        'physics_weight': 0.01,
    },

    # --- Operator Methods ---
    'deeponet': {
        'class': DeepONet3D,
        'params': {
            'spatial_size': SPATIAL_SIZE,
            'branch_dim': 32,
            'trunk_dim': 128,
            'basis_dim': 64
        },
        'use_amp': True,
    },
    'pi_deeponet': {
        'class': PIDeepONet3D,
        'params': {
            'spatial_size': SPATIAL_SIZE,
            'branch_dim': 32,
            'trunk_dim': 128,
            'basis_dim': 64
        },
        'use_amp': True,
        'use_physics_loss': True,
        'physics_weight': 0.01,
    },

    # --- Tensorized Methods ---
    'tfno': {
        'class': TFNO3D,
        'params': {
            'width': 96,
            'modes_z': min(16, FNO_MODES['z'] + 4),
            'modes_h': 16,
            'modes_w': 16,
            'num_layers': 6,
            'rank': 16
        },
        'use_amp': False,
    },
}


BASELINE_SIZE_CONFIGS = {
    'fno': {
        'small':   {'width': 20, 'num_layers': 3},
        'mid':     {'width': 32, 'num_layers': 4},
        'large':   {'width': 48, 'num_layers': 5},
        'xlarge':  {'width': 64, 'num_layers': 6},
    },
    'ufno': {
        'small':   {'base_width': 20},
        'mid':     {'base_width': 32},
        'large':   {'base_width': 48},
        'xlarge':  {'base_width': 64},
    },
    'tfno': {
        'small':   {'width': 48, 'num_layers': 4, 'rank': 12},
        'mid':     {'width': 96, 'num_layers': 6, 'rank': 16},
        'large':   {'width': 128, 'num_layers': 8, 'rank': 24},
        'xlarge':  {'width': 192, 'num_layers': 10, 'rank': 32},
    },
    'transolver': {
        'small':   {'dim': 192, 'depth': 4, 'num_slices': 32, 'num_heads': 6},
        'mid':     {'dim': 384, 'depth': 8, 'num_slices': 64, 'num_heads': 12},
        'large':   {'dim': 512, 'depth': 12, 'num_slices': 96, 'num_heads': 16},
        'xlarge':  {'dim': 768, 'depth': 16, 'num_slices': 128, 'num_heads': 24},
    },
    'dpot': {
        'small':   {'dim': 192, 'depth': 6, 'modes': 12, 'num_heads': 6},
        'mid':     {'dim': 384, 'depth': 12, 'modes': 16, 'num_heads': 12},
        'large':   {'dim': 512, 'depth': 16, 'modes': 20, 'num_heads': 16},
        'xlarge':  {'dim': 768, 'depth': 20, 'modes': 24, 'num_heads': 24},
    },
    'factformer': {
        'small':   {'dim': 160, 'depth': 6, 'num_heads': 5},
        'mid':     {'dim': 320, 'depth': 10, 'num_heads': 10},
        'large':   {'dim': 448, 'depth': 14, 'num_heads': 14},
        'xlarge':  {'dim': 640, 'depth': 18, 'num_heads': 20},
    },
    'pino': {
        'small':   {'width': 20, 'num_layers': 3},
        'mid':     {'width': 32, 'num_layers': 4},
        'large':   {'width': 48, 'num_layers': 5},
        'xlarge':  {'width': 64, 'num_layers': 6},
    },
    'deeponet': {
        'small':   {'branch_dim': 16, 'trunk_dim': 64, 'basis_dim': 32},
        'mid':     {'branch_dim': 32, 'trunk_dim': 128, 'basis_dim': 64},
        'large':   {'branch_dim': 64, 'trunk_dim': 256, 'basis_dim': 128},
        'xlarge':  {'branch_dim': 96, 'trunk_dim': 384, 'basis_dim': 192},
    },
    'pi_deeponet': {
        'small':   {'branch_dim': 16, 'trunk_dim': 64, 'basis_dim': 32},
        'mid':     {'branch_dim': 32, 'trunk_dim': 128, 'basis_dim': 64},
        'large':   {'branch_dim': 64, 'trunk_dim': 256, 'basis_dim': 128},
        'xlarge':  {'branch_dim': 96, 'trunk_dim': 384, 'basis_dim': 192},
    },
}


def get_baseline_config(model_name, size='mid'):
    """Get baseline model config with specified size.

    Args:
        model_name: One of the keys in BASELINE_CONFIGS
        size: 'small', 'mid', 'large', or 'xlarge'

    Returns:
        Config dict with 'class', 'params', etc.
    """
    if model_name not in BASELINE_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(BASELINE_CONFIGS.keys())}")

    config = {k: v for k, v in BASELINE_CONFIGS[model_name].items()}
    config['params'] = dict(config['params'])

    if model_name in BASELINE_SIZE_CONFIGS and size in BASELINE_SIZE_CONFIGS[model_name]:
        config['params'].update(BASELINE_SIZE_CONFIGS[model_name][size])

    return config

