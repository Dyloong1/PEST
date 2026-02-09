"""
3D Swin Transformer Blocks for Physics-Informed Flow Reconstruction

Key Design Principles:
1. Pixel-level (no patch embedding) - preserves spatial continuity for physics
2. 3D Window-based attention - local receptive field aligns with PDE locality
3. Shifted window mechanism - enables cross-window information flow
4. Support for both standard attention and axial attention

Window Size: (5, 8, 8) for data shape (65, 128, 128)
- Z direction: 65/5 = 13 windows
- H direction: 128/8 = 16 windows
- W direction: 128/8 = 16 windows
- Tokens per window: 5*8*8 = 320 pixels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np


# ============================================================================
# Window Partition Utilities
# ============================================================================

def window_partition_3d(x, window_size):
    """
    Partition input into 3D windows (pixel-level, no patch merging).

    Args:
        x: (B, C, Z, H, W)
        window_size: (Wz, Wh, Ww)

    Returns:
        windows: (B * num_windows, Wz * Wh * Ww, C)
                 Each pixel in window is a separate token
    """
    B, C, Z, H, W = x.shape
    Wz, Wh, Ww = window_size

    # Reshape to windows
    x = x.view(B, C, Z // Wz, Wz, H // Wh, Wh, W // Ww, Ww)
    # Permute to: (B, num_windows_z, num_windows_h, num_windows_w, Wz, Wh, Ww, C)
    x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
    # Flatten windows: (B * num_windows, Wz * Wh * Ww, C)
    windows = x.view(-1, Wz * Wh * Ww, C)

    return windows


def window_reverse_3d(windows, window_size, Z, H, W):
    """
    Reverse window partition back to full spatial dimensions.

    Args:
        windows: (B * num_windows, Wz * Wh * Ww, C)
        window_size: (Wz, Wh, Ww)
        Z, H, W: original spatial dimensions

    Returns:
        x: (B, C, Z, H, W)
    """
    Wz, Wh, Ww = window_size
    num_windows_z = Z // Wz
    num_windows_h = H // Wh
    num_windows_w = W // Ww

    B = windows.shape[0] // (num_windows_z * num_windows_h * num_windows_w)
    C = windows.shape[-1]

    # Reshape: (B, num_windows_z, num_windows_h, num_windows_w, Wz, Wh, Ww, C)
    x = windows.view(B, num_windows_z, num_windows_h, num_windows_w, Wz, Wh, Ww, C)
    # Permute back
    x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
    # Merge: (B, C, Z, H, W)
    x = x.view(B, C, Z, H, W)

    return x


def create_shifted_window_mask_3d(Z, H, W, window_size, shift_size, device='cuda'):
    """
    Create attention mask for shifted windows to prevent cross-boundary attention.

    Args:
        Z, H, W: spatial dimensions
        window_size: (Wz, Wh, Ww)
        shift_size: (Sz, Sh, Sw) - typically half of window_size

    Returns:
        attn_mask: (num_windows, Wz*Wh*Ww, Wz*Wh*Ww)
    """
    Wz, Wh, Ww = window_size
    Sz, Sh, Sw = shift_size

    # Create coordinate mask
    img_mask = torch.zeros((1, Z, H, W, 1), device=device)

    # Divide into regions
    z_slices = (slice(0, -Wz), slice(-Wz, -Sz), slice(-Sz, None))
    h_slices = (slice(0, -Wh), slice(-Wh, -Sh), slice(-Sh, None))
    w_slices = (slice(0, -Ww), slice(-Ww, -Sw), slice(-Sw, None))

    cnt = 0
    for z in z_slices:
        for h in h_slices:
            for w in w_slices:
                img_mask[:, z, h, w, :] = cnt
                cnt += 1

    # Partition into windows: (num_windows, Wz*Wh*Ww, 1)
    # img_mask shape is (1, Z, H, W, 1), need to rearrange to (B, C, Z, H, W)
    img_mask = rearrange(img_mask, 'b z h w c -> b c z h w')
    mask_windows = window_partition_3d(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)  # (num_windows, Wz*Wh*Ww)

    # Create attention mask
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
    attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))

    return attn_mask


# ============================================================================
# Window-based Attention Modules
# ============================================================================

class WindowAttention3D(nn.Module):
    """
    3D Window-based Multi-Head Self-Attention (pixel-level).

    Supports two modes:
    1. Standard attention: full QKV interaction within window
    2. Axial attention: factorized attention along Z, H, W axes

    Args:
        dim: feature dimension
        window_size: (Wz, Wh, Ww)
        num_heads: number of attention heads
        qkv_bias: whether to use bias in qkv projection
        attn_drop: attention dropout rate
        proj_drop: projection dropout rate
        attention_type: 'standard' or 'axial'
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True,
                 attn_drop=0., proj_drop=0., attention_type='standard'):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wz, Wh, Ww)
        self.num_heads = num_heads
        self.attention_type = attention_type
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads)
        )

        # Get pair-wise relative position index
        coords_z = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_z, coords_h, coords_w], indexing='ij'))  # (3, Wz, Wh, Ww)
        coords_flatten = torch.flatten(coords, 1)  # (3, Wz*Wh*Ww)

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (3, Wz*Wh*Ww, Wz*Wh*Ww)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (Wz*Wh*Ww, Wz*Wh*Ww, 3)

        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)

        relative_position_index = relative_coords.sum(-1)  # (Wz*Wh*Ww, Wz*Wh*Ww)
        self.register_buffer("relative_position_index", relative_position_index)

        # QKV projection (support both self-attention and cross-attention)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # For self-attention
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)  # For cross-attention query
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)  # For cross-attention key/value

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Initialize
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, x_kv=None, mask=None):
        """
        Args:
            x: (B*num_windows, Wz*Wh*Ww, C) - query input
            x_kv: (B*num_windows, Wz*Wh*Ww, C) - key/value input for cross-attention
                  If None, performs self-attention (Q=K=V from x)
            mask: (num_windows, Wz*Wh*Ww, Wz*Wh*Ww) or None

        Returns:
            output: (B*num_windows, Wz*Wh*Ww, C)
        """
        BW, N, C = x.shape  # N = Wz*Wh*Ww
        is_cross_attn = x_kv is not None

        if self.attention_type == 'standard':
            return self._standard_attention(x, x_kv, mask)
        elif self.attention_type == 'axial':
            return self._axial_attention(x, x_kv, mask)
        else:
            raise ValueError(f"Unknown attention type: {self.attention_type}")

    def _standard_attention(self, x, x_kv, mask):
        """Standard multi-head attention (self or cross)."""
        BW, N, C = x.shape
        is_cross_attn = x_kv is not None

        # QKV projection
        if is_cross_attn:
            # Cross-attention: Q from x, K/V from x_kv
            q = self.q_proj(x).reshape(BW, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            kv = self.kv_proj(x_kv).reshape(BW, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]  # (BW, num_heads, N, head_dim)
        else:
            # Self-attention: Q=K=V from x
            qkv = self.qkv(x).reshape(BW, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # (BW, num_heads, N, head_dim)

        # Attention
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # (BW, num_heads, N, N)

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            N, N, -1
        )  # (N, N, num_heads)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # (num_heads, N, N)
        attn = attn + relative_position_bias.unsqueeze(0)

        # Apply mask (for shifted window)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(BW // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Aggregate
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
            # Cross-attention: Q from x, K/V from x_kv
            q = self.q_proj(x).reshape(BW, N, self.num_heads, C // self.num_heads)
            q = q.view(BW, Wz, Wh, Ww, self.num_heads, C // self.num_heads)

            kv = self.kv_proj(x_kv).reshape(BW, N, 2, self.num_heads, C // self.num_heads)
            kv_spatial = kv.view(BW, Wz, Wh, Ww, 2, self.num_heads, C // self.num_heads)
            k, v = kv_spatial[..., 0, :, :], kv_spatial[..., 1, :, :]
        else:
            # Self-attention: Q=K=V from x
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
        out_z = attn_z @ v_z  # (bw*h*w, nh, z, d)
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


# ============================================================================
# Swin Transformer Block
# ============================================================================

class SwinTransformerBlock3D(nn.Module):
    """
    Swin Transformer Block with Window-based and Shifted-Window attention.

    Architecture:
        x -> [LN -> W-MSA -> residual] -> [LN -> MLP -> residual]
          -> [LN -> SW-MSA -> residual] -> [LN -> MLP -> residual] -> out

    Args:
        dim: feature dimension
        input_resolution: (Z, H, W)
        num_heads: number of attention heads
        window_size: (Wz, Wh, Ww)
        shift_size: (Sz, Sh, Sw) for shifted window (typically half of window_size)
        mlp_ratio: ratio of mlp hidden dim to embedding dim
        qkv_bias: whether to use bias in qkv projection
        drop: dropout rate
        attn_drop: attention dropout rate
        drop_path: stochastic depth rate
        attention_type: 'standard' or 'axial'
        use_checkpoint: whether to use gradient checkpointing to save memory
    """
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

        # Check window size
        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in [0, window_size)"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in [0, window_size)"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in [0, window_size)"

        # Normalization layers
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)

        # Attention layers
        self.attn_regular = WindowAttention3D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            attention_type=attention_type
        )

        self.attn_shifted = WindowAttention3D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            attention_type=attention_type
        )

        # MLP layers
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        # Drop path (stochastic depth)
        from timm.models.layers import DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Create attention mask for shifted window
        if any(s > 0 for s in self.shift_size):
            Z, H, W = self.input_resolution
            self.register_buffer("attn_mask", create_shifted_window_mask_3d(
                Z, H, W, self.window_size, self.shift_size
            ))
        else:
            self.attn_mask = None

    def _forward_part1(self, x):
        """Regular window attention + MLP."""
        B, C, Z, H, W = x.shape
        shortcut = x

        # Regular Window Attention
        x = rearrange(x, 'b c z h w -> b z h w c')
        x_flat = x.view(B, Z * H * W, C)
        x_flat = self.norm1(x_flat)
        x = x_flat.view(B, Z, H, W, C)
        x = rearrange(x, 'b z h w c -> b c z h w')

        x_windows = window_partition_3d(x, self.window_size)
        attn_windows = self.attn_regular(x_windows, mask=None)
        x = window_reverse_3d(attn_windows, self.window_size, Z, H, W)

        x = shortcut + self.drop_path(x)

        # MLP
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
        """Shifted window attention + MLP."""
        B, C, Z, H, W = x.shape
        shortcut = x

        # Shifted Window Attention
        x = rearrange(x, 'b c z h w -> b z h w c')
        x_flat = x.view(B, Z * H * W, C)
        x_flat = self.norm3(x_flat)
        x = x_flat.view(B, Z, H, W, C)
        x = rearrange(x, 'b z h w c -> b c z h w')

        # Cyclic shift
        if any(s > 0 for s in self.shift_size):
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]),
                                  dims=(2, 3, 4))
        else:
            shifted_x = x

        x_windows = window_partition_3d(shifted_x, self.window_size)
        attn_windows = self.attn_shifted(x_windows, mask=self.attn_mask)
        shifted_x = window_reverse_3d(attn_windows, self.window_size, Z, H, W)

        # Reverse cyclic shift
        if any(s > 0 for s in self.shift_size):
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]),
                          dims=(2, 3, 4))
        else:
            x = shifted_x

        x = shortcut + self.drop_path(x)

        # MLP
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
        """
        Args:
            x: (B, C, Z, H, W)
        Returns:
            output: (B, C, Z, H, W)
        """
        B, C, Z, H, W = x.shape
        assert C == self.dim, f"Input channel {C} doesn't match dim {self.dim}"

        if self.use_checkpoint and self.training:
            # Use gradient checkpointing to save memory
            x = torch.utils.checkpoint.checkpoint(self._forward_part1, x, use_reentrant=False)
            x = torch.utils.checkpoint.checkpoint(self._forward_part2, x, use_reentrant=False)
        else:
            x = self._forward_part1(x)
            x = self._forward_part2(x)

        return x


class SwinStage3D(nn.Module):
    """
    A stage of Swin Transformer consisting of multiple Swin blocks.

    Args:
        dim: feature dimension
        input_resolution: (Z, H, W)
        depth: number of Swin blocks in this stage
        num_heads: number of attention heads
        window_size: (Wz, Wh, Ww)
        mlp_ratio: ratio of mlp hidden dim to embedding dim
        qkv_bias: whether to use bias in qkv projection
        drop: dropout rate
        attn_drop: attention dropout rate
        drop_path: stochastic depth rate (can be a list)
        attention_type: 'standard' or 'axial'
        use_checkpoint: whether to use gradient checkpointing
    """
    def __init__(self, dim, input_resolution, depth, num_heads, window_size=(5, 8, 8),
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 attention_type='standard', use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # Build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else (window_size[0] // 2, window_size[1] // 2, window_size[2] // 2),
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                attention_type=attention_type,
                use_checkpoint=use_checkpoint
            )
            for i in range(depth)
        ])

    def forward(self, x):
        """
        Args:
            x: (B, C, Z, H, W)
        Returns:
            output: (B, C, Z, H, W)
        """
        for blk in self.blocks:
            x = blk(x)
        return x


# ============================================================================
# Patch Merging & Expanding (for U-Net style architecture)
# ============================================================================

class PatchMerging3D(nn.Module):
    """
    3D Patch Merging Layer for downsampling.

    Merges 2x2x2 neighboring patches and projects to 2C dimensions.
    Spatial resolution: (Z, H, W) → (Z/2, H/2, W/2)
    Channel dimension: C → 2C

    Similar to Swin Transformer's patch merging but extended to 3D.
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        # Merge 2x2x2 = 8 patches, then project to 2*dim
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(8 * dim)

    def forward(self, x):
        """
        Args:
            x: (B, C, Z, H, W)
        Returns:
            (B, 2C, Z/2, H/2, W/2)
        """
        B, C, Z, H, W = x.shape

        # Ensure dimensions are even
        assert Z % 2 == 0 and H % 2 == 0 and W % 2 == 0, \
            f"Spatial dimensions must be even, got Z={Z}, H={H}, W={W}"

        # Reshape to merge 2x2x2 patches
        # (B, C, Z, H, W) -> (B, C, Z/2, 2, H/2, 2, W/2, 2)
        x = x.view(B, C, Z // 2, 2, H // 2, 2, W // 2, 2)

        # Permute to (B, Z/2, H/2, W/2, 2, 2, 2, C)
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()

        # Flatten the 2x2x2 patches: (B, Z/2, H/2, W/2, 8*C)
        x = x.view(B, Z // 2, H // 2, W // 2, 8 * C)

        # Normalize and reduce
        x = self.norm(x)
        x = self.reduction(x)  # (B, Z/2, H/2, W/2, 2C)

        # Permute back to (B, 2C, Z/2, H/2, W/2)
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        return x


class PatchExpanding3D(nn.Module):
    """
    3D Patch Expanding Layer for upsampling.

    Expands each patch to 2x2x2 patches and reduces channel dimension.
    Spatial resolution: (Z, H, W) → (2Z, 2H, 2W)
    Channel dimension: C → C/2

    Used in U-Net decoder for upsampling.
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        # Expand to 8x channels first, then reshape to 2x2x2 spatial
        self.expand = nn.Linear(dim, 8 * (dim // 2), bias=False)
        self.norm = norm_layer(dim // 2)

    def forward(self, x):
        """
        Args:
            x: (B, C, Z, H, W)
        Returns:
            (B, C/2, 2Z, 2H, 2W)
        """
        B, C, Z, H, W = x.shape

        # Permute to (B, Z, H, W, C)
        x = x.permute(0, 2, 3, 4, 1).contiguous()

        # Expand: (B, Z, H, W, C) -> (B, Z, H, W, 8 * C/2)
        x = self.expand(x)

        # Reshape to (B, Z, H, W, 2, 2, 2, C/2)
        x = x.view(B, Z, H, W, 2, 2, 2, C // 2)

        # Permute to (B, Z, 2, H, 2, W, 2, C/2)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()

        # Reshape to (B, 2Z, 2H, 2W, C/2)
        x = x.view(B, Z * 2, H * 2, W * 2, C // 2)

        # Normalize
        x = self.norm(x)

        # Permute back to (B, C/2, 2Z, 2H, 2W)
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        return x


class PatchExpandingConcat3D(nn.Module):
    """
    3D Patch Expanding with concatenation for skip connection.

    After expanding, concatenates with skip connection from encoder,
    then projects back to target dimension.

    Used in U-Net decoder with skip connections.
    """
    def __init__(self, dim, out_dim=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim or dim // 2

        # Expand spatial resolution
        self.expand = PatchExpanding3D(dim, norm_layer)

        # After concatenation with skip: dim/2 + skip_dim -> out_dim
        # Assuming skip_dim = dim/2 (same as expanded dim)
        self.concat_proj = nn.Sequential(
            nn.Conv3d(dim // 2 + dim // 2, self.out_dim, kernel_size=1),
            nn.GroupNorm(8, self.out_dim),
            nn.GELU()
        )

    def forward(self, x, skip):
        """
        Args:
            x: (B, C, Z, H, W) - from previous decoder stage
            skip: (B, C/2, 2Z, 2H, 2W) - from encoder skip connection
        Returns:
            (B, out_dim, 2Z, 2H, 2W)
        """
        # Expand: (B, C, Z, H, W) -> (B, C/2, 2Z, 2H, 2W)
        x = self.expand(x)

        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)  # (B, C, 2Z, 2H, 2W)

        # Project to output dimension
        x = self.concat_proj(x)

        return x


if __name__ == "__main__":
    # Quick test
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Testing 3D Swin Blocks...")
    print("=" * 60)

    # Test window partition
    B, C, Z, H, W = 2, 64, 65, 128, 128
    window_size = (5, 8, 8)

    x = torch.randn(B, C, Z, H, W).to(device)
    print(f"Input shape: {x.shape}")

    windows = window_partition_3d(x, window_size)
    print(f"Windows shape: {windows.shape}")
    print(f"Expected: ({B * (Z//5) * (H//8) * (W//8)}, {5*8*8}, {C})")

    x_reversed = window_reverse_3d(windows, window_size, Z, H, W)
    print(f"Reversed shape: {x_reversed.shape}")
    print(f"Reconstruction error: {(x - x_reversed).abs().max().item():.6f}")

    print("\n" + "=" * 60)
    print("Testing WindowAttention3D (standard)...")
    attn = WindowAttention3D(dim=C, window_size=window_size, num_heads=8, attention_type='standard').to(device)
    out = attn(windows[:10])  # Test on first 10 windows
    print(f"Attention output shape: {out.shape}")

    print("\n" + "=" * 60)
    print("Testing WindowAttention3D (axial)...")
    attn_axial = WindowAttention3D(dim=C, window_size=window_size, num_heads=8, attention_type='axial').to(device)
    out_axial = attn_axial(windows[:10])
    print(f"Axial attention output shape: {out_axial.shape}")

    print("\n" + "=" * 60)
    print("Testing SwinTransformerBlock3D...")
    block = SwinTransformerBlock3D(
        dim=C,
        input_resolution=(Z, H, W),
        num_heads=8,
        window_size=window_size,
        shift_size=(2, 4, 4),
        attention_type='standard'
    ).to(device)

    out_block = block(x)
    print(f"Block output shape: {out_block.shape}")
    print(f"Shape preserved: {out_block.shape == x.shape}")

    print("\n" + "=" * 60)
    print("Testing SwinStage3D...")
    stage = SwinStage3D(
        dim=C,
        input_resolution=(Z, H, W),
        depth=2,
        num_heads=8,
        window_size=window_size,
        attention_type='axial'
    ).to(device)

    out_stage = stage(x)
    print(f"Stage output shape: {out_stage.shape}")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
