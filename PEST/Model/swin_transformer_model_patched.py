"""
3D Swin Transformer with Patch Embedding for Memory-Efficient Physics-Informed Reconstruction

Task: 5→5 prediction with sparse-to-dense reconstruction
- Input: 5 dense frames (B, 5, 4, 64, 128, 128)
- Output: 5 frames with sparse observation → predict full 64 layers

Key Changes from Original:
1. **Patch Embedding**: Reduces spatial resolution 32x: (64,128,128) → (32,32,32)
2. **Memory Efficient**: ~32x reduction in intermediate feature map size
3. **Patch Recovery**: Learned upsampling back to (64,128,128)
4. **Coordinated Window Size**: (8, 8, 8) divides patched resolution perfectly

Architecture:
1. Encoder: Patch embed → 3D Swin Transformer in patch space
2. Temporal Attention: Model temporal evolution
3. Decoder: Process in patch space → Patch recovery to full resolution

Compatible with:
- MultiplaneSparseDataset (same input/output format)
- Physics loss framework
- UnifiedTrainer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import sys
sys.path.append('/home/ydai17/Turbo/Code')

from Model.modules.swin_blocks import SwinStage3D, window_partition_3d, window_reverse_3d


# ============================================================================
# Patch Embedding & Recovery (NEW!)
# ============================================================================

class PatchEmbed3D(nn.Module):
    """
    3D Patch Embedding using strided convolution.

    Reduces spatial resolution: (64, 128, 128) → (32, 32, 32)
    Patch size: (2, 4, 4) → 32x memory reduction!
    """
    def __init__(self, patch_size=(2, 4, 4), in_channels=4, embed_dim=96):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size  # No overlap - each patch processed once
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Args:
            x: (B, C, Z, H, W) = (B, 4, 64, 128, 128)
        Returns:
            (B, D, Z', H', W') = (B, D, 32, 32, 32)
        """
        x = self.proj(x)  # Strided conv: (B, D, 32, 32, 32)

        # Apply norm (optional but helps training)
        B, C, Z, H, W = x.shape
        x = rearrange(x, 'b c z h w -> b z h w c')
        x = self.norm(x)
        x = rearrange(x, 'b z h w c -> b c z h w')

        return x


class PatchEmbedSparse(nn.Module):
    """
    Patch Embedding for sparse observations with position-aware z-interpolation.

    Sparse input: (B, 4, 12, 128, 128) - only 12 z-layers observed
    Output: (B, D, 32, 32, 32) - matching encoder output via position-aware interpolation

    Key improvement: Use sparse_indices to correctly map sparse layers to their
    true positions in patch space, instead of assuming uniform distribution.

    Process:
    1. Patch embed H, W: (B, 4, 12, 128, 128) → (B, D, 12, 32, 32)
    2. Position-aware interpolate z: 12 sparse positions → 32 patch positions
    """
    def __init__(self, patch_size_hw=(4, 4), in_channels=4, embed_dim=96,
                 num_sparse_z_layers=12, patched_z=32, total_z=64):
        super().__init__()
        self.num_sparse_z_layers = num_sparse_z_layers
        self.patched_z = patched_z
        self.total_z = total_z
        self.patch_size_z = total_z // patched_z  # 64/32 = 2

        # Patch embed only H, W: (B, 4, 12, 128, 128) → (B, D, 12, 32, 32)
        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=(1, patch_size_hw[0], patch_size_hw[1]),
            stride=(1, patch_size_hw[0], patch_size_hw[1])
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, sparse_indices=None):
        """
        Args:
            x: (B, C, num_sparse_z_layers, H, W) = (B, 4, 12, 128, 128)
            sparse_indices: list of int - original z indices [0, 5, 10, ...]
        Returns:
            (B, D, patched_z, H', W') = (B, D, 32, 32, 32)
        """
        # Patch embed H, W
        x = self.proj(x)  # (B, D, 12, 32, 32)

        B, C, Z_sparse, H, W = x.shape
        x = rearrange(x, 'b c z h w -> b z h w c')
        x = self.norm(x)
        x = rearrange(x, 'b z h w c -> b c z h w')

        # Position-aware z interpolation
        if Z_sparse != self.patched_z:
            if sparse_indices is not None:
                # 使用 grid_sample 进行位置感知插值
                x = self._position_aware_interpolate(x, sparse_indices)
            else:
                # Fallback: 简单的 trilinear 插值
                x = F.interpolate(x, size=(self.patched_z, H, W),
                                mode='trilinear', align_corners=False)

        return x

    def _position_aware_interpolate(self, x, sparse_indices):
        """
        位置感知的 z 轴插值

        将 sparse 层根据其真实 z 位置映射到 patch space，然后插值填充。

        Args:
            x: (B, C, num_sparse, H, W) - sparse 层的特征
            sparse_indices: list - sparse 层的原始 z 索引 [0, 5, 10, ...]

        Returns:
            (B, C, patched_z, H, W) - 完整的 patch space 特征
        """
        B, C, num_sparse, H, W = x.shape
        device = x.device

        # 计算 sparse 层在 patch space 中的位置
        # 原始 z=k → patch z = k / patch_size_z
        sparse_patch_positions = torch.tensor(
            [idx / self.patch_size_z for idx in sparse_indices],
            device=device, dtype=torch.float32
        )  # e.g., [0, 2.5, 5, 7.5, ...]

        # 创建输出 tensor
        output = torch.zeros(B, C, self.patched_z, H, W, device=device, dtype=x.dtype)

        # 对每个 patch z 位置，找到最近的两个 sparse 层进行线性插值
        for pz in range(self.patched_z):
            pz_float = float(pz)

            # 找到 pz 在 sparse_patch_positions 中的插值位置
            # 即找到 left 和 right 使得 sparse_patch_positions[left] <= pz <= sparse_patch_positions[right]

            # 找到右边界
            right_idx = 0
            for i, pos in enumerate(sparse_patch_positions):
                if pos >= pz_float:
                    right_idx = i
                    break
                right_idx = i + 1

            if right_idx == 0:
                # pz 在第一个 sparse 层之前，使用第一层
                output[:, :, pz, :, :] = x[:, :, 0, :, :]
            elif right_idx >= num_sparse:
                # pz 在最后一个 sparse 层之后，使用最后一层
                output[:, :, pz, :, :] = x[:, :, -1, :, :]
            else:
                # 线性插值
                left_idx = right_idx - 1
                left_pos = sparse_patch_positions[left_idx].item()
                right_pos = sparse_patch_positions[right_idx].item()

                if right_pos == left_pos:
                    weight = 0.5
                else:
                    weight = (pz_float - left_pos) / (right_pos - left_pos)

                output[:, :, pz, :, :] = (1 - weight) * x[:, :, left_idx, :, :] + weight * x[:, :, right_idx, :, :]

        return output


class PatchRecovery3D(nn.Module):
    """
    Recover from patch space back to full resolution.

    Uses transposed convolution (learnable upsampling):
    (B, D, 32, 32, 32) → (B, 4, 64, 128, 128)
    """
    def __init__(self, patch_size=(2, 4, 4), embed_dim=96, out_channels=4):
        super().__init__()
        self.patch_size = patch_size

        # Transposed conv for upsampling
        self.proj = nn.ConvTranspose3d(
            embed_dim, out_channels,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        """
        Args:
            x: (B, D, 32, 32, 32)
        Returns:
            (B, 4, 64, 128, 128)
        """
        x = self.proj(x)
        return x


def interpolate_sparse_to_full(sparse_obs, sparse_indices, total_z=64):
    """
    在原始分辨率进行 z-方向线性插值。

    这是关键的 skip connection 基础！插值结果非常接近 GT，
    模型只需要学习残差即可。

    Args:
        sparse_obs: (B, C, num_sparse, H, W) - sparse 层的观测值
        sparse_indices: list - sparse 层的 z 索引 [0, 5, 10, ...]
        total_z: int - 输出的 z 层数 (64)

    Returns:
        interpolated: (B, C, total_z, H, W) - 插值后的完整场
    """
    B, C, num_sparse, H, W = sparse_obs.shape
    device = sparse_obs.device
    dtype = sparse_obs.dtype

    # 创建输出 tensor
    output = torch.zeros(B, C, total_z, H, W, device=device, dtype=dtype)

    # 将 sparse_indices 转为 tensor (使用 as_tensor 避免警告)
    sparse_z = torch.as_tensor(sparse_indices, device=device, dtype=torch.float32)

    # 对每个 z 位置进行插值
    for z in range(total_z):
        z_float = float(z)

        # 找到右边界索引
        right_idx = 0
        for i, pos in enumerate(sparse_z):
            if pos >= z_float:
                right_idx = i
                break
            right_idx = i + 1

        if right_idx == 0:
            # z 在第一个 sparse 层之前，使用第一层
            output[:, :, z, :, :] = sparse_obs[:, :, 0, :, :]
        elif right_idx >= num_sparse:
            # z 在最后一个 sparse 层之后，使用最后一层
            output[:, :, z, :, :] = sparse_obs[:, :, -1, :, :]
        else:
            # 线性插值
            left_idx = right_idx - 1
            left_pos = sparse_z[left_idx].item()
            right_pos = sparse_z[right_idx].item()

            if right_pos == left_pos:
                weight = 0.5
            else:
                weight = (z_float - left_pos) / (right_pos - left_pos)

            output[:, :, z, :, :] = (1 - weight) * sparse_obs[:, :, left_idx, :, :] + \
                                     weight * sparse_obs[:, :, right_idx, :, :]

    return output


# ============================================================================
# Position Encoding (Updated for Patched Resolution)
# ============================================================================

class PositionEncoding3D(nn.Module):
    """3D position encoding for PATCHED dimensions (32, 32, 32)."""
    def __init__(self, dim, max_z=32, max_h=32, max_w=32):
        super().__init__()
        # Distribute dimensions evenly
        dim_z = dim // 3
        dim_h = dim // 3
        dim_w = dim - dim_z - dim_h

        self.z_embed = nn.Parameter(torch.randn(1, dim_z, max_z, 1, 1) * 0.02)
        self.h_embed = nn.Parameter(torch.randn(1, dim_h, 1, max_h, 1) * 0.02)
        self.w_embed = nn.Parameter(torch.randn(1, dim_w, 1, 1, max_w) * 0.02)

    def forward(self, x):
        """
        Args:
            x: (B, C, Z, H, W) in patch space
        Returns:
            x + position encoding
        """
        B, C, Z, H, W = x.shape
        z_enc = self.z_embed[:, :, :Z, :, :]
        h_enc = self.h_embed[:, :, :, :H, :]
        w_enc = self.w_embed[:, :, :, :, :W]

        pos_enc = torch.cat([
            z_enc.expand(B, -1, Z, H, W),
            h_enc.expand(B, -1, Z, H, W),
            w_enc.expand(B, -1, Z, H, W)
        ], dim=1)

        return x + pos_enc


class TemporalPositionEncoding(nn.Module):
    """Temporal position encoding for time dimension."""
    def __init__(self, dim, max_t=10):
        super().__init__()
        self.t_embed = nn.Parameter(torch.randn(1, max_t, dim) * 0.02)

    def forward(self, x):
        B, T, D = x.shape
        return x + self.t_embed[:, :T, :D]


# ============================================================================
# Encoder: Process Dense Input in Patch Space
# ============================================================================

class SwinEncoder(nn.Module):
    """
    Encode dense input frames using 3D Swin Transformer IN PATCH SPACE.

    Input: (B, T=5, C=4, Z=64, H=128, W=128)
    Patch embed: → (B, T=5, D, Z'=32, H'=32, W'=32)
    Process with Swin blocks in patch space
    Output: (B, T=5, D, 32, 32, 32) - 32x smaller!
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        in_channels = config.IN_CHANNELS  # 4
        embed_dim = config.EMBED_DIM  # e.g., 96
        depths = config.DEPTHS
        num_heads = config.NUM_HEADS
        # Support per-stage window sizes (WINDOW_SIZES) or single window size (WINDOW_SIZE)
        window_sizes = getattr(config, 'WINDOW_SIZES', None)
        if window_sizes is None:
            window_size = config.WINDOW_SIZE  # (8, 8, 8) for patched resolution
            window_sizes = [window_size] * len(depths)  # Same size for all stages
        self.window_sizes = window_sizes  # Store for debugging
        mlp_ratio = config.MLP_RATIO
        drop_rate = config.DROP_RATE
        attn_drop_rate = config.ATTN_DROP_RATE
        drop_path_rate = config.DROP_PATH_RATE
        attention_type = config.ATTENTION_TYPE
        patch_size = config.PATCH_SIZE  # (2, 4, 4)

        # Patch embedding: (64,128,128) → (32,32,32)
        self.patch_embed = PatchEmbed3D(patch_size, in_channels, embed_dim)

        # Position encoding for PATCHED resolution
        patched_resolution = config.PATCHED_RESOLUTION  # (32, 32, 32)
        self.pos_embed = PositionEncoding3D(
            embed_dim,
            max_z=patched_resolution[0],
            max_h=patched_resolution[1],
            max_w=patched_resolution[2]
        )
        self.pos_drop = nn.Dropout(drop_rate)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build Swin stages (operate in patch space)
        # Support progressive windows: different window size per stage
        use_checkpoint = getattr(config, 'USE_GRADIENT_CHECKPOINTING', False)
        self.stages = nn.ModuleList()
        for i_stage in range(len(depths)):
            stage_window_size = window_sizes[i_stage]  # Per-stage window size
            stage = SwinStage3D(
                dim=embed_dim,
                input_resolution=patched_resolution,  # (32, 32, 32)
                depth=depths[i_stage],
                num_heads=num_heads[i_stage],
                window_size=stage_window_size,  # Progressive window support
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_stage]):sum(depths[:i_stage + 1])],
                attention_type=attention_type,
                use_checkpoint=use_checkpoint
            )
            self.stages.append(stage)

        self.num_features = embed_dim
        self.norm = nn.LayerNorm(self.num_features)

    def forward(self, x):
        """
        Args:
            x: (B, T, C, Z, H, W) = (B, 5, 4, 64, 128, 128)
        Returns:
            features: (B, T, D, Z', H', W') = (B, 5, D, 32, 32, 32)
        """
        B, T, C, Z, H, W = x.shape

        # Process each timestep independently
        features_list = []
        for t in range(T):
            x_t = x[:, t]  # (B, C, Z, H, W)

            # Patch embedding: (B, 4, 64, 128, 128) → (B, D, 32, 32, 32)
            x_t = self.patch_embed(x_t)

            # Position encoding
            x_t = self.pos_embed(x_t)
            x_t = self.pos_drop(x_t)

            # Apply Swin stages in patch space
            for stage in self.stages:
                x_t = stage(x_t)

            # Normalization
            x_t = rearrange(x_t, 'b c z h w -> b z h w c')
            x_t = self.norm(x_t)
            x_t = rearrange(x_t, 'b z h w c -> b c z h w')

            features_list.append(x_t)

        # Stack along time
        features = torch.stack(features_list, dim=1)  # (B, T, D, 32, 32, 32)

        return features


# ============================================================================
# Temporal Attention: Model Time Evolution
# ============================================================================

class TemporalAttention(nn.Module):
    """
    Temporal self-attention in patch space.

    For each spatial location (z', h', w') in patch space,
    apply attention across time T=5.
    """
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
        """
        Args:
            x: (B, T, D, Z', H', W') in patch space
        Returns:
            (B, T, D, Z', H', W')
        """
        B, T, D, Z, H, W = x.shape

        # Reshape: treat each spatial location as sequence over time
        x = rearrange(x, 'b t d z h w -> (b z h w) t d')  # (B*Z'*H'*W', T, D)

        # Temporal position encoding
        x = self.temporal_pos(x)
        x = self.norm(x)

        # QKV
        N = x.shape[0]
        qkv = self.qkv(x).reshape(N, T, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention across time
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(N, T, D)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Reshape back
        x = rearrange(x, '(b z h w) t d -> b t d z h w', b=B, z=Z, h=H, w=W)

        return x


# ============================================================================
# Decoder: Sparse to Dense Reconstruction (IN PATCH SPACE + RECOVERY)
# ============================================================================

class SwinDecoder(nn.Module):
    """
    Decoder with patch embedding and recovery.

    Flow:
    1. Patch embed sparse obs: (B, 4, 12, 128, 128) → (B, D, 32, 32, 32) via interpolation
    2. Process in patch space with Swin blocks
    3. Patch recovery: (B, D, 32, 32, 32) → (B, 4, 64, 128, 128)

    Supports dual residual mode (when config.DUAL_RESIDUAL=True):
    - Temporal prior: input_last as base
    - Sparse hint: z_interp(sparse) - temporal_base
    - Output: temporal_base + delta
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Check if dual residual mode is enabled
        self.dual_residual = getattr(config, 'DUAL_RESIDUAL', False)

        dim = config.DECODER_DIM
        num_heads = config.DECODER_NUM_HEADS
        window_size = config.WINDOW_SIZE  # (8, 8, 8)
        attention_type = config.ATTENTION_TYPE
        patch_size = config.PATCH_SIZE
        patched_resolution = config.PATCHED_RESOLUTION  # (32, 32, 32)

        # Patch embedding for sparse observations with position-aware z-interpolation
        self.sparse_patch_embed = PatchEmbedSparse(
            patch_size_hw=(4, 4),
            in_channels=config.IN_CHANNELS,
            embed_dim=dim,
            num_sparse_z_layers=config.NUM_SPARSE_Z_LAYERS,  # 12 for DNS, 16 for JHU
            patched_z=patched_resolution[0],  # 32
            total_z=config.INPUT_RESOLUTION[0]  # 64 for DNS, 128 for JHU
        )

        # Project encoder features to decoder dim
        self.encoder_proj = nn.Conv3d(config.EMBED_DIM, dim, kernel_size=1)

        # For dual residual: additional embedding for temporal base
        if self.dual_residual:
            self.temporal_patch_embed = PatchEmbed3D(patch_size, config.IN_CHANNELS, dim)
            # hint_feat = sparse_feat - temporal_feat (在patch空间计算)
            # 复用 sparse_patch_embed，不需要额外的 hint_patch_embed
            # Fusion layer
            self.dual_fusion = nn.Sequential(
                nn.Conv3d(dim * 3, dim, kernel_size=1),  # temporal + hint + encoder
                nn.GroupNorm(8, dim),
                nn.GELU()
            )
            # Learnable delta scale
            delta_scale_init = getattr(config, 'DELTA_SCALE_INIT', 0.1)
            self.delta_scale = nn.Parameter(torch.ones(1) * delta_scale_init)

        # Cross attention in patch space
        from Model.modules.swin_blocks import WindowAttention3D
        self.cross_attn = WindowAttention3D(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=True,
            attention_type=attention_type
        )
        self.cross_attn_norm = nn.LayerNorm(dim)

        # Refinement in patch space
        self.refinement = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, dim),
            nn.GELU(),
            nn.Conv3d(dim, dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, dim),
            nn.GELU()
        )

        # Patch recovery: (32, 32, 32) → (64, 128, 128)
        self.patch_recovery = PatchRecovery3D(patch_size, dim, config.IN_CHANNELS)

    def forward(self, sparse_obs, encoder_features, sparse_indices, input_last=None):
        """
        Args:
            sparse_obs: (B, T_out, C, 12, H, W) - sparse observations
            encoder_features: (B, T_in, D, 32, 32, 32) - from encoder in patch space
            sparse_indices: list of z indices - [0, 5, 10, ...] for position-aware interpolation
            input_last: (B, C, Z, H, W) - last input frame (for dual residual mode)
        Returns:
            output: (B, T_out, C, 64, 128, 128) - full predicted field
            delta: (B, T_out, C, 64, 128, 128) - learned delta (only in dual_residual mode)

        关键改进 (Skip Connection):
        - 先在原始分辨率做 z-插值，作为 baseline (非常接近 GT!)
        - 模型只学习残差 delta
        - output = interpolated_baseline + delta
        - 这避免了从头重建，消除 block artifacts

        Dual Residual Mode (when self.dual_residual=True):
        双残差学习，结合空间和时间信息:
        - 空间先验: interpolated_baseline (当前时刻稀疏观测的z插值)
        - 时间提示: hint_feat = sparse_feat - temporal_feat (从上一帧到当前的变化方向)
        - 网络输入: sparse_feat + hint_feat + encoder_context → 学习 delta
        - output = interpolated_baseline + delta * scale

        注: 两种模式的 base 都是 interpolated_baseline，
        Dual Residual 的优势在于利用时间信息帮助学习更准确的 delta
        """
        B, T_out, C, num_sparse_z_layers, H, W = sparse_obs.shape
        total_z = self.config.INPUT_RESOLUTION[0]  # 64 or 128

        # Average encoder features over time
        encoder_context = encoder_features.mean(dim=1)  # (B, D_enc, 32, 32, 32)
        encoder_context = self.encoder_proj(encoder_context)  # (B, dim, 32, 32, 32)

        outputs = []
        deltas = []

        for t in range(T_out):
            sparse_t = sparse_obs[:, t]  # (B, 4, 12/16, H, W)

            # ============================================================
            # 空间先验: 稀疏观测的z插值 (两种模式都需要)
            # ============================================================
            interpolated_baseline = interpolate_sparse_to_full(
                sparse_t, sparse_indices, total_z
            )  # (B, 4, 64/128, H, W)

            if self.dual_residual and input_last is not None:
                # ============================================================
                # DUAL RESIDUAL MODE: 空间残差 + 时间残差
                # - 空间残差: delta 相对于 interpolated_baseline
                # - 时间残差: hint = sparse_interp - temporal_base (变化方向)
                # ============================================================
                temporal_base = input_last  # (B, C, Z, H, W)

                # Patch embed temporal_base
                temporal_feat = self.temporal_patch_embed(temporal_base)  # (B, dim, Z', H', W')

                # Patch embed sparse observations (包含z插值信息)
                sparse_feat = self.sparse_patch_embed(sparse_t, sparse_indices)  # (B, dim, Z', H', W')

                # 时间残差hint: 从temporal_base到当前时刻的变化方向
                hint_feat = sparse_feat - temporal_feat  # (B, dim, Z', H', W')

                # Fuse: sparse(空间先验) + hint(时间变化) + encoder(历史context)
                fused = torch.cat([sparse_feat, hint_feat, encoder_context], dim=1)
                full_field = self.dual_fusion(fused)  # (B, dim, Z', H', W')

            else:
                # ============================================================
                # ORIGINAL MODE: spatial residual only
                # ============================================================
                sparse_t_patched = self.sparse_patch_embed(sparse_t, sparse_indices)
                full_field = sparse_t_patched + encoder_context

            # Cross attention in patch space
            query_windows = window_partition_3d(full_field, self.config.WINDOW_SIZE)
            kv_windows = window_partition_3d(encoder_context, self.config.WINDOW_SIZE)

            query_windows = self.cross_attn_norm(query_windows)
            kv_windows = self.cross_attn_norm(kv_windows)

            attn_out = self.cross_attn(query_windows, kv_windows)
            attn_out = window_reverse_3d(
                attn_out,
                self.config.WINDOW_SIZE,
                self.config.PATCHED_RESOLUTION[0],
                self.config.PATCHED_RESOLUTION[1],
                self.config.PATCHED_RESOLUTION[2]
            )

            full_field = full_field + attn_out

            # Refinement in patch space
            full_field = full_field + self.refinement(full_field)

            # Patch recovery: → delta
            delta = self.patch_recovery(full_field)

            # 两种模式都使用 interpolated_baseline 作为 base
            # Dual residual 的区别在于：网络额外利用了 temporal hint 来学习 delta
            if self.dual_residual and input_last is not None:
                # Dual residual: delta 经过 scale 调制
                delta = delta * self.delta_scale

            # output = 空间先验(interpolated_baseline) + 学到的修正(delta)
            output_t = interpolated_baseline + delta

            outputs.append(output_t)
            deltas.append(delta)

        output = torch.stack(outputs, dim=1)
        delta_stack = torch.stack(deltas, dim=1)

        return output, delta_stack


# ============================================================================
# Complete Model
# ============================================================================

class SwinTransformerPhysics(nn.Module):
    """
    Complete 3D Swin Transformer with Patch Embedding.

    Memory Efficient: 32x reduction via patching
    - Original: (64, 128, 128) → 1,048,576 pixels
    - Patched: (32, 32, 32) → 32,768 pixels (32x smaller!)

    Compatible with existing training pipeline.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Encoder (operates in patch space)
        self.encoder = SwinEncoder(config)

        # Temporal attention (operates in patch space)
        self.temporal_attn = TemporalAttention(
            dim=config.EMBED_DIM,
            num_heads=config.TEMPORAL_NUM_HEADS,
            attn_drop=config.ATTN_DROP_RATE,
            proj_drop=config.DROP_RATE
        )

        # Decoder (patch space → recovery to full resolution)
        self.decoder = SwinDecoder(config)

    def forward(self, batch):
        """
        Args:
            batch: dict with keys:
                - 'input_dense': (B, 5, 4, 64, 128, 128) - 前5帧完整GT
                - 'output_sparse_z': (B, 5, 4, 12, 128, 128) - 后5帧稀疏GT
                - 'sparse_indices': dict with 'z' key
        Returns:
            predictions: dict with:
                - 'output_full': (B, 5, 4, 64, 128, 128)
                - 'delta': (B, 5, 4, 64, 128, 128) - learned delta
                - 'delta_scale': float (only in dual_residual mode)

        数据流:
        1. input_dense (5帧完整GT) → Encoder → encoder_features
        2. encoder_features → Temporal Attention → temporal_features
        3. output_sparse_z (5帧稀疏GT) + temporal_features → Decoder → output_full
           - Sparse信息通过 PatchEmbedSparse 进行 z-插值融合
        4. 强制替换 sparse 层 = GT

        Dual Residual Mode:
        - 额外传入 input_last = input_dense[:, -1] 作为时间基准
        - Decoder利用时间信息帮助学习 delta (输出仍是 interpolated_baseline + delta)
        """
        input_dense = batch['input_dense']
        output_sparse = batch['output_sparse_z']
        sparse_indices = batch['sparse_indices']['z']

        # 确保sparse_indices是1D (batch中所有样本使用相同的sparse indices)
        if sparse_indices.dim() > 1:
            sparse_indices = sparse_indices[0]

        # Step 1: Encode 前5帧完整GT in patch space
        # (B, 5, 4, 64, 128, 128) → (B, 5, D, 32, 32, 32)
        encoder_features = self.encoder(input_dense)

        # Step 2: Temporal attention in patch space
        # 建模时间演化: 从前5帧到后5帧
        temporal_features = self.temporal_attn(encoder_features)

        # Step 3: Decode with sparse GT guidance
        # For dual residual mode, pass input_last as temporal base
        input_last = input_dense[:, -1] if self.decoder.dual_residual else None

        output_full, delta = self.decoder(
            output_sparse, temporal_features, sparse_indices, input_last
        )

        # Step 4: 强制将 sparse 层替换为 GT 值
        # 这确保观测层的预测值 == GT，只有 missing 层是网络预测的
        for idx, z_idx in enumerate(sparse_indices):
            output_full[:, :, :, z_idx, :, :] = output_sparse[:, :, :, idx, :, :]

        result = {
            'output_full': output_full,
            'delta': delta,
            'encoder_features': encoder_features,
            'temporal_features': temporal_features
        }

        # Add delta_scale if in dual residual mode
        if self.decoder.dual_residual:
            result['delta_scale'] = self.decoder.delta_scale.item()

        return result

    def get_num_params(self):
        """Calculate number of parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Testing Patched SwinTransformerPhysics...")
    print("=" * 80)

    # Create config with patching
    class DummyConfig:
        IN_CHANNELS = 4
        EMBED_DIM = 96
        DEPTHS = [2, 2]  # Reduced for testing
        NUM_HEADS = [3, 6]
        PATCH_SIZE = (2, 4, 4)  # Updated for 64 z-layers
        INPUT_RESOLUTION = (64, 128, 128)  # Original resolution
        PATCHED_RESOLUTION = (32, 32, 32)  # After patching
        WINDOW_SIZE = (8, 8, 8)  # Divides patched resolution perfectly
        MLP_RATIO = 4.0
        DROP_RATE = 0.0
        ATTN_DROP_RATE = 0.0
        DROP_PATH_RATE = 0.1
        ATTENTION_TYPE = 'standard'  # or 'axial'
        TEMPORAL_NUM_HEADS = 8
        DECODER_DIM = 128
        DECODER_NUM_HEADS = 8
        NUM_SPARSE_Z_LAYERS = 12  # total count of sparse layers

    config = DummyConfig()
    model = SwinTransformerPhysics(config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    print(f"✓ Model created!")
    print(f"  Parameters: {model.get_num_params() / 1e6:.2f}M")
    print(f"\n  Patch configuration:")
    print(f"    Original resolution: {config.INPUT_RESOLUTION}")
    print(f"    Patched resolution:  {config.PATCHED_RESOLUTION} (32x smaller!)")
    print(f"    Patch size: {config.PATCH_SIZE}")
    print(f"    Window size: {config.WINDOW_SIZE}")

    # Test forward pass
    B, T_in, T_out = 2, 5, 5
    batch = {
        'input_dense': torch.randn(B, T_in, 4, 64, 128, 128).to(device),
        'output_sparse_z': torch.randn(B, T_out, 4, 12, 128, 128).to(device),
        'sparse_indices': {'z': list(range(0, 64, 5))[:12]}  # 12 sparse layers
    }

    print(f"\nInput shapes:")
    print(f"  input_dense: {batch['input_dense'].shape}")
    print(f"  output_sparse_z: {batch['output_sparse_z'].shape}")

    print(f"\nRunning forward pass...")
    with torch.no_grad():
        output = model(batch)

    print(f"\nOutput shapes:")
    print(f"  output_full: {output['output_full'].shape}")
    print(f"  encoder_features (in patch space): {output['encoder_features'].shape}")

    print("\n" + "=" * 80)
    print("✓ All tests passed! Model works with patching.")
    print(f"✓ Memory savings: ~32x reduction in feature map size")
