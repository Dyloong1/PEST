#!/usr/bin/env python3
"""
Autoregressive Test Script for Swin-DiT/UNet + PDE-Refiner Models

Tests the model's ability to perform multi-round autoregressive prediction:
- Round 1: Input t0-t4 (GT), sparse t5-t9 GT -> Predict t5-t9 dense
- Round 2: Input t5-t9 (pred), sparse t10-t14 GT -> Predict t10-t14 dense
- Round 3: Input t10-t14 (pred), sparse t15-t19 GT -> Predict t15-t19 dense

Uses K=3 step iterative refinement (PDE-Refiner style).
Supports both Swin-DiT and Swin-UNet architectures.

FIXED: Now properly loads consecutive time windows for each AR round.
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
import sys

CODE_ROOT = Path(__file__).parent
sys.path.insert(0, str(CODE_ROOT))

# Also add Code directory for SwinTransformerPhysics (swin_patched)
sys.path.insert(0, '/home/ydai17/Turbo/Code')

from Model.swin_dit_pde_refiner import SwinDiTPDERefiner, get_swin_dit_pde_refiner_config
from Model.swin_unet_pde_refiner import SwinUNetPDERefiner, get_swin_unet_pde_refiner_config
from Model.swin_unet_progressive import SwinUNetProgressive, get_swin_unet_progressive_config
from Dataset.jhu_dataset import SparseJHUDataset
from Dataset.sparse_dataset import SparseDNSDataset

# Import SwinTransformerPhysics from Code directory (for swin_patched model type)
from Model.swin_transformer_model_patched import SwinTransformerPhysics
from Model.configs.swin_patched_config_fixed import SwinPatchedConfigFixed

# Import SwinTransformerUNet for swin_unet_v2 model type (from train_swin_v2.py)
from Model.swin_transformer_unet import SwinTransformerUNet
from Model.configs.swin_jhu_config_v2 import get_jhu_swin_config_v2


class ARFrameLoader:
    """Helper class to load consecutive time windows for AR testing."""

    def __init__(self, data_dir, indicators, norm_stats, sparse_z_interval, dense_z_size=128, data_source='jhu'):
        self.data_dir = Path(data_dir)
        self.indicators = indicators
        self.norm_stats = norm_stats
        self.sparse_z_interval = sparse_z_interval
        self.dense_z_size = dense_z_size
        self.data_source = data_source
        self.sparse_z_indices = list(range(0, dense_z_size, sparse_z_interval))
        self.missing_z_indices = [i for i in range(dense_z_size) if i not in self.sparse_z_indices]

    def _load_frame(self, indicator, timestamp):
        """Load and normalize a single frame."""
        if self.data_source == 'jhu':
            # JHU format: {indicator}_{timestamp}.npy (e.g., u_5.020.npy)
            ts_float = float(timestamp)
            ts_formatted = f"{ts_float:.3f}"
            filename = f"{indicator}_{ts_formatted}.npy"
        else:
            # DNS format: img_{indicator}_dns{timestamp}.npy (e.g., img_u_dns0.npy)
            ts_int = int(timestamp)
            filename = f"img_{indicator}_dns{ts_int}.npy"

        filepath = self.data_dir / filename

        if not filepath.exists():
            return None

        data = np.load(filepath).astype(np.float32)

        # DNS data: truncate from 65 to 64 z-layers
        if self.data_source == 'dns' and data.shape[0] == 65:
            data = data[:64, :, :]

        if indicator in self.norm_stats:
            mean = self.norm_stats[indicator]['mean']
            std = self.norm_stats[indicator]['std']
            data = (data - mean) / (std + 1e-8)

        return data

    def load_window(self, start_timestamp, length, device):
        """
        Load a consecutive time window starting from start_timestamp.

        Args:
            start_timestamp: Starting timestamp (float for JHU, int for DNS)
            length: Number of timesteps to load
            device: torch device

        Returns:
            dense_data: (1, length, C, Z, H, W) tensor
            sparse_data: (1, length, C, n_sparse, H, W) tensor
            success: bool indicating if all frames were loaded
        """
        if self.data_source == 'jhu':
            start_ts = float(start_timestamp)
            # JHU: 0.01 timestep interval
            timestamps = [start_ts + i * 0.01 for i in range(length)]
        else:
            start_ts = int(start_timestamp)
            # DNS: integer timesteps
            timestamps = [start_ts + i for i in range(length)]

        # Load all frames
        all_data = []
        for indicator in self.indicators:
            indicator_frames = []
            for ts in timestamps:
                frame = self._load_frame(indicator, ts)
                if frame is None:
                    return None, None, False
                indicator_frames.append(frame)
            all_data.append(np.stack(indicator_frames, axis=0))  # (T, Z, H, W)

        # Stack to (T, C, Z, H, W)
        dense_data = np.stack(all_data, axis=1)

        # Extract sparse data
        sparse_data = dense_data[:, :, self.sparse_z_indices, :, :]

        # Convert to tensors and add batch dimension
        dense_tensor = torch.from_numpy(dense_data).unsqueeze(0).to(device)
        sparse_tensor = torch.from_numpy(sparse_data).unsqueeze(0).to(device)

        return dense_tensor, sparse_tensor, True


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return super().default(obj)


def load_normalization_stats(data_source='jhu'):
    """Load normalization stats."""
    if data_source == 'jhu':
        stats_path = CODE_ROOT / 'Data' / 'jhu_normalization_stats.json'
    else:
        stats_path = CODE_ROOT / 'Data' / 'normalization_stats.json'

    if stats_path.exists():
        with open(stats_path, 'r') as f:
            raw_stats = json.load(f)
        stats = {}
        channels = raw_stats.get('indicators', ['u', 'v', 'w', 'p'])
        for i, ch in enumerate(channels):
            stats[ch] = {'mean': raw_stats['mean'][i], 'std': raw_stats['std'][i]}
        return stats
    else:
        print(f"WARNING: Normalization stats not found at {stats_path}")
        return {
            'u': {'mean': 0.0, 'std': 1.0},
            'v': {'mean': 0.0, 'std': 1.0},
            'w': {'mean': 0.0, 'std': 1.0},
            'p': {'mean': 0.0, 'std': 1.0}
        }


def denormalize(data, channel_name, stats):
    """Denormalize data."""
    if channel_name in stats:
        mean = stats[channel_name]['mean']
        std = stats[channel_name]['std']
        return data * std + mean
    return data


def compute_metrics(pred, target, stats, channels=['u', 'v', 'w', 'p']):
    """Compute evaluation metrics in both normalized and physical space."""
    # Normalized space metrics
    mse_norm = F.mse_loss(pred, target).item()
    rmse_norm = np.sqrt(mse_norm)
    mae_norm = F.l1_loss(pred, target).item()

    # Physical space metrics (denormalized)
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()

    mse_phys_list = []
    rmse_per_channel = {}

    for c_idx, ch in enumerate(channels):
        pred_ch = pred_np[..., c_idx, :, :, :]
        target_ch = target_np[..., c_idx, :, :, :]

        # Denormalize
        pred_ch_phys = pred_ch * stats[ch]['std'] + stats[ch]['mean']
        target_ch_phys = target_ch * stats[ch]['std'] + stats[ch]['mean']

        mse_ch = np.mean((pred_ch_phys - target_ch_phys) ** 2)
        mse_phys_list.append(mse_ch)
        rmse_per_channel[f'rmse_{ch}_physical'] = np.sqrt(mse_ch)

    mse_phys = np.mean(mse_phys_list)
    rmse_phys = np.sqrt(mse_phys)

    # MAE in physical space
    pred_phys = pred_np.copy()
    target_phys = target_np.copy()
    for c_idx, ch in enumerate(channels):
        pred_phys[..., c_idx, :, :, :] = pred_np[..., c_idx, :, :, :] * stats[ch]['std'] + stats[ch]['mean']
        target_phys[..., c_idx, :, :, :] = target_np[..., c_idx, :, :, :] * stats[ch]['std'] + stats[ch]['mean']
    mae_phys = np.mean(np.abs(pred_phys - target_phys))

    metrics = {
        'mse_normalized': mse_norm,
        'rmse_normalized': rmse_norm,
        'mae_normalized': mae_norm,
        'mse_physical': mse_phys,
        'rmse_physical': rmse_phys,
        'mae_physical': mae_phys,
        **rmse_per_channel
    }

    return metrics


def load_model(checkpoint_path, config_name, device, model_type='swin_dit', data_source='jhu'):
    """Load Swin-DiT, Swin-UNet, Swin-UNet Progressive, Swin-UNet V2, or Swin Patched model from checkpoint."""
    if model_type == 'swin_unet_v2':
        # SwinTransformerUNet from train_swin_v2.py (UNet without PDE-Refiner)
        config = get_jhu_swin_config_v2(config_name)
        # Adjust config for data source
        if data_source == 'dns':
            config.INPUT_RESOLUTION = (64, 128, 128)
            config.PATCHED_RESOLUTION = (32, 32, 32)
            config.PATCH_SIZE = (2, 4, 4)
            config.NUM_SPARSE_Z_LAYERS = 12
        model = SwinTransformerUNet(config)

        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()

        epoch = checkpoint.get('epoch', 'unknown')
        best_val_loss = checkpoint.get('best_val_loss', 'unknown')
        print(f"Loaded SwinTransformerUNet (V2) from {checkpoint_path}")
        print(f"  Epoch: {epoch}, Best Val Loss: {best_val_loss}")
        print(f"  Config: {config_name}")

        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params:,}")

        return model, config, total_params

    elif model_type == 'swin_patched':
        # Swin Patched supports both DNS and JHU configs
        if data_source == 'jhu':
            # Use JHU config (supports spectral, spectral_progressive, pde_refiner_large, etc.)
            from Model.configs.swin_jhu_config import get_jhu_swin_config
            try:
                config = get_jhu_swin_config(config_name)
                print(f"Using JHU config: {config_name}")
            except ValueError:
                # Fall back to default
                config = SwinPatchedConfigFixed()
                print(f"Config '{config_name}' not found, using default SwinPatchedConfigFixed")
        else:
            # Use DNS config
            config = SwinPatchedConfigFixed()
            print("Using default SwinPatchedConfigFixed for DNS")

        model = SwinTransformerPhysics(config)

        # Load checkpoint (swin_patched uses 'model_state_dict' key)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()

        epoch = checkpoint.get('epoch', 'unknown')
        best_val_loss = checkpoint.get('best_val_loss', 'unknown')
        print(f"Loaded Swin Patched model from {checkpoint_path}")
        print(f"  Epoch: {epoch}, Best Val Loss: {best_val_loss}")

        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params:,}")

        return model, config, total_params

    elif model_type == 'swin_unet_progressive':
        config = get_swin_unet_progressive_config(config_name, data_source=data_source)
        model = SwinUNetProgressive(
            in_channels=len(config.INDICATORS),
            t_in=config.INPUT_TIMESTEPS,
            t_out=config.OUTPUT_TIMESTEPS,
            total_z=config.TOTAL_Z_LAYERS,
            sparse_z=config.SPARSE_Z_LAYERS,
            embed_dim=config.EMBED_DIM,
            depths=config.DEPTHS,
            num_heads=config.NUM_HEADS,
            window_sizes=config.WINDOW_SIZES,
            patch_size=config.PATCH_SIZE,
            mlp_ratio=config.MLP_RATIO,
            num_refine_steps=config.REFINER_STEPS,
            min_noise_std=config.MIN_NOISE_STD,
            drop_rate=config.DROP_RATE,
            attn_drop_rate=config.ATTN_DROP_RATE,
        )
    elif model_type == 'swin_unet':
        config = get_swin_unet_pde_refiner_config(config_name, data_source=data_source)
        model = SwinUNetPDERefiner(
            in_channels=len(config.INDICATORS),
            t_in=config.INPUT_TIMESTEPS,
            t_out=config.OUTPUT_TIMESTEPS,
            total_z=config.TOTAL_Z_LAYERS,
            sparse_z=config.SPARSE_Z_LAYERS,
            embed_dim=config.EMBED_DIM,
            depths=config.DEPTHS,
            num_heads=config.NUM_HEADS,
            window_size=config.WINDOW_SIZE,
            patch_size=config.PATCH_SIZE,
            mlp_ratio=config.MLP_RATIO,
            num_refine_steps=config.REFINER_STEPS,
            min_noise_std=config.MIN_NOISE_STD,
            drop_rate=config.DROP_RATE,
            attn_drop_rate=config.ATTN_DROP_RATE,
        )
    else:
        # swin_dit uses config object
        config = get_swin_dit_pde_refiner_config(config_name, data_source=data_source)
        model = SwinDiTPDERefiner(config)

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    epoch = checkpoint.get('epoch', 'unknown')
    best_val_loss = checkpoint.get('best_val_loss', 'unknown')
    print(f"Loaded model from {checkpoint_path}")
    print(f"  Epoch: {epoch}, Best Val Loss: {best_val_loss}")
    print(f"  Config: {config_name}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    return model, config, total_params


def create_dataloader(data_dir, split_config_path, split='test', batch_size=1, sparse_z_interval=8, data_source='jhu'):
    """Create data loader for JHU or DNS data."""
    norm_stats = load_normalization_stats(data_source)

    if data_source == 'jhu':
        dataset = SparseJHUDataset(
            data_dir=data_dir,
            split=split,
            split_config_path=str(split_config_path),
            indicators=['u', 'v', 'w', 'p'],
            sparse_z_interval=sparse_z_interval,
            normalize=True,
            norm_stats=norm_stats,
            dense_z_size=128
        )
        dense_z_size = 128
    else:
        dataset = SparseDNSDataset(
            data_dir=data_dir,
            split=split,
            split_config_path=str(split_config_path),
            indicators=['u', 'v', 'w', 'p'],
            sparse_z_interval=sparse_z_interval,
            normalize=True,
            norm_stats=norm_stats
        )
        dense_z_size = 64

    # JHU data is very large (~320MB/sample), use num_workers=0 to avoid memory issues
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    return loader, norm_stats, dense_z_size


@torch.no_grad()
def run_autoregressive_test(model, loader, stats, device, num_rounds=3, channels=['u', 'v', 'w', 'p'],
                            frame_loader=None, t_out=5, model_type='swin_dit', data_source='jhu'):
    """
    Run multi-round autoregressive test with proper consecutive time windows.

    For num_rounds=3:
    - Round 1: t0-t4 (GT input) + sparse t5-t9 -> predict t5-t9 (compare with dense t5-t9)
    - Round 2: t5-t9 (predicted) + sparse t10-t14 -> predict t10-t14 (compare with dense t10-t14)
    - Round 3: t10-t14 (predicted) + sparse t15-t19 -> predict t15-t19 (compare with dense t15-t19)

    Supports both PDE-Refiner models (swin_dit, swin_unet, swin_unet_progressive) and
    Swin Patched model (swin_patched) which has a different forward interface.
    """
    round_metrics = defaultdict(list)
    skipped_samples = 0

    # Determine timestep delta based on data source
    ts_delta = 0.01 if data_source == 'jhu' else 1  # JHU: 0.01, DNS: 1 (integer)

    for batch_idx, batch in enumerate(tqdm(loader, desc="AR Test")):
        # Get initial data
        input_dense = batch['input_dense'].to(device)  # (B, T_in, C, Z, H, W)
        output_sparse = batch['output_sparse'].to(device)  # (B, T_out, C, n_sparse, H, W)
        output_dense = batch['output_dense'].to(device)  # (B, T_out, C, Z, H, W)
        sparse_z_indices = batch['sparse_z_indices'].to(device)  # (B, n_sparse)
        global_timesteps = batch['global_timesteps']  # (B, T_in + T_out)

        B, T_in, C, Z, H, W = input_dense.shape

        # Compute missing z indices
        all_z = set(range(Z))
        sparse_set = set(sparse_z_indices[0].cpu().numpy().tolist())
        missing_z = sorted(list(all_z - sparse_set))
        missing_z_indices = torch.tensor(missing_z, device=device).unsqueeze(0).expand(B, -1)

        # Get sparse z indices as a list (for swin_patched)
        sparse_z_list = sparse_z_indices[0].cpu().numpy().tolist()

        # Get initial timestamps
        initial_timesteps = global_timesteps[0].numpy()  # (T_in + T_out,)

        # Current input for autoregressive prediction
        current_input = input_dense.clone()

        # Track the current output window start timestamp
        # After round k, the next window starts at: initial_output_start + k * t_out * ts_delta
        initial_output_start = initial_timesteps[T_in]  # First output timestamp

        # Variables to track current sparse/dense targets
        current_output_sparse = output_sparse
        current_output_dense = output_dense

        sample_valid = True

        for round_idx in range(num_rounds):
            if not sample_valid:
                break

            # Forward pass - different interface for different model types
            if model_type in ['swin_patched', 'swin_unet_v2']:
                # Swin Patched and SwinTransformerUNet (V2) use batch dict interface
                # Note: expects 'output_sparse_z' key (from training)
                # Both expect tensor sparse_indices
                model_batch = {
                    'input_dense': current_input,
                    'output_sparse_z': current_output_sparse,
                    'sparse_indices': {'z': sparse_z_indices[0]}
                }
                pred_dict = model(model_batch)
                pred = pred_dict['output_full']
            else:
                # PDE-Refiner models (swin_dit, swin_unet, swin_unet_progressive)
                pred = model(
                    current_input,
                    current_output_sparse,
                    sparse_z_indices,
                    missing_z_indices
                )

            # Compute metrics for this round
            metrics = compute_metrics(pred, current_output_dense, stats, channels)
            round_metrics[round_idx + 1].append(metrics)

            # Prepare next round input (if not last round)
            if round_idx < num_rounds - 1:
                # Use prediction as next input
                current_input = pred.clone()

                # Load the next time window's sparse and dense targets
                if frame_loader is not None:
                    # Calculate next window start timestamp
                    # Round k+1 target window starts at: initial_output_start + (k+1) * t_out * ts_delta
                    next_window_start = initial_output_start + (round_idx + 1) * t_out * ts_delta

                    # Load next window
                    next_dense, next_sparse, success = frame_loader.load_window(
                        next_window_start, t_out, device
                    )

                    if success:
                        current_output_dense = next_dense
                        current_output_sparse = next_sparse
                    else:
                        # Cannot load next window, skip remaining rounds for this sample
                        sample_valid = False
                        skipped_samples += 1
                else:
                    # No frame loader, cannot do proper AR test beyond round 1
                    sample_valid = False
                    if batch_idx == 0 and round_idx == 0:
                        print("WARNING: No frame_loader provided. Only round 1 will have valid metrics.")

    if skipped_samples > 0:
        print(f"Note: {skipped_samples} samples skipped in later rounds due to missing data files.")

    return round_metrics


def aggregate_metrics(round_metrics):
    """Aggregate metrics across samples."""
    results = {}

    for round_idx, metrics_list in round_metrics.items():
        if not metrics_list:
            continue

        # Aggregate each metric
        aggregated = {}
        metric_keys = metrics_list[0].keys()

        for key in metric_keys:
            values = [m[key] for m in metrics_list]
            aggregated[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }

        aggregated['num_samples'] = len(metrics_list)
        results[f'round_{round_idx}'] = aggregated

    return results


def main():
    parser = argparse.ArgumentParser(description='AR Test for Swin-DiT/UNet PDE Refiner')

    # Model
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='giant',
                        choices=['mid', 'giant',
                                 'spectral_progressive_large', 'spectral_progressive_giant'],
                        help='Model configuration: giant for swin_dit, spectral_progressive_large/giant for swin_patched')
    parser.add_argument('--model_type', type=str, default='swin_dit',
                        choices=['swin_dit', 'swin_unet', 'swin_unet_progressive', 'swin_unet_v2', 'swin_patched'],
                        help='Model architecture type')

    # Data
    parser.add_argument('--data_source', type=str, default='jhu',
                        choices=['jhu', 'dns'], help='Data source: jhu (128^3) or dns (64x128x128)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Data directory (auto-detected if not specified)')
    parser.add_argument('--split_config', type=str, default=None,
                        help='Split configuration file')
    parser.add_argument('--sparse_z_interval', type=int, default=8,
                        help='Sparse z interval')

    # Test settings
    parser.add_argument('--num_rounds', type=int, default=3,
                        help='Number of AR rounds')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results')

    args = parser.parse_args()

    # Set default paths based on data source
    if args.data_dir is None:
        if args.data_source == 'jhu':
            args.data_dir = '/media/ydai17/T7 Shield/JHU_data/DNS128'
        else:  # dns
            args.data_dir = '/home/ydai17/Turbo/DNS_data/DNS_data'

    if args.split_config is None:
        if args.data_source == 'jhu':
            args.split_config = str(CODE_ROOT / 'Dataset' / 'jhu_data_splits_ar.json')
        else:  # dns
            args.split_config = str(CODE_ROOT / 'Data' / 'split_config_ar.json')

    if args.output_dir is None:
        checkpoint_dir = Path(args.checkpoint).parent
        args.output_dir = str(checkpoint_dir / f'test_ar_{args.num_rounds}rounds')

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model (type: {args.model_type}, data_source: {args.data_source})...")
    model, config, total_params = load_model(
        args.checkpoint, args.config, device, args.model_type, data_source=args.data_source
    )

    # Create dataloader
    print("\nCreating dataloader...")
    loader, norm_stats, dense_z_size = create_dataloader(
        args.data_dir, args.split_config,
        split='test', batch_size=1,
        sparse_z_interval=args.sparse_z_interval,
        data_source=args.data_source
    )
    print(f"Test samples: {len(loader)}")
    print(f"Data source: {args.data_source}, Dense Z size: {dense_z_size}")

    # Create frame loader for loading consecutive windows in AR testing
    frame_loader = ARFrameLoader(
        data_dir=args.data_dir,
        indicators=['u', 'v', 'w', 'p'],
        norm_stats=norm_stats,
        sparse_z_interval=args.sparse_z_interval,
        dense_z_size=dense_z_size,
        data_source=args.data_source
    )

    # Run AR test
    print(f"\nRunning {args.num_rounds}-round autoregressive test...")
    round_metrics = run_autoregressive_test(
        model, loader, norm_stats, device,
        num_rounds=args.num_rounds,
        channels=['u', 'v', 'w', 'p'],
        frame_loader=frame_loader,
        t_out=5,  # Output timesteps per round
        model_type=args.model_type,
        data_source=args.data_source
    )

    # Aggregate results
    results = aggregate_metrics(round_metrics)

    # Add metadata
    model_name = args.model_type if args.model_type in ['swin_patched', 'swin_unet_v2'] else f'{args.model_type}_pde_refiner'
    results['metadata'] = {
        'model': model_name,
        'checkpoint': str(args.checkpoint),
        'config': args.config,
        'data_source': args.data_source,
        'num_rounds': args.num_rounds,
        'total_params': total_params
    }

    # Print results
    print("\n" + "=" * 60)
    print("AUTOREGRESSIVE TEST RESULTS")
    print("=" * 60)

    for round_idx in range(1, args.num_rounds + 1):
        key = f'round_{round_idx}'
        if key in results:
            r = results[key]
            print(f"\nRound {round_idx}:")
            print(f"  RMSE (normalized): {r['rmse_normalized']['mean']:.6f} ± {r['rmse_normalized']['std']:.6f}")
            print(f"  RMSE (physical):   {r['rmse_physical']['mean']:.6f} ± {r['rmse_physical']['std']:.6f}")
            print(f"  MSE (normalized):  {r['mse_normalized']['mean']:.6f} ± {r['mse_normalized']['std']:.6f}")
            print(f"  Samples: {r['num_samples']}")

    # Save results
    results_path = output_dir / 'ar_evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to: {results_path}")

    # Save summary
    summary_path = output_dir / 'summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f"Swin-DiT PDE Refiner AR Test Summary\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Config: {args.config}\n")
        f.write(f"Parameters: {total_params:,}\n")
        f.write(f"Rounds: {args.num_rounds}\n\n")

        for round_idx in range(1, args.num_rounds + 1):
            key = f'round_{round_idx}'
            if key in results:
                r = results[key]
                f.write(f"Round {round_idx}:\n")
                f.write(f"  RMSE (phys): {r['rmse_physical']['mean']:.6f}\n")
                f.write(f"  RMSE (norm): {r['rmse_normalized']['mean']:.6f}\n")

    print(f"Summary saved to: {summary_path}")

    return results


if __name__ == '__main__':
    main()
