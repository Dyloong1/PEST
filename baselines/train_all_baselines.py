#!/usr/bin/env python3
"""
Unified training script for ALL baseline models on DNS and JHU datasets.

Supported models:
- DeepONet, PI-DeepONet
- FNO3D, UFNO3D, TFNO3D
- Transolver3D
- DPOT3D
- Factformer3D
- PINO3D

Usage:
    python train_all_baselines.py --models deeponet,fno --datasets DNS,JHU
    python train_all_baselines.py --all  # Train all models on all datasets
    python train_all_baselines.py --models fno --datasets DNS --epochs 100
"""

import argparse
import sys
import os

# Ensure _baseline_models is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _baseline_models import (
    DeepONet3D,
    PIDeepONet3D,
    FNO3D,
    UFNO3D,
    TFNO3D,
    Transolver3D,
    DPOT3D,
    Factformer3D,
    PINO3D,
    get_baseline_config,
    BASELINE_SIZE_CONFIGS,
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from datetime import datetime
from pathlib import Path

# ==============================================================================
# Configuration
# ==============================================================================

TURBO_DATA = "/home/ydai17/Turbo/ICML_code_submission/TurboData"

# Training hyperparameters
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 1
DEFAULT_LR = 0.001

# Model configurations are loaded from _baseline_models via get_baseline_config()
# Available sizes: small, default, large, xlarge
AVAILABLE_MODELS = [
    'fno', 'ufno', 'tfno', 'transolver', 'dpot',
    'factformer', 'pino', 'deeponet', 'pi_deeponet'
]

# Dataset configurations
DATASET_CONFIGS = {
    'DNS': {
        'data_dir': f"{TURBO_DATA}/DNS_data",
        'split_config': f"{TURBO_DATA}/dns_splits.json",
        'norm_stats': f"{TURBO_DATA}/dns_normalization_stats.json",
        'spatial_size': (64, 128, 128),
        'file_format': 'dns',
        'crop_z': 64,
        'ts_delta': 0.002,
        # Grid spacing for physics loss (normalized, assuming domain [0, 2π])
        # DNS: 64x128x128, domain 2π x 2π x 2π
        'grid_spacing': (2 * np.pi / 64, 2 * np.pi / 128, 2 * np.pi / 128)
    },
    'JHU': {
        'data_dir': f"{TURBO_DATA}/JHU_DNS128",
        'split_config': f"{TURBO_DATA}/jhu_splits.json",
        'norm_stats': f"{TURBO_DATA}/jhu_normalization_stats.json",
        'spatial_size': (128, 128, 128),
        'file_format': 'jhu',
        'crop_z': None,
        'ts_delta': 0.01,
        # Grid spacing for physics loss (normalized, assuming domain [0, 2π])
        # JHU: 128x128x128, domain 2π x 2π x 2π
        'grid_spacing': (2 * np.pi / 128, 2 * np.pi / 128, 2 * np.pi / 128)
    }
}

# Models that use physics loss
PHYSICS_MODELS = {'pi_deeponet', 'pino'}
DEFAULT_PHYSICS_WEIGHT = 0.01


# ==============================================================================
# Dataset
# ==============================================================================

class TurbulenceDataset(Dataset):
    """Unified dataset for DNS and JHU turbulence data."""

    def __init__(self, data_dir, split_config, split_type, norm_stats_path, file_format='dns', crop_z=None):
        self.data_dir = Path(data_dir)
        self.file_format = file_format
        self.crop_z = crop_z

        # Load split configuration
        with open(split_config, 'r') as f:
            splits = json.load(f)

        # Both DNS and JHU use 'splits' key with list of timestamp sequences
        if 'splits' in splits and split_type in splits['splits']:
            self.samples = splits['splits'][split_type]
        else:
            raise ValueError(f"Split '{split_type}' not found in {split_config}")

        # Load and parse normalization stats
        with open(norm_stats_path, 'r') as f:
            stats = json.load(f)

        # Convert array format to dict format for easier access
        # Both DNS and JHU use array format with 'mean' and 'std' lists
        channels = stats.get('channels', stats.get('indicators', ['u', 'v', 'w', 'p']))
        self.norm_stats = {}
        for i, ch in enumerate(channels):
            self.norm_stats[ch] = {
                'mean': stats['mean'][i],
                'std': stats['std'][i]
            }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Each sample is a list of 10 timestamps (5 input + 5 output)
        timestamps = self.samples[idx]

        if len(timestamps) < 10:
            raise ValueError(f"Sample {idx} has only {len(timestamps)} timesteps, need 10")

        # Load data
        input_frames = []
        target_frames = []

        for i, ts in enumerate(timestamps[:10]):
            frame = self._load_frame(ts)
            if i < 5:
                input_frames.append(frame)
            else:
                target_frames.append(frame)

        input_tensor = torch.stack(input_frames)  # (5, 4, Z, H, W)
        target_tensor = torch.stack(target_frames)  # (5, 4, Z, H, W)

        return input_tensor, target_tensor

    def _load_frame(self, timestamp):
        """Load and normalize a single frame."""
        channels = []

        for ch_name in ['u', 'v', 'w', 'p']:
            if self.file_format == 'dns':
                # DNS format: img_{channel}_dns{timestamp}.npy (NO zero padding)
                filepath = self.data_dir / f"img_{ch_name}_dns{timestamp}.npy"
            else:
                # JHU format: {channel}_{timestamp:.3f}.npy
                ts_float = float(timestamp)
                filepath = self.data_dir / f"{ch_name}_{ts_float:.3f}.npy"

            if not filepath.exists():
                raise FileNotFoundError(f"File not found: {filepath}")

            data = np.load(filepath).astype(np.float32)

            # Crop if needed
            if self.crop_z is not None and data.shape[0] > self.crop_z:
                data = data[:self.crop_z, :, :]

            # Normalize
            mean = self.norm_stats[ch_name]['mean']
            std = self.norm_stats[ch_name]['std']
            data = (data - mean) / (std + 1e-8)

            channels.append(torch.from_numpy(data))

        return torch.stack(channels)  # (4, Z, H, W)


# ==============================================================================
# Helper Functions
# ==============================================================================

def load_norm_stats(norm_stats_path):
    """Load normalization statistics and convert to dict format."""
    with open(norm_stats_path, 'r') as f:
        stats = json.load(f)

    # Both DNS and JHU use array format with 'mean' and 'std' lists
    # Convert to dict format: {'u': {'mean': x, 'std': y}, ...}
    channels = stats.get('channels', stats.get('indicators', ['u', 'v', 'w', 'p']))
    norm_stats = {}
    for i, ch in enumerate(channels):
        norm_stats[ch] = {
            'mean': stats['mean'][i],
            'std': stats['std'][i]
        }
    return norm_stats


def denormalize_tensor(pred, norm_stats, device):
    """
    Denormalize prediction tensor for physics loss computation.

    Args:
        pred: (B, T, C, Z, H, W) normalized prediction
        norm_stats: dict with 'u', 'v', 'w', 'p' keys containing 'mean' and 'std'
        device: torch device

    Returns:
        Denormalized tensor in physical units
    """
    B, T, C, Z, H, W = pred.shape
    pred_denorm = pred.clone()

    channels = ['u', 'v', 'w', 'p']
    for i, ch in enumerate(channels):
        mean = norm_stats[ch]['mean']
        std = norm_stats[ch]['std']
        pred_denorm[:, :, i] = pred[:, :, i] * std + mean

    return pred_denorm


# ==============================================================================
# Training
# ==============================================================================

def train_one_epoch(model, loader, optimizer, device, grad_clip=1.0,
                    use_physics_loss=False, physics_weight=0.01,
                    norm_stats=None, grid_spacing=None):
    """
    Train for one epoch.

    Args:
        model: The model to train
        loader: DataLoader
        optimizer: Optimizer
        device: Device
        grad_clip: Gradient clipping value
        use_physics_loss: Whether to use physics loss (for PI-DeepONet, PINO)
        physics_weight: Weight for physics loss term
        norm_stats: Normalization statistics for denormalization
        grid_spacing: (dz, dy, dx) grid spacing for finite difference

    Returns:
        If use_physics_loss: (total_loss, data_loss, physics_loss)
        Else: total_loss
    """
    model.train()
    total_loss = 0.0
    total_data_loss = 0.0
    total_physics_loss = 0.0
    criterion = nn.MSELoss()

    for input_seq, target_seq in loader:
        input_seq = input_seq.to(device)
        target_seq = target_seq.to(device)

        optimizer.zero_grad()
        pred = model(input_seq)
        data_loss = criterion(pred, target_seq)

        # Add physics loss for PI-DeepONet and PINO
        if use_physics_loss and hasattr(model, 'compute_physics_loss'):
            # Denormalize prediction before computing physics loss
            if norm_stats is not None:
                pred_denorm = denormalize_tensor(pred, norm_stats, device)
            else:
                pred_denorm = pred

            # Compute physics loss with correct grid spacing
            if grid_spacing is not None:
                physics_loss = model.compute_physics_loss(pred_denorm, grid_spacing)
            else:
                physics_loss = model.compute_physics_loss(pred_denorm)

            loss = data_loss + physics_weight * physics_loss
            total_physics_loss += physics_loss.item()
        else:
            loss = data_loss

        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_data_loss += data_loss.item()

    n_batches = len(loader)
    if use_physics_loss:
        return total_loss / n_batches, total_data_loss / n_batches, total_physics_loss / n_batches
    return total_loss / n_batches


def validate(model, loader, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for input_seq, target_seq in loader:
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)

            pred = model(input_seq)
            loss = criterion(pred, target_seq)
            total_loss += loss.item()

    return total_loss / len(loader)


def train_model(model_name, model_config, dataset_name, dataset_config,
                device, epochs, batch_size, lr, physics_weight=None):
    """Train a single model on a single dataset."""

    print(f"\n{'='*80}")
    print(f"TRAINING: {model_name.upper()} on {dataset_name}")
    print(f"{'='*80}\n")

    # Check if this model uses physics loss
    use_physics_loss = model_name in PHYSICS_MODELS
    if physics_weight is None:
        physics_weight = DEFAULT_PHYSICS_WEIGHT

    if use_physics_loss:
        print(f"Physics-Informed Model: Using physics loss with weight={physics_weight}")

    # Create output directory
    output_dir = f"{TURBO_DATA}/ICML_checkpoints/baselines_unified/{model_name}_{dataset_name.lower()}"
    os.makedirs(output_dir, exist_ok=True)

    # Create model
    model_class = model_config['class']
    model_params = model_config['params'].copy()
    model_params.pop('spatial_size', None)

    # Only DeepONet needs spatial_size parameter
    if model_name in ['deeponet', 'pi_deeponet']:
        model = model_class(
            in_channels=4,
            out_channels=4,
            input_timesteps=5,
            output_timesteps=5,
            spatial_size=dataset_config['spatial_size'],
            **model_params
        ).to(device)
    else:
        model = model_class(
            in_channels=4,
            out_channels=4,
            input_timesteps=5,
            output_timesteps=5,
            **model_params
        ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,} ({num_params/1e6:.2f}M)\n")

    # Create datasets
    train_dataset = TurbulenceDataset(
        dataset_config['data_dir'],
        dataset_config['split_config'],
        'train',
        dataset_config['norm_stats'],
        dataset_config['file_format'],
        dataset_config.get('crop_z', None)
    )

    val_dataset = TurbulenceDataset(
        dataset_config['data_dir'],
        dataset_config['split_config'],
        'val',
        dataset_config['norm_stats'],
        dataset_config['file_format'],
        dataset_config.get('crop_z', None)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}\n")

    # Load normalization stats for physics loss
    norm_stats = None
    grid_spacing = None
    if use_physics_loss:
        norm_stats = load_norm_stats(dataset_config['norm_stats'])
        grid_spacing = dataset_config.get('grid_spacing', None)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")

        # Train with physics loss if applicable
        train_result = train_one_epoch(
            model, train_loader, optimizer, device,
            grad_clip=1.0,
            use_physics_loss=use_physics_loss,
            physics_weight=physics_weight,
            norm_stats=norm_stats,
            grid_spacing=grid_spacing
        )

        if use_physics_loss:
            train_loss, data_loss, phys_loss = train_result
            print(f"  Train: {train_loss:.6f} (data: {data_loss:.6f}, phys: {phys_loss:.6f})")
        else:
            train_loss = train_result
            print(f"  Train: {train_loss:.6f}", end="")

        val_loss = validate(model, val_loader, device)
        if not use_physics_loss:
            print(f", Val: {val_loss:.6f}")
        else:
            print(f"  Val: {val_loss:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(output_dir, f"{model_name}_best.pt")
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': model_params,
                'use_physics_loss': use_physics_loss,
                'physics_weight': physics_weight if use_physics_loss else None
            }
            torch.save(save_dict, checkpoint_path)
            print(f"  ✓ Saved (best: {val_loss:.6f})")

    print(f"\nCompleted! Best val loss: {best_val_loss:.6f}\n")
    return best_val_loss


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train baseline models')
    parser.add_argument('--models', type=str, default='deeponet',
                       help='Comma-separated list of models to train')
    parser.add_argument('--datasets', type=str, default='DNS,JHU',
                       help='Comma-separated list of datasets')
    parser.add_argument('--all', action='store_true',
                       help='Train all available models on all datasets')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=DEFAULT_LR,
                       help='Learning rate')
    parser.add_argument('--physics-weight', type=float, default=DEFAULT_PHYSICS_WEIGHT,
                       help='Weight for physics loss (for PI-DeepONet, PINO)')
    parser.add_argument('--size', type=str, default='default',
                       choices=['small', 'default', 'large', 'xlarge'],
                       help='Model size variant')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Training configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Physics weight: {args.physics_weight} (for PI-DeepONet, PINO)\n")

    # Parse model and dataset lists
    if args.all:
        models_to_train = list(AVAILABLE_MODELS)
        datasets_to_use = list(DATASET_CONFIGS.keys())
    else:
        models_to_train = [m.strip() for m in args.models.split(',')]
        datasets_to_use = [d.strip() for d in args.datasets.split(',')]

    print(f"Models to train: {models_to_train}")
    print(f"Datasets to use: {datasets_to_use}")
    print(f"Model size: {args.size}\n")

    # Validate
    for model_name in models_to_train:
        if model_name not in AVAILABLE_MODELS:
            print(f"Error: Unknown model '{model_name}'")
            print(f"Available models: {AVAILABLE_MODELS}")
            return

    for dataset_name in datasets_to_use:
        if dataset_name not in DATASET_CONFIGS:
            print(f"Error: Unknown dataset '{dataset_name}'")
            print(f"Available datasets: {list(DATASET_CONFIGS.keys())}")
            return

    # Train all combinations
    results = {}
    total = len(models_to_train) * len(datasets_to_use)
    current = 0

    for model_name in models_to_train:
        model_config = get_baseline_config(model_name, size=args.size)
        for dataset_name in datasets_to_use:
            current += 1
            print(f"\n{'='*80}")
            print(f"PROGRESS: {current} / {total}")
            print(f"{'='*80}")

            try:
                val_loss = train_model(
                    model_name,
                    model_config,
                    dataset_name,
                    DATASET_CONFIGS[dataset_name],
                    device,
                    args.epochs,
                    args.batch_size,
                    args.lr,
                    physics_weight=args.physics_weight
                )

                key = f"{model_name}_{dataset_name}"
                results[key] = {'success': True, 'val_loss': float(val_loss)}
                print(f"✓ {key}: val_loss = {val_loss:.6f}")

            except Exception as e:
                print(f"✗ Error training {model_name} on {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
                results[f"{model_name}_{dataset_name}"] = {'success': False, 'error': str(e)}

    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'epochs': args.epochs,
            'lr': args.lr,
            'batch_size': args.batch_size
        },
        'results': results
    }

    summary_path = f"{TURBO_DATA}/ICML_checkpoints/baselines_unified/training_summary.json"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*80}")
    print("TRAINING SUMMARY")
    print(f"{'='*80}")
    for key, result in results.items():
        if result['success']:
            print(f"  ✓ {key}: {result['val_loss']:.6f}")
        else:
            print(f"  ✗ {key}: {result['error']}")
    print(f"\n✓ Summary saved to: {summary_path}\n")


if __name__ == '__main__':
    main()
