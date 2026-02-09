#!/usr/bin/env python3
"""
Unified evaluation script for ALL baseline models on DNS and JHU datasets.

Generates LaTeX table entries for paper.

Usage:
    python evaluate_all_baselines.py --models deeponet,fno --datasets DNS,JHU
    python evaluate_all_baselines.py --all  # Evaluate all trained models
    python evaluate_all_baselines.py --checkpoint-dir baselines_unified  # Use specific checkpoint dir
"""

import argparse
import sys
import os
from pathlib import Path

# Ensure _baseline_models is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _baseline_models import (
    DeepONet3D,
    PIDeepONet3D,
    FNO3D,
    UFNO3D,
    TFNO3D,
    UNet3D,
    Transolver3D,
    DPOT3D,
    Factformer3D,
    PINO3D
)

import torch
import numpy as np
import json
from datetime import datetime
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

# ==============================================================================
# Configuration
# ==============================================================================

TURBO_DATA = "/home/ydai17/Turbo/ICML_code_submission/TurboData"

MODEL_CLASSES = {
    'deeponet': DeepONet3D,
    'pi_deeponet': PIDeepONet3D,
    'fno': FNO3D,
    'ufno': UFNO3D,
    'tfno': TFNO3D,
    'unet': UNet3D,
    'transolver': Transolver3D,
    'dpot': DPOT3D,
    'factformer': Factformer3D,
    'pino': PINO3D
}

MODEL_DISPLAY_NAMES = {
    'deeponet': 'DeepONet',
    'pi_deeponet': 'PI-DeepONet',
    'fno': 'FNO',
    'ufno': 'U-FNO',
    'tfno': 'TFNO',
    'unet': 'U-Net',
    'transolver': 'Transolver',
    'dpot': 'DPOT',
    'factformer': 'Factformer',
    'pino': 'PINO'
}

DATASETS = {
    'DNS': {
        'data_dir': f"{TURBO_DATA}/DNS_data",
        'split_config': f"{TURBO_DATA}/dns_splits.json",
        'norm_stats': f"{TURBO_DATA}/dns_normalization_stats.json",
        'spatial_size': (64, 128, 128),
        'ar_rounds': 2,
        'file_format': 'dns',
        'ts_delta': 0.002,
        'crop_z': 64
    },
    'JHU': {
        'data_dir': f"{TURBO_DATA}/JHU_DNS128",
        'split_config': f"{TURBO_DATA}/jhu_splits.json",
        'norm_stats': f"{TURBO_DATA}/jhu_normalization_stats.json",
        'spatial_size': (128, 128, 128),
        'ar_rounds': 3,
        'file_format': 'jhu',
        'ts_delta': 0.01,
        'crop_z': None
    }
}


# ==============================================================================
# Data Loading
# ==============================================================================

def load_frame(data_dir, indicator, timestamp, norm_stats, file_format='jhu', crop_z=None):
    """Load and normalize a single frame."""
    data_dir = Path(data_dir)

    if file_format == 'dns':
        filename = f"img_{indicator}_dns{int(timestamp)}.npy"
    else:
        ts_float = float(timestamp)
        filename = f"{indicator}_{ts_float:.3f}.npy"

    filepath = data_dir / filename

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    data = np.load(filepath).astype(np.float32)

    # Crop Z if needed
    if crop_z is not None and data.shape[0] > crop_z:
        data = data[:crop_z, :, :]

    # Normalize
    mean = norm_stats[indicator]['mean']
    std = norm_stats[indicator]['std']
    data = (data - mean) / (std + 1e-8)

    return data


def denormalize(data, indicator, norm_stats):
    """Denormalize data."""
    mean = norm_stats[indicator]['mean']
    std = norm_stats[indicator]['std']
    return data * std + mean


# ==============================================================================
# Metrics
# ==============================================================================

def compute_ssim_per_channel(pred, target):
    """
    Compute SSIM for a single channel.
    pred, target: (Z, H, W) arrays
    Returns: mean SSIM across Z slices

    Uses the entire 3D volume's data_range for consistency across all slices.
    """
    # Use whole volume's data_range for consistency
    data_range = target.max() - target.min()
    if data_range < 1e-8:
        data_range = 1.0

    ssim_vals = []
    for z in range(pred.shape[0]):
        val = ssim(target[z], pred[z], data_range=data_range)
        ssim_vals.append(val)

    return np.mean(ssim_vals)


def compute_rmse(pred, target):
    """Compute RMSE."""
    return np.sqrt(np.mean((pred - target) ** 2))


# ==============================================================================
# Model Loading
# ==============================================================================

def load_model_generic(checkpoint_path, model_class, spatial_size, device):
    """Generic model loading that works for all model types."""

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})

    # Try to instantiate model with saved config
    try:
        # Only DeepONet models need spatial_size parameter
        if model_class in [DeepONet3D, PIDeepONet3D]:
            model = model_class(
                in_channels=4,
                out_channels=4,
                input_timesteps=5,
                output_timesteps=5,
                spatial_size=spatial_size,
                **config
            ).to(device)
        else:
            # For FNO, U-Net, Transolver, DPOT, Factformer, PINO, etc. - no spatial_size
            model = model_class(
                in_channels=4,
                out_channels=4,
                input_timesteps=5,
                output_timesteps=5,
                **config
            ).to(device)

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

    except Exception as e:
        print(f"    Warning: Could not load with saved config: {e}")
        print(f"    Trying default parameters...")

        # Fallback to default parameters
        if model_class in [DeepONet3D, PIDeepONet3D]:
            model = model_class(
                in_channels=4,
                out_channels=4,
                input_timesteps=5,
                output_timesteps=5,
                spatial_size=spatial_size,
                branch_dim=80,
                trunk_dim=384,
                basis_dim=160
            ).to(device)
        elif model_class == FNO3D:
            model = model_class(
                in_channels=4,
                out_channels=4,
                input_timesteps=5,
                output_timesteps=5,
                width=32,
                modes_z=12,
                modes_h=12,
                modes_w=12
            ).to(device)
        else:
            raise ValueError(f"Cannot create default model for {model_class}")

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model


# ==============================================================================
# Autoregressive Evaluation
# ==============================================================================

def run_ar_evaluation(model, dataset_config, device):
    """
    Run autoregressive evaluation.
    Returns per-channel metrics for each round.
    """
    # Load config
    with open(dataset_config['split_config'], 'r') as f:
        split_config = json.load(f)

    with open(dataset_config['norm_stats'], 'r') as f:
        stats = json.load(f)

    # Handle both norm stats formats
    if 'mean' in stats and isinstance(stats['mean'], list):
        norm_stats = {}
        for i, ind in enumerate(['u', 'v', 'w', 'p']):
            norm_stats[ind] = {'mean': stats['mean'][i], 'std': stats['std'][i]}
    else:
        norm_stats = stats

    # Get test sequences - prefer test_ar for autoregressive evaluation
    if 'splits' in split_config:
        # Prefer test_ar if available (designed for AR evaluation with enough timesteps)
        if 'test_ar' in split_config['splits']:
            test_sequences = split_config['splits']['test_ar']
            print(f"  Using test_ar sequences ({len(test_sequences)} sequences)")
        elif 'test' in split_config['splits']:
            test_sequences = split_config['splits']['test']
            print(f"  Using test sequences ({len(test_sequences)} sequences)")
        else:
            test_sequences = split_config.get('test_sequences', [])
    else:
        test_sequences = split_config.get('test_sequences', [])

    ar_rounds = dataset_config['ar_rounds']
    ts_delta = dataset_config['ts_delta']
    data_dir = dataset_config['data_dir']
    file_format = dataset_config['file_format']
    crop_z = dataset_config.get('crop_z', None)
    indicators = ['u', 'v', 'w', 'p']

    # Results storage: round -> channel -> [rmse_list, ssim_list]
    results = {}
    for r in range(1, ar_rounds + 1):
        results[f'round_{r}'] = {ch: {'rmse': [], 'ssim': []} for ch in indicators}

    print(f"  Evaluating {len(test_sequences)} test sequences...")

    for seq in tqdm(test_sequences, desc="  Test sequences"):
        try:
            # seq is a list of timestamps (10 for single-round, more for AR)
            # e.g., DNS: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            # e.g., JHU: ["5.02", "5.03", "5.04", ...]
            if not isinstance(seq, list):
                print(f"  Skipping invalid sequence format: {type(seq)}")
                continue

            all_timestamps = seq  # Direct list of timestamps

            # Need at least 5 + ar_rounds*5 timestamps
            required_ts = 5 + ar_rounds * 5
            if len(all_timestamps) < required_ts:
                # Try to extend by computing with ts_delta (for backwards compatibility)
                start_ts = float(all_timestamps[0])
                if file_format == 'dns':
                    all_timestamps = [int(start_ts + i) for i in range(required_ts)]
                else:
                    all_timestamps = [start_ts + i * ts_delta for i in range(required_ts)]

            # Load initial 5 frames for all channels
            input_frames = []
            for t in range(5):
                ts = all_timestamps[t]
                frame_data = []
                for ch in indicators:
                    data = load_frame(data_dir, ch, ts, norm_stats, file_format, crop_z)
                    frame_data.append(data)
                input_frames.append(np.stack(frame_data))  # (4, Z, H, W)

            current_input = np.stack(input_frames)  # (5, 4, Z, H, W)

            # Autoregressive rounds
            for round_idx in range(1, ar_rounds + 1):
                # Predict next 5 frames
                with torch.no_grad():
                    input_tensor = torch.from_numpy(current_input).float().unsqueeze(0).to(device)
                    pred_tensor = model(input_tensor)
                    pred_frames = pred_tensor[0].cpu().numpy()  # (5, 4, Z, H, W)

                # Load ground truth for this round
                gt_frames = []
                for t in range(5):
                    gt_idx = 5 * round_idx + t  # Index into all_timestamps
                    ts = all_timestamps[gt_idx]
                    frame_data = []
                    for ch in indicators:
                        data = load_frame(data_dir, ch, ts, norm_stats, file_format, crop_z)
                        frame_data.append(data)
                    gt_frames.append(np.stack(frame_data))

                gt_frames = np.stack(gt_frames)  # (5, 4, Z, H, W)

                # Compute metrics for each channel (averaged over 5 timesteps)
                for ch_idx, ch in enumerate(indicators):
                    rmse_vals = []
                    ssim_vals = []

                    for t in range(5):
                        pred_ch = pred_frames[t, ch_idx]  # (Z, H, W)
                        gt_ch = gt_frames[t, ch_idx]

                        # Denormalize
                        pred_denorm = denormalize(pred_ch, ch, norm_stats)
                        gt_denorm = denormalize(gt_ch, ch, norm_stats)

                        # Compute metrics
                        rmse_vals.append(compute_rmse(pred_denorm, gt_denorm))
                        ssim_vals.append(compute_ssim_per_channel(pred_denorm, gt_denorm))

                    # Store averaged metrics
                    results[f'round_{round_idx}'][ch]['rmse'].append(np.mean(rmse_vals))
                    results[f'round_{round_idx}'][ch]['ssim'].append(np.mean(ssim_vals))

                # Update input for next round
                current_input = pred_frames

        except FileNotFoundError as e:
            # Skip invalid test sequences (e.g., timestamps beyond available data)
            continue

    # Compute mean and std
    for round_key in results:
        for ch in indicators:
            rmse_list = results[round_key][ch]['rmse']
            ssim_list = results[round_key][ch]['ssim']

            results[round_key][ch]['rmse'] = {
                'mean': np.mean(rmse_list),
                'std': np.std(rmse_list)
            }
            results[round_key][ch]['ssim'] = {
                'mean': np.mean(ssim_list),
                'std': np.std(ssim_list)
            }

    return results


# ==============================================================================
# LaTeX Formatting
# ==============================================================================

def format_table_row_jhu(model_name, results):
    """Format results for JHU table (3 rounds)."""
    row_parts = [model_name]

    for round_idx in [1, 2, 3]:
        round_key = f'round_{round_idx}'
        for ch in ['u', 'v', 'w', 'p']:
            rmse = results[round_key][ch]['rmse']['mean']
            ssim_val = results[round_key][ch]['ssim']['mean']
            row_parts.append(f"{rmse:.3f}")
            row_parts.append(f"{ssim_val:.3f}")

    return " & ".join(row_parts) + " \\\\"


def format_table_row_dns(model_name, results):
    """Format results for DNS table (2 rounds)."""
    row_parts = [model_name]

    for round_idx in [1, 2]:
        round_key = f'round_{round_idx}'
        for ch in ['u', 'v', 'w', 'p']:
            rmse = results[round_key][ch]['rmse']['mean']
            ssim_val = results[round_key][ch]['ssim']['mean']
            row_parts.append(f"{rmse:.3f}")
            row_parts.append(f"{ssim_val:.3f}")

    return " & ".join(row_parts) + " \\\\"


# ==============================================================================
# Evaluation
# ==============================================================================

def evaluate_model(model_name, model_class, dataset_name, dataset_config, device, checkpoint_dir):
    """Evaluate a single model on a single dataset."""

    print(f"\nDataset: {dataset_name}")
    print(f"  Spatial size: {dataset_config['spatial_size']}")
    print(f"  AR rounds: {dataset_config['ar_rounds']}")

    # Check if checkpoint exists
    checkpoint_path = f"{TURBO_DATA}/ICML_checkpoints/{checkpoint_dir}/{model_name}_{dataset_name.lower()}/{model_name}_best.pt"

    if not os.path.exists(checkpoint_path):
        print(f"  ✗ Checkpoint not found: {checkpoint_path}\n")
        return None

    print(f"  Loading: {checkpoint_path}")

    # Load model
    model = load_model_generic(checkpoint_path, model_class, dataset_config['spatial_size'], device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}\n")

    # Run evaluation
    print(f"  Running AR evaluation...")
    results = run_ar_evaluation(model, dataset_config, device)

    # Print summary
    print(f"\n  Results summary:")
    for round_key in sorted(results.keys()):
        print(f"    {round_key}:")
        for ch in ['u', 'v', 'w', 'p']:
            rmse = results[round_key][ch]['rmse']['mean']
            ssim_val = results[round_key][ch]['ssim']['mean']
            print(f"      {ch}: RMSE={rmse:.4f}, SSIM={ssim_val:.4f}")
    print()

    return results


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate baseline models')
    parser.add_argument('--models', type=str, default='deeponet,pi_deeponet',
                       help='Comma-separated list of models to evaluate')
    parser.add_argument('--datasets', type=str, default='DNS,JHU',
                       help='Comma-separated list of datasets')
    parser.add_argument('--all', action='store_true',
                       help='Evaluate all trained models')
    parser.add_argument('--checkpoint-dir', type=str, default='baselines_unified',
                       help='Checkpoint directory name under ICML_checkpoints/')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    print(f"Checkpoint dir: {TURBO_DATA}/ICML_checkpoints/{args.checkpoint_dir}\n")

    # Parse model and dataset lists
    if args.all:
        # Find all trained models
        checkpoint_dir = f"{TURBO_DATA}/ICML_checkpoints/{args.checkpoint_dir}"
        if not os.path.exists(checkpoint_dir):
            print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
            return

        trained_dirs = [d for d in os.listdir(checkpoint_dir)
                       if os.path.isdir(os.path.join(checkpoint_dir, d))]

        models_to_eval = set()
        for d in trained_dirs:
            if '_' in d:
                model_name = d.rsplit('_', 1)[0]
                if model_name in MODEL_CLASSES:
                    models_to_eval.add(model_name)

        models_to_eval = sorted(list(models_to_eval))
        datasets_to_use = ['DNS', 'JHU']
    else:
        models_to_eval = [m.strip() for m in args.models.split(',')]
        datasets_to_use = [d.strip() for d in args.datasets.split(',')]

    print(f"Models to evaluate: {models_to_eval}")
    print(f"Datasets to use: {datasets_to_use}\n")

    # Evaluate all combinations
    all_results = {}

    for model_name in models_to_eval:
        if model_name not in MODEL_CLASSES:
            print(f"Warning: Unknown model '{model_name}', skipping...")
            continue

        print(f"{'='*80}")
        print(f"Evaluating: {MODEL_DISPLAY_NAMES.get(model_name, model_name)}")
        print(f"{'='*80}\n")

        for dataset_name in datasets_to_use:
            if dataset_name not in DATASETS:
                print(f"Warning: Unknown dataset '{dataset_name}', skipping...")
                continue

            results = evaluate_model(
                model_name,
                MODEL_CLASSES[model_name],
                dataset_name,
                DATASETS[dataset_name],
                device,
                args.checkpoint_dir
            )

            if results:
                key = f"{model_name}_{dataset_name}"
                all_results[key] = results

    # Generate LaTeX table entries
    print(f"\n{'='*80}")
    print("LATEX TABLE ENTRIES")
    print(f"{'='*80}\n")

    print("--- JHU Table (replace corresponding rows) ---\n")
    for model_name in models_to_eval:
        key = f"{model_name}_JHU"
        if key in all_results:
            display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
            row = format_table_row_jhu(display_name, all_results[key])
            print(row)
            print()

    print("\n--- DNS Table (replace corresponding rows) ---\n")
    for model_name in models_to_eval:
        key = f"{model_name}_DNS"
        if key in all_results:
            display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
            row = format_table_row_dns(display_name, all_results[key])
            print(row)
            print()

    # Save detailed results
    output_file = f"{TURBO_DATA}/ICML_checkpoints/{args.checkpoint_dir}/evaluation_results_all.json"

    # Convert numpy types to Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64, np.floating)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64, np.integer)):
            return int(obj)
        else:
            return obj

    all_results_serializable = convert_numpy(all_results)

    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': all_results_serializable
        }, f, indent=2)

    print(f"\n✓ Detailed results saved to: {output_file}\n")


if __name__ == '__main__':
    main()
