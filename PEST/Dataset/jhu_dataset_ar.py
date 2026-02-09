"""
JHU DNS128 Autoregressive Dataset

Provides extended time windows for multi-round autoregressive testing.

For N rounds of AR testing (5-frame input, 5-frame output per round):
- Total timesteps needed: 5 + N*5 = 5*(N+1)
- Round 1: Input t0-t4 (GT), Predict t5-t9, Compare with GT t5-t9
- Round 2: Input t5-t9 (pred), Predict t10-t14, Compare with GT t10-t14
- Round 3: Input t10-t14 (pred), Predict t15-t19, Compare with GT t15-t19
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json


class SparseJHUDatasetAR(Dataset):
    """
    JHU DNS128 dataset for multi-round autoregressive testing.

    Provides extended time windows: 5 + num_ar_rounds * 5 consecutive timesteps.
    """

    def __init__(self,
                 data_dir,
                 split='test',
                 split_config_path=None,
                 indicators=['u', 'v', 'w', 'p'],
                 sparse_z_interval=8,
                 normalize=True,
                 norm_stats=None,
                 dense_z_size=128,
                 num_ar_rounds=3):
        """
        Args:
            data_dir: JHU data directory
            split: 'train', 'val', 'test'
            split_config_path: Split config JSON path (AR version)
            indicators: Physical quantities to use
            sparse_z_interval: Z-axis sparse sampling interval
            normalize: Whether to normalize
            norm_stats: Normalization statistics dict
            dense_z_size: Total Z layers (128 for JHU)
            num_ar_rounds: Number of AR rounds (determines window size)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.indicators = indicators
        self.dense_z_size = dense_z_size
        self.sparse_z_interval = sparse_z_interval
        self.normalize = normalize
        self.num_ar_rounds = num_ar_rounds

        # For AR testing: need 5 input + num_rounds * 5 output timesteps
        self.input_length = 5
        self.output_length = 5
        self.total_length = self.input_length + self.num_ar_rounds * self.output_length

        # Load split config
        if split_config_path is None:
            raise ValueError("Must provide split_config_path")

        with open(split_config_path, 'r') as f:
            config = json.load(f)

        self.sample_indices = config['splits'][split]
        self.metadata = config.get('metadata', {})

        # Calculate sparse and missing z indices
        self.sparse_z_indices = list(range(0, self.dense_z_size, sparse_z_interval))
        self.missing_z_indices = [i for i in range(self.dense_z_size)
                                  if i not in self.sparse_z_indices]

        print(f"\n{split.upper()} JHU AR Dataset:")
        print(f"  Data dir: {data_dir}")
        print(f"  Samples: {len(self.sample_indices)}")
        print(f"  AR Rounds: {num_ar_rounds}")
        print(f"  Timesteps per sample: {self.total_length} (5 input + {num_ar_rounds}*5 output)")
        print(f"  Sparse z layers: {len(self.sparse_z_indices)} / {self.dense_z_size}")

        # Normalization stats
        if normalize:
            if norm_stats is None:
                self.norm_stats = {
                    'u': {'mean': 0.0, 'std': 1.0},
                    'v': {'mean': 0.0, 'std': 1.0},
                    'w': {'mean': 0.0, 'std': 1.0},
                    'p': {'mean': 0.0, 'std': 1.0}
                }
            else:
                self.norm_stats = norm_stats
        else:
            self.norm_stats = None

    def __len__(self):
        return len(self.sample_indices)

    def _load_frame(self, indicator, timestamp):
        """Load single frame data."""
        ts_float = float(timestamp)
        ts_formatted = f"{ts_float:.3f}"
        filename = f"{indicator}_{ts_formatted}.npy"
        filepath = self.data_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        data = np.load(filepath).astype(np.float32)

        if self.normalize and self.norm_stats and indicator in self.norm_stats:
            mean = self.norm_stats[indicator]['mean']
            std = self.norm_stats[indicator]['std']
            data = (data - mean) / (std + 1e-8)

        return data

    def __getitem__(self, idx):
        """
        Returns data for multi-round AR testing.

        Returns:
            dict: {
                'timestamps': list of all timestamps
                'round_data': list of dicts for each round, each containing:
                    - 'input_dense': (5, C, Z, H, W) - input frames
                    - 'output_sparse': (5, C, n_sparse, H, W) - sparse GT
                    - 'output_dense': (5, C, Z, H, W) - dense GT
                'sparse_z_indices': tensor of sparse z indices
                'missing_z_indices': tensor of missing z indices
            }
        """
        sample_timestamps = self.sample_indices[idx]

        # Verify we have enough timesteps
        if len(sample_timestamps) < self.total_length:
            raise ValueError(
                f"Sample {idx} has only {len(sample_timestamps)} timesteps, "
                f"but {self.total_length} required for {self.num_ar_rounds} AR rounds"
            )

        # Load all frames at once
        all_data = []  # (total_length, C, Z, H, W)
        for indicator in self.indicators:
            indicator_frames = []
            for t in sample_timestamps[:self.total_length]:
                frame = self._load_frame(indicator, t)
                indicator_frames.append(frame)
            all_data.append(np.stack(indicator_frames, axis=0))  # (T, Z, H, W)

        all_data = np.stack(all_data, axis=1)  # (T, C, Z, H, W)

        # Build round data
        round_data = []
        for r in range(self.num_ar_rounds):
            # Round r uses:
            # - Input: frames [r*5 : r*5+5] (for r=0: frames 0-4, for r=1: frames 5-9, etc)
            # - Output: frames [(r+1)*5 : (r+2)*5]
            input_start = r * self.output_length
            input_end = input_start + self.input_length
            output_start = input_end
            output_end = output_start + self.output_length

            input_dense = all_data[input_start:input_end]  # (5, C, Z, H, W)
            output_dense = all_data[output_start:output_end]  # (5, C, Z, H, W)
            output_sparse = output_dense[:, :, self.sparse_z_indices, :, :]  # (5, C, n_sparse, H, W)

            round_data.append({
                'input_dense': torch.from_numpy(input_dense.copy()),
                'output_sparse': torch.from_numpy(output_sparse.copy()),
                'output_dense': torch.from_numpy(output_dense.copy()),
            })

        return {
            'timestamps': sample_timestamps[:self.total_length],
            'round_data': round_data,
            'sparse_z_indices': torch.tensor(self.sparse_z_indices),
            'missing_z_indices': torch.tensor(self.missing_z_indices),
            'sample_idx': idx,
        }


def create_ar_split_config(original_config_path, output_path, num_ar_rounds=3):
    """
    Create AR-specific split config with extended time windows.

    For N rounds of AR, need 5 + N*5 consecutive timesteps per sample.
    """
    with open(original_config_path, 'r') as f:
        original_config = json.load(f)

    total_length = 5 + num_ar_rounds * 5  # e.g., 20 for 3 rounds

    new_config = {
        'metadata': {
            'input_length': 5,
            'output_length': 5,
            'num_ar_rounds': num_ar_rounds,
            'total_length': total_length,
            'description': f'JHU AR testing config for {num_ar_rounds} rounds'
        },
        'splits': {}
    }

    # Get all unique timesteps from original config
    all_timesteps = set()
    for split in ['train', 'val', 'test']:
        if split in original_config['splits']:
            for sample in original_config['splits'][split]:
                all_timesteps.update(sample)

    all_timesteps = sorted(all_timesteps, key=float)
    print(f"Total unique timesteps: {len(all_timesteps)}")
    print(f"Range: {all_timesteps[0]} to {all_timesteps[-1]}")

    # Create extended samples (non-overlapping for test, overlapping for train/val)
    for split in ['train', 'val', 'test']:
        if split not in original_config['splits']:
            continue

        samples = []

        if split == 'test':
            # Non-overlapping for test
            stride = total_length
        else:
            # Overlapping for train/val
            stride = 5

        for i in range(0, len(all_timesteps) - total_length + 1, stride):
            sample = all_timesteps[i:i + total_length]
            if len(sample) == total_length:
                samples.append(sample)

        # Filter to only keep samples that belong to this split
        # Check if the first timestep is in any of the original split samples
        original_starts = set()
        for orig_sample in original_config['splits'][split]:
            original_starts.add(orig_sample[0])

        if split == 'test':
            # For test, keep samples where start is near an original test sample
            filtered_samples = []
            for sample in samples:
                # Check if this sample's start is close to any original test sample start
                sample_start = float(sample[0])
                for orig_start in original_starts:
                    if abs(float(orig_start) - sample_start) < 0.01:  # tolerance
                        filtered_samples.append(sample)
                        break
            samples = filtered_samples if filtered_samples else samples[:len(original_config['splits'][split])//2]

        new_config['splits'][split] = samples
        print(f"  {split}: {len(samples)} samples (need {total_length} timesteps each)")

    # Save
    with open(output_path, 'w') as f:
        json.dump(new_config, f, indent=2)

    print(f"Saved AR config to: {output_path}")
    return new_config


if __name__ == "__main__":
    import sys

    # Create AR split config
    original_config = "Dataset/jhu_data_splits_ar.json"
    output_config = "Dataset/jhu_data_splits_ar_extended.json"

    if len(sys.argv) > 1:
        num_rounds = int(sys.argv[1])
    else:
        num_rounds = 3

    create_ar_split_config(original_config, output_config, num_ar_rounds=num_rounds)
