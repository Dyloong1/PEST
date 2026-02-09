"""
JHU DNS128 数据集

支持JHU DNS128立方体数据 (128, 128, 128)
文件格式: img_{indicator}_{timestep}.npy
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json


class SparseJHUDataset(Dataset):
    """
    JHU DNS128 稀疏数据集

    与SparseDNSDataset类似，但支持:
    - 立方体网格 (128, 128, 128)
    - 不同的文件命名格式
    - 更多时间步 (200个)
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
            data_dir: JHU数据目录
            split: 'train', 'val', 'test'
            split_config_path: 划分配置JSON路径
            indicators: 使用的物理量
            sparse_z_interval: z轴稀疏采样**间隔** (e.g., 8 means [0, 8, 16, ...])
            normalize: 是否归一化
            norm_stats: 归一化统计量字典
            dense_z_size: Z轴总层数 (JHU为128)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.indicators = indicators
        self.dense_z_size = dense_z_size

        if sparse_z_interval is None:
            raise ValueError(
                "sparse_z_interval must be explicitly provided! "
                "This parameter is critical and should match the training configuration."
            )
        self.sparse_z_interval = sparse_z_interval

        self.normalize = normalize

        if split_config_path is None:
            raise ValueError("必须提供 split_config_path")

        with open(split_config_path, 'r') as f:
            config = json.load(f)

        self.sample_indices = config['splits'][split]
        self.metadata = config['metadata']
        self.input_length = self.metadata['input_length']
        self.output_length = self.metadata['output_length']

        # 计算稀疏和缺失层索引
        self.sparse_z_indices = list(range(0, self.dense_z_size, sparse_z_interval))
        self.missing_z_indices = [i for i in range(self.dense_z_size)
                                  if i not in self.sparse_z_indices]

        print(f"\n{split.upper()} JHU数据集初始化:")
        print(f"  数据目录: {data_dir}")
        print(f"  样本数: {len(self.sample_indices)}")
        print(f"  物理量: {indicators}")
        print(f"  网格尺寸: ({self.dense_z_size}, 128, 128) - 立方体")
        print(f"  稀疏z层: {len(self.sparse_z_indices)} / {self.dense_z_size}")
        print(f"  稀疏z索引: {self.sparse_z_indices[:5]}...{self.sparse_z_indices[-3:]}")
        print(f"  缺失z层数: {len(self.missing_z_indices)}")

        # 归一化统计量
        if normalize:
            if norm_stats is None:
                # JHU数据默认统计量 (需要从数据计算更新)
                self.norm_stats = {
                    'u': {'mean': 0.0, 'std': 1.0},
                    'v': {'mean': 0.0, 'std': 1.0},
                    'w': {'mean': 0.0, 'std': 1.0},
                    'p': {'mean': 0.0, 'std': 1.0}
                }
                print(f"  归一化: 启用 (使用默认统计量，建议从数据计算)")
            else:
                self.norm_stats = norm_stats
                print(f"  归一化: 启用")

            for ind in indicators:
                if ind in self.norm_stats:
                    print(f"    {ind}: mean={self.norm_stats[ind]['mean']:.4f}, "
                          f"std={self.norm_stats[ind]['std']:.4f}")
        else:
            print(f"  归一化: 禁用")

    def __len__(self):
        return len(self.sample_indices)

    def _load_frame(self, indicator, timestamp):
        """
        加载单帧数据

        JHU文件命名: {indicator}_{timestep}.npy
        时间戳格式: X.XXX (3位小数，如 5.020)
        """
        # 将时间戳转换为3位小数格式
        # 例如 "5.02" -> "5.020", "5.1" -> "5.100"
        ts_float = float(timestamp)
        ts_formatted = f"{ts_float:.3f}"

        # JHU文件命名格式（无img_前缀）
        filename = f"{indicator}_{ts_formatted}.npy"
        filepath = self.data_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"文件不存在: {filepath}")

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
                'sparse_z_indices': 稀疏z层的索引
                'missing_z_indices': 缺失z层的索引
                'sample_start': 样本起始帧索引
                'global_timesteps': (input_length + output_length,) 全局时间步序列
            }
        """
        # JHU config存储的是完整timestamp列表，例如 ["5.02", "5.03", ..., "5.11"]
        sample_timestamps = self.sample_indices[idx]

        # 分割为input和output timestamps
        input_timestamps = sample_timestamps[:self.input_length]
        output_timestamps = sample_timestamps[self.input_length:]

        # 加载输入数据
        input_data = []
        for indicator in self.indicators:
            indicator_frames = []
            for t in input_timestamps:
                frame = self._load_frame(indicator, t)
                indicator_frames.append(frame)
            input_data.append(np.stack(indicator_frames, axis=0))

        input_dense = np.stack(input_data, axis=1)

        # 加载输出数据
        output_data = []
        for indicator in self.indicators:
            indicator_frames = []
            for t in output_timestamps:
                frame = self._load_frame(indicator, t)
                indicator_frames.append(frame)
            output_data.append(np.stack(indicator_frames, axis=0))

        output_dense = np.stack(output_data, axis=1)
        output_sparse = output_dense[:, :, self.sparse_z_indices, :, :]

        # 转换时间戳为浮点数tensor
        timestamps_float = [float(t) for t in sample_timestamps]

        return {
            'input_dense': torch.from_numpy(input_dense),
            'output_sparse': torch.from_numpy(output_sparse),
            'output_dense': torch.from_numpy(output_dense),
            'sparse_z_indices': torch.tensor(self.sparse_z_indices),
            'missing_z_indices': torch.tensor(self.missing_z_indices),
            'sample_start': idx,  # 使用样本索引
            'global_timesteps': torch.tensor(timestamps_float, dtype=torch.float32)  # 真实时间戳
        }


def compute_jhu_norm_stats(data_dir, indicators=['u', 'v', 'w', 'p'],
                            sample_timesteps=None, max_samples=20):
    """
    计算JHU数据的归一化统计量

    Args:
        data_dir: JHU数据目录
        indicators: 物理量列表
        sample_timesteps: 采样的时间步（None则自动选择）
        max_samples: 最大采样数量

    Returns:
        dict: 归一化统计量
    """
    data_dir = Path(data_dir)

    # 自动发现时间步
    if sample_timesteps is None:
        all_files = list(data_dir.glob("u_*.npy"))
        timesteps = sorted([f.stem.split('_')[-1] for f in all_files])

        if len(timesteps) > max_samples:
            step = len(timesteps) // max_samples
            sample_timesteps = timesteps[::step][:max_samples]
        else:
            sample_timesteps = timesteps

    print(f"计算归一化统计量...")
    print(f"  数据目录: {data_dir}")
    print(f"  采样时间步数: {len(sample_timesteps)}")

    norm_stats = {}

    for indicator in indicators:
        all_data = []
        for t in sample_timesteps:
            filepath = data_dir / f"{indicator}_{t}.npy"
            if filepath.exists():
                data = np.load(filepath)
                all_data.append(data.flatten())

        if all_data:
            combined = np.concatenate(all_data)
            norm_stats[indicator] = {
                'mean': float(np.mean(combined)),
                'std': float(np.std(combined))
            }
            print(f"  {indicator}: mean={norm_stats[indicator]['mean']:.6f}, "
                  f"std={norm_stats[indicator]['std']:.6f}")

    return norm_stats


def create_jhu_dataloaders(data_dir,
                           split_config_path,
                           indicators=['u', 'v', 'w', 'p'],
                           sparse_z_interval=None,
                           batch_size=1,
                           num_workers=0,  # JHU data is large (~320MB/sample), use 0 to avoid memory issues
                           normalize=True,
                           norm_stats=None,
                           dense_z_size=128):
    """
    创建JHU数据的训练/验证/测试数据加载器

    Args:
        data_dir: JHU数据目录
        split_config_path: 数据划分配置路径
        indicators: 物理量列表
        sparse_z_interval: z轴稀疏采样**间隔** (e.g., 8 means [0, 8, 16, ...])
        batch_size: 批大小
        num_workers: 工作线程数
        normalize: 是否归一化
        norm_stats: 归一化统计量
        dense_z_size: Z轴总层数

    Returns:
        train_loader, val_loader, test_loader
    """
    if sparse_z_interval is None:
        raise ValueError(
            "sparse_z_interval must be explicitly provided to create_jhu_dataloaders! "
            "This ensures consistency across train/val/test splits."
        )

    train_dataset = SparseJHUDataset(
        data_dir=data_dir,
        split='train',
        split_config_path=split_config_path,
        indicators=indicators,
        sparse_z_interval=sparse_z_interval,
        normalize=normalize,
        norm_stats=norm_stats,
        dense_z_size=dense_z_size
    )

    val_dataset = SparseJHUDataset(
        data_dir=data_dir,
        split='val',
        split_config_path=split_config_path,
        indicators=indicators,
        sparse_z_interval=sparse_z_interval,
        normalize=normalize,
        norm_stats=norm_stats,
        dense_z_size=dense_z_size
    )

    test_dataset = SparseJHUDataset(
        data_dir=data_dir,
        split='test',
        split_config_path=split_config_path,
        indicators=indicators,
        sparse_z_interval=sparse_z_interval,
        normalize=normalize,
        norm_stats=norm_stats,
        dense_z_size=dense_z_size
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # 测试数据统计量计算
    jhu_path = "/media/ydai17/T7 Shield/JHU_data/DNS128"
    print("Testing JHU dataset utilities...")

    # 计算归一化统计量
    stats = compute_jhu_norm_stats(jhu_path, max_samples=10)
    print(f"\nComputed stats: {stats}")
