"""
Autoregressive测试脚本

功能:
1. 多轮自回归测试 (Round 1使用GT输入，Round 2+使用预测作为输入)
2. 支持DNS和JHU两种数据集
3. 支持FNO和Swin Transformer模型
4. 使用固定采样位置进行可视化（用于跨方法对比）
5. 输出详细的数值评估指标 (JSON)
6. 按轮次、时间步、z层、通道分别统计误差
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from collections import defaultdict
import sys

# 添加路径
CODE_ROOT = Path(__file__).parent
sys.path.insert(0, str(CODE_ROOT))


class NumpyEncoder(json.JSONEncoder):
    """自定义JSON编码器"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return super().default(obj)


def load_sampling_config(data_source):
    """加载固定采样位置配置"""
    if data_source == 'dns':
        config_path = CODE_ROOT / 'Data' / 'test_sampling_config_dns.json'
    else:
        config_path = CODE_ROOT / 'Dataset' / 'test_sampling_config_jhu.json'

    with open(config_path, 'r') as f:
        return json.load(f)


def load_normalization_stats(data_source):
    """加载归一化统计信息"""
    if data_source == 'dns':
        stats_path = CODE_ROOT / 'Data' / 'normalization_stats.json'
        with open(stats_path, 'r') as f:
            raw_stats = json.load(f)

        stats = {}
        channels = raw_stats.get('channels', ['u', 'v', 'w', 'p'])
        means = raw_stats['mean']
        stds = raw_stats['std']

        for i, ch in enumerate(channels):
            stats[ch] = {'mean': means[i], 'std': stds[i]}
    else:
        # JHU数据 - 需要转换JSON格式
        jhu_stats_path = CODE_ROOT / 'Data' / 'jhu_normalization_stats.json'
        if jhu_stats_path.exists():
            with open(jhu_stats_path, 'r') as f:
                raw_stats = json.load(f)
            # Convert from {"mean": [...], "std": [...], "indicators": [...]}
            # to {"u": {"mean": x, "std": y}, ...}
            stats = {}
            channels = raw_stats.get('indicators', ['u', 'v', 'w', 'p'])
            for i, ch in enumerate(channels):
                stats[ch] = {'mean': raw_stats['mean'][i], 'std': raw_stats['std'][i]}
        else:
            stats = {
                'u': {'mean': 0.0, 'std': 1.0},
                'v': {'mean': 0.0, 'std': 1.0},
                'w': {'mean': 0.0, 'std': 1.0},
                'p': {'mean': 0.0, 'std': 1.0}
            }
            print("WARNING: JHU normalization stats not found, using defaults!")

    return stats


def denormalize_field(data, channel_name, stats):
    """反归一化"""
    if channel_name in stats:
        mean = stats[channel_name]['mean']
        std = stats[channel_name]['std']
        return data * std + mean
    return data


def compute_divergence(pred, channels=['u', 'v', 'w'], dx=1.0, dy=1.0, dz=1.0):
    """计算3D散度"""
    if len(channels) < 3:
        return None

    u = pred[..., 0, :, :, :]
    v = pred[..., 1, :, :, :]
    w = pred[..., 2, :, :, :]

    du_dx = (u[..., :, :, 2:] - u[..., :, :, :-2]) / (2 * dx)
    dv_dy = (v[..., :, 2:, :] - v[..., :, :-2, :]) / (2 * dy)
    dw_dz = (w[..., 2:, :, :] - w[..., :-2, :, :]) / (2 * dz)

    min_z = min(du_dx.shape[-3], dv_dy.shape[-3], dw_dz.shape[-3])
    min_h = min(du_dx.shape[-2], dv_dy.shape[-2], dw_dz.shape[-2])
    min_w = min(du_dx.shape[-1], dv_dy.shape[-1], dw_dz.shape[-1])

    du_dx = du_dx[..., :min_z, :min_h, :min_w]
    dv_dy = dv_dy[..., :min_z, :min_h, :min_w]
    dw_dz = dw_dz[..., :min_z, :min_h, :min_w]

    return du_dx + dv_dy + dw_dz


def load_model(model_type, config_name, checkpoint_path, device, data_source='dns', jhu_config='baseline', dns_config='baseline'):
    """加载模型"""
    if model_type == 'swin':
        from Model.swin_transformer_model_patched import SwinTransformerPhysics
        from Model.configs.hollow_fix_configs import HOLLOW_FIX_CONFIGS
        from Model.configs.gradient_weight_configs import GRADIENT_WEIGHT_CONFIGS

        # 合并所有配置
        ALL_CONFIGS = {**HOLLOW_FIX_CONFIGS, **GRADIENT_WEIGHT_CONFIGS}

        # 根据数据源和配置名选择配置
        if data_source == 'jhu':
            # JHU使用专门的配置
            from Model.configs.swin_jhu_config import get_jhu_swin_config
            config = get_jhu_swin_config(jhu_config)
            print(f"JHU config: {jhu_config} ({config.__class__.__name__})")
        elif data_source == 'dns':
            # DNS使用新配置系统
            from Model.configs.swin_patched_config_fixed import get_dns_swin_config
            config = get_dns_swin_config(dns_config)
            print(f"DNS config: {dns_config} ({config.__class__.__name__})")
        elif config_name in ALL_CONFIGS:
            config = ALL_CONFIGS[config_name]()
        else:
            from Model.configs.swin_patched_config_fixed import SwinPatchedConfigFixed
            config = SwinPatchedConfigFixed()

        # 两种数据源都用4通道
        config.IN_CHANNELS = 4

        model = SwinTransformerPhysics(config).to(device)

    elif model_type == 'fno':
        from Model.fno_model import FNO3d
        from Model.configs.config_examples import FNOConfig as OriginalFNOConfig

        # 创建FNO配置 (与train_fno_unified.py保持一致)
        config = OriginalFNOConfig()
        config.FNO_WIDTH = 44
        config.FNO_LAYERS = 4
        config.USE_GRADIENT_CHECKPOINT = False  # 测试时不需要

        # 两种数据源都用4通道
        config.INDICATORS = ['u', 'v', 'w', 'p']
        if data_source == 'jhu':
            config.TOTAL_Z_LAYERS = 128
            config.SPARSE_Z_LAYERS = 16
            config.FNO_MODES_Z = 18
            config.FNO_MODES_H = 28
            config.FNO_MODES_W = 28
        else:  # dns
            config.TOTAL_Z_LAYERS = 64  # 原始65层，在数据加载时截断为64层
            config.SPARSE_Z_LAYERS = 8
            config.FNO_MODES_Z = 18
            config.FNO_MODES_H = 36
            config.FNO_MODES_W = 36

        model = FNO3d(config).to(device)

    elif model_type == 'dual_residual':
        from Model.swin_dual_residual import SwinDualResidual, DualResidualConfigJHU

        config = DualResidualConfigJHU()
        model = SwinDualResidual(config).to(device)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    epoch = checkpoint.get('epoch', 'unknown')
    print(f"  Model loaded from epoch {epoch}")

    return model, config


def create_dataloader(data_source, split='test', batch_size=1, sparse_z_interval=8):
    """创建数据加载器"""
    if data_source == 'dns':
        from Dataset.sparse_dataset import SparseDNSDataset

        data_dir = '/home/ydai17/Turbo/DNS_data/DNS_data'
        split_config_path = CODE_ROOT / 'Data' / 'split_config_ar.json'

        dataset = SparseDNSDataset(
            data_dir=data_dir,
            split=split,
            split_config_path=str(split_config_path),
            indicators=['u', 'v', 'w', 'p'],  # 4通道
            sparse_z_interval=sparse_z_interval,
            normalize=True
        )
    else:
        from Dataset.jhu_dataset import SparseJHUDataset

        data_dir = '/media/ydai17/T7 Shield/JHU_data/DNS128'
        split_config_path = CODE_ROOT / 'Dataset' / 'jhu_data_splits_ar.json'

        # 加载归一化统计量
        norm_stats = load_normalization_stats('jhu')

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

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return loader, dataset


def prepare_model_input(model_type, dense_data, sparse_indices, device, output_sparse=None):
    """准备模型输入

    NOTE: DNS数据已在数据加载时截断为64层，无需额外处理
    """
    if model_type == 'swin':
        # Swin需要batch格式，包含output_sparse_z用于decoder guidance
        batch = {
            'input_dense': dense_data.to(device),
            'sparse_indices': {'z': sparse_indices.to(device) if isinstance(sparse_indices, torch.Tensor) else torch.tensor(sparse_indices).to(device)}
        }
        if output_sparse is not None:
            batch['output_sparse_z'] = output_sparse.to(device)
        return batch, dense_data
    elif model_type == 'dual_residual':
        # DualResidual使用不同的key
        batch = {
            'input_dense': dense_data.to(device),
            'sparse_info': {'sparse_z_indices': sparse_indices.to(device) if isinstance(sparse_indices, torch.Tensor) else torch.tensor(sparse_indices).to(device)}
        }
        if output_sparse is not None:
            batch['output_sparse'] = output_sparse.to(device)
        return batch, dense_data
    elif model_type == 'fno':
        # FNO直接使用dense输入
        return dense_data.to(device), dense_data
    return None, dense_data


def run_model_inference(model, model_type, model_input, output_sparse=None,
                        sparse_z_indices=None, missing_z_indices=None):
    """运行模型推理"""
    with torch.no_grad():
        if model_type == 'swin':
            # Swin支持混合精度
            with torch.amp.autocast('cuda'):
                output = model(model_input)
                pred = output['output_full']  # (B, T, C, Z, H, W)
        elif model_type == 'dual_residual':
            # DualResidual支持混合精度
            with torch.amp.autocast('cuda'):
                output = model(model_input)
                pred = output['output_full']  # (B, T, C, Z, H, W)
        elif model_type == 'fno':
            # FNO需要完整输入
            # FNO的复数einsum不支持半精度，必须用fp32
            pred = model(
                model_input,
                output_sparse,
                sparse_z_indices,
                missing_z_indices
            )  # (B, T_out, C, Z, H, W)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    return pred


def load_gt_frames(data_source, start_frame, num_frames, channels, norm_stats, dense_z_size):
    """
    直接从数据目录加载GT帧用于AR测试Round 2+

    Args:
        data_source: 'dns' or 'jhu'
        start_frame: 起始帧索引/时间戳
        num_frames: 需要加载的帧数
        channels: 通道列表
        norm_stats: 归一化统计量
        dense_z_size: Z维度大小

    Returns:
        gt_data: (1, num_frames, C, Z, H, W) tensor，如果加载失败返回None
    """
    try:
        if data_source == 'dns':
            data_dir = Path('/home/ydai17/Turbo/DNS_data/DNS_data')
            frames_data = []
            for t in range(start_frame, start_frame + num_frames):
                channel_data = []
                for ch in channels:
                    filepath = data_dir / f"img_{ch}_dns{t}.npy"
                    if not filepath.exists():
                        return None
                    data = np.load(filepath).astype(np.float32)
                    data = data[:dense_z_size, :, :]  # 截断到64层
                    if ch in norm_stats:
                        data = (data - norm_stats[ch]['mean']) / (norm_stats[ch]['std'] + 1e-8)
                    channel_data.append(data)
                frames_data.append(np.stack(channel_data, axis=0))  # (C, Z, H, W)
            gt = np.stack(frames_data, axis=0)  # (T, C, Z, H, W)
            return torch.from_numpy(gt).unsqueeze(0)  # (1, T, C, Z, H, W)
        else:  # jhu
            data_dir = Path('/media/ydai17/T7 Shield/JHU_data/DNS128')
            # JHU时间戳需要特殊处理
            # start_frame在JHU中是浮点数时间戳
            frames_data = []
            for i in range(num_frames):
                t = start_frame + i * 0.01  # JHU时间步长是0.01
                ts_formatted = f"{t:.3f}"
                channel_data = []
                for ch in channels:
                    filepath = data_dir / f"{ch}_{ts_formatted}.npy"
                    if not filepath.exists():
                        return None
                    data = np.load(filepath).astype(np.float32)
                    if ch in norm_stats:
                        data = (data - norm_stats[ch]['mean']) / (norm_stats[ch]['std'] + 1e-8)
                    channel_data.append(data)
                frames_data.append(np.stack(channel_data, axis=0))
            gt = np.stack(frames_data, axis=0)
            return torch.from_numpy(gt).unsqueeze(0)
    except Exception as e:
        print(f"  Warning: Failed to load GT frames starting at {start_frame}: {e}")
        return None


def autoregressive_test(model, model_type, test_loader, num_ar_rounds,
                        channels, sparse_z_interval, dense_z_size, device,
                        sampling_config, norm_stats, save_dir, data_source='dns'):
    """
    自回归测试

    Args:
        model: 模型
        model_type: 'swin' or 'fno'
        test_loader: 测试数据加载器
        num_ar_rounds: AR轮数
        channels: 通道名称列表
        sparse_z_interval: 稀疏步长
        dense_z_size: Z维度大小 (DNS=64, JHU=128)
        device: 设备
        sampling_config: 固定采样配置
        norm_stats: 归一化统计量
        save_dir: 保存目录
        data_source: 'dns' or 'jhu'

    NOTE: DNS数据已在数据加载时截断为64层
    """
    num_channels = len(channels)
    output_length = 5

    # 计算sparse和missing indices
    sparse_z_indices = list(range(0, dense_z_size, sparse_z_interval))
    missing_z_indices = [i for i in range(dense_z_size) if i not in sparse_z_indices]

    # 固定可视化位置
    vis_z_slices = sampling_config['visualization_points']['z_slices']['indices']
    vis_xy_points = sampling_config['visualization_points']['xy_points']['grid_points']

    # 结果容器
    results = {
        'model_type': model_type,
        'num_ar_rounds': num_ar_rounds,
        'channels': channels,
        'sparse_z_interval': sparse_z_interval,
        'dense_z_size': dense_z_size,
        'num_sparse_layers': len(sparse_z_indices),
        'num_missing_layers': len(missing_z_indices),
        'per_round': {},
        'per_sample': [],
        'global': {}
    }

    # 按轮次收集误差
    round_mse_norm = {r: [[] for _ in range(num_channels)] for r in range(1, num_ar_rounds + 1)}
    round_mse_phys = {r: [[] for _ in range(num_channels)] for r in range(1, num_ar_rounds + 1)}
    round_div_error = {r: [] for r in range(1, num_ar_rounds + 1)}

    # 可视化数据
    vis_data = []

    print(f"\nRunning {num_ar_rounds}-round autoregressive test...")
    print(f"  Sparse layers: {len(sparse_z_indices)}, Missing layers: {len(missing_z_indices)}")

    for batch_idx, batch in enumerate(tqdm(test_loader, desc="AR Testing")):
        sample_result = {
            'sample_idx': batch_idx,
            'sample_start': batch['sample_start'].item() if isinstance(batch['sample_start'], torch.Tensor) else batch['sample_start'],
            'rounds': {}
        }

        # 初始输入 (GT)
        current_input = batch['input_dense'].to(device)  # (B, T_in, C, Z, H, W)
        B, T_in, C, Z, H, W = current_input.shape

        sparse_indices_tensor = batch['sparse_z_indices'].to(device)  # (B, n_sparse)
        missing_indices_tensor = batch['missing_z_indices'].to(device)  # (B, n_missing)
        sparse_indices = batch['sparse_z_indices'][0]  # 假设batch_size=1

        # 存储所有轮次的预测和GT
        all_round_preds = []
        all_round_gts = []

        # 初始化帧追踪
        sample_start = batch['sample_start'].item() if isinstance(batch['sample_start'], torch.Tensor) else batch['sample_start']
        if data_source == 'dns':
            next_gt_start_frame = sample_start + 5  # Round 1 GT从sample_start+5开始
        else:  # jhu
            timestamps = batch.get('global_timesteps', None)
            # timestamps shape: (B, T) after DataLoader, 需要用shape[-1]而不是len()
            if timestamps is not None and timestamps.shape[-1] > 5:
                next_gt_start_frame = float(timestamps[0, 5].item())  # 第6帧开始是输出
            else:
                next_gt_start_frame = None

        for ar_round in range(1, num_ar_rounds + 1):
            # 1. 首先获取当前轮次的GT（用于计算误差和提取sparse guidance）
            if ar_round == 1:
                gt = batch['output_dense']  # Round 1的GT (已截断)
                current_output_sparse = batch['output_sparse'].to(device)
                if data_source == 'dns':
                    next_gt_start_frame = sample_start + 10
                else:
                    if next_gt_start_frame is not None:
                        next_gt_start_frame = next_gt_start_frame + output_length * 0.01
            else:
                # Round 2+: 从数据目录加载对应帧作为GT
                if data_source == 'dns' and next_gt_start_frame is not None:
                    gt = load_gt_frames(data_source, next_gt_start_frame, output_length, channels, norm_stats, dense_z_size)
                    if gt is not None:
                        sparse_idx_list = sparse_indices.tolist() if isinstance(sparse_indices, torch.Tensor) else list(sparse_indices)
                        current_output_sparse = gt[:, :, :, sparse_idx_list, :, :].to(device)
                        next_gt_start_frame = next_gt_start_frame + output_length
                    else:
                        # 没有GT，使用上一轮预测的sparse层
                        current_output_sparse = all_round_preds[-1][:, :, :, sparse_idx_list, :, :].to(device)
                elif data_source == 'jhu' and next_gt_start_frame is not None:
                    gt = load_gt_frames(data_source, next_gt_start_frame, output_length, channels, norm_stats, dense_z_size)
                    if gt is not None:
                        sparse_idx_list = sparse_indices.tolist() if isinstance(sparse_indices, torch.Tensor) else list(sparse_indices)
                        current_output_sparse = gt[:, :, :, sparse_idx_list, :, :].to(device)
                        next_gt_start_frame = next_gt_start_frame + output_length * 0.01
                    else:
                        current_output_sparse = all_round_preds[-1][:, :, :, sparse_idx_list, :, :].to(device)
                else:
                    gt = None
                    if len(all_round_preds) > 0:
                        sparse_idx_list = sparse_indices.tolist() if isinstance(sparse_indices, torch.Tensor) else list(sparse_indices)
                        current_output_sparse = all_round_preds[-1][:, :, :, sparse_idx_list, :, :].to(device)

            # 2. 准备模型输入
            model_input, _ = prepare_model_input(
                model_type, current_input, sparse_indices, device,
                output_sparse=current_output_sparse
            )

            # 3. 推理
            if model_type == 'fno':
                pred = run_model_inference(
                    model, model_type, model_input,
                    output_sparse=current_output_sparse,
                    sparse_z_indices=sparse_indices_tensor,
                    missing_z_indices=missing_indices_tensor
                )
            else:
                pred = run_model_inference(model, model_type, model_input)
            pred = pred.cpu()  # (B, T_out, C, Z, H, W)

            # 只在missing层计算误差
            pred_missing = pred[0, :, :, missing_z_indices, :, :]  # (T, C, M, H, W)

            if gt is not None:
                gt_missing = gt[0, :, :, missing_z_indices, :, :]  # (T, C, M, H, W)

                # 计算逐通道MSE
                round_sample_mse_norm = []
                round_sample_mse_phys = []

                for c in range(num_channels):
                    mse_norm = ((pred_missing[:, c] - gt_missing[:, c]) ** 2).mean().item()
                    round_mse_norm[ar_round][c].append(mse_norm)
                    round_sample_mse_norm.append(mse_norm)

                    # 物理空间
                    pred_phys = denormalize_field(pred_missing[:, c].numpy(), channels[c], norm_stats)
                    gt_phys = denormalize_field(gt_missing[:, c].numpy(), channels[c], norm_stats)
                    mse_phys = float(((pred_phys - gt_phys) ** 2).mean())
                    round_mse_phys[ar_round][c].append(mse_phys)
                    round_sample_mse_phys.append(mse_phys)

                # 散度误差
                if num_channels >= 3:
                    pred_full_phys = torch.zeros_like(pred)
                    gt_full_phys = torch.zeros_like(gt)

                    for c in range(min(3, num_channels)):
                        pred_full_phys[0, :, c] = torch.from_numpy(
                            denormalize_field(pred[0, :, c].numpy(), channels[c], norm_stats)
                        )
                        gt_full_phys[0, :, c] = torch.from_numpy(
                            denormalize_field(gt[0, :, c].numpy(), channels[c], norm_stats)
                        )

                    div_pred = compute_divergence(pred_full_phys)
                    div_gt = compute_divergence(gt_full_phys)

                    if div_pred is not None and div_gt is not None:
                        div_error = torch.abs(div_pred - div_gt).mean().item()
                        round_div_error[ar_round].append(div_error)

                sample_result['rounds'][ar_round] = {
                    'mse_normalized': round_sample_mse_norm,
                    'mse_physical': round_sample_mse_phys,
                    'rmse_normalized': [np.sqrt(m) for m in round_sample_mse_norm],
                    'rmse_physical': [np.sqrt(m) for m in round_sample_mse_phys]
                }

            all_round_preds.append(pred.clone())
            if gt is not None:
                all_round_gts.append(gt.clone())

            # 准备下一轮输入: 使用当前预测的5帧作为下一轮输入
            current_input = pred.to(device)

        results['per_sample'].append(sample_result)

        # 收集可视化数据 (只保留少量样本)
        if len(vis_data) < 5:
            vis_data.append({
                'sample_idx': batch_idx,
                'sample_start': sample_result['sample_start'],
                'input': batch['input_dense'][0].cpu().numpy(),
                'predictions': [p[0].cpu().numpy() for p in all_round_preds],
                'ground_truths': [g[0].cpu().numpy() for g in all_round_gts] if all_round_gts else None,
                'sparse_z_indices': sparse_z_indices,
                'missing_z_indices': missing_z_indices
            })

    # 计算全局统计
    print("\nComputing global statistics...")

    for ar_round in range(1, num_ar_rounds + 1):
        results['per_round'][ar_round] = {
            'mse_normalized': [np.mean(round_mse_norm[ar_round][c]) if round_mse_norm[ar_round][c] else 0
                              for c in range(num_channels)],
            'mse_physical': [np.mean(round_mse_phys[ar_round][c]) if round_mse_phys[ar_round][c] else 0
                           for c in range(num_channels)],
            'rmse_normalized': [np.sqrt(np.mean(round_mse_norm[ar_round][c])) if round_mse_norm[ar_round][c] else 0
                               for c in range(num_channels)],
            'rmse_physical': [np.sqrt(np.mean(round_mse_phys[ar_round][c])) if round_mse_phys[ar_round][c] else 0
                            for c in range(num_channels)],
            'divergence_error_mean': float(np.mean(round_div_error[ar_round])) if round_div_error[ar_round] else 0,
            'divergence_error_std': float(np.std(round_div_error[ar_round])) if round_div_error[ar_round] else 0,
            'num_samples': len(round_mse_norm[ar_round][0]) if round_mse_norm[ar_round][0] else 0
        }

    # 全局平均（Round 1）
    results['global'] = results['per_round'].get(1, {})

    return results, vis_data


def save_visualizations(vis_data, sampling_config, channels, save_dir):
    """
    保存固定位置的可视化

    每轮单独保存PNG，每个通道都有：预测、GT、误差 三行对比
    """
    vis_dir = save_dir / 'visualizations'
    vis_dir.mkdir(exist_ok=True, parents=True)

    vis_z_slices = sampling_config['visualization_points']['z_slices']['indices']

    for data in vis_data:
        sample_idx = data['sample_idx']
        predictions = data['predictions']
        ground_truths = data['ground_truths']

        num_rounds = len(predictions)
        num_channels = predictions[0].shape[1]

        # 选择代表性z切片进行可视化
        for z_idx in vis_z_slices[:3]:  # 最多3个z切片
            if z_idx >= predictions[0].shape[2]:
                continue

            # 每轮单独保存一个PNG
            for r in range(num_rounds):
                pred = predictions[r]
                gt = ground_truths[r] if ground_truths and len(ground_truths) > r else None

                if gt is None:
                    continue  # 没有GT就跳过这轮的可视化

                # 创建图: 3行(Pred/GT/Error) x num_channels列
                fig, axes = plt.subplots(3, num_channels, figsize=(4 * num_channels, 12))

                # 确保axes是2D数组
                if num_channels == 1:
                    axes = axes.reshape(-1, 1)

                fig.suptitle(f'Sample {sample_idx} | Round {r+1} | z={z_idx} (Missing Layer)', fontsize=16)

                # 选择中间时间步 t=2
                t_idx = 2

                for c in range(num_channels):
                    pred_slice = pred[t_idx, c, z_idx]
                    gt_slice = gt[t_idx, c, z_idx]
                    error_slice = pred_slice - gt_slice

                    # 计算统一的颜色范围
                    vmin = min(pred_slice.min(), gt_slice.min())
                    vmax = max(pred_slice.max(), gt_slice.max())
                    error_max = max(abs(error_slice.min()), abs(error_slice.max()), 0.01)

                    # 行1: 预测
                    im1 = axes[0, c].imshow(pred_slice, cmap='RdBu_r', vmin=vmin, vmax=vmax)
                    axes[0, c].set_title(f'Prediction {channels[c].upper()}', fontsize=12)
                    axes[0, c].axis('off')
                    plt.colorbar(im1, ax=axes[0, c], fraction=0.046, pad=0.04)

                    # 行2: GT
                    im2 = axes[1, c].imshow(gt_slice, cmap='RdBu_r', vmin=vmin, vmax=vmax)
                    axes[1, c].set_title(f'Ground Truth {channels[c].upper()}', fontsize=12)
                    axes[1, c].axis('off')
                    plt.colorbar(im2, ax=axes[1, c], fraction=0.046, pad=0.04)

                    # 行3: 误差
                    im3 = axes[2, c].imshow(error_slice, cmap='seismic', vmin=-error_max, vmax=error_max)
                    rmse = np.sqrt((error_slice ** 2).mean())
                    axes[2, c].set_title(f'Error {channels[c].upper()} (RMSE={rmse:.4f})', fontsize=12)
                    axes[2, c].axis('off')
                    plt.colorbar(im3, ax=axes[2, c], fraction=0.046, pad=0.04)

                plt.tight_layout()
                save_path = vis_dir / f'sample{sample_idx:03d}_round{r+1}_z{z_idx:03d}.png'
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()

    print(f"  Saved visualizations to {vis_dir}")


def save_predictions_for_spectral_analysis(vis_data, save_dir, channels):
    """
    保存预测和GT数据用于频谱分析

    Args:
        vis_data: 可视化数据列表
        save_dir: 保存目录
        channels: 通道名称列表

    Returns:
        str: 保存的NPZ文件路径
    """
    spectral_dir = save_dir / 'spectral_analysis'
    spectral_dir.mkdir(exist_ok=True, parents=True)

    # 聚合所有样本的Round 1和Round 2数据
    all_pred_r1 = []
    all_gt_r1 = []
    all_pred_r2 = []
    all_gt_r2 = []

    for data in vis_data:
        predictions = data['predictions']
        ground_truths = data['ground_truths']

        if ground_truths is None or len(ground_truths) < 1:
            continue

        # Round 1
        all_pred_r1.append(predictions[0])  # (T, C, Z, H, W)
        all_gt_r1.append(ground_truths[0])

        # Round 2 (如果有)
        if len(predictions) >= 2 and len(ground_truths) >= 2:
            all_pred_r2.append(predictions[1])
            all_gt_r2.append(ground_truths[1])

    if len(all_pred_r1) == 0:
        print("  Warning: No data available for spectral analysis")
        return None

    # 转换为numpy数组并取平均 (或者堆叠)
    # 这里取第一个样本的中间时间步作为代表
    pred_r1 = all_pred_r1[0]  # (T, C, Z, H, W)
    gt_r1 = all_gt_r1[0]
    t_mid = pred_r1.shape[0] // 2
    pred_r1_mid = pred_r1[t_mid]  # (C, Z, H, W)
    gt_r1_mid = gt_r1[t_mid]

    if len(all_pred_r2) > 0:
        pred_r2 = all_pred_r2[0]
        gt_r2 = all_gt_r2[0]
        pred_r2_mid = pred_r2[t_mid]
        gt_r2_mid = gt_r2[t_mid]
    else:
        # 如果没有Round 2，复制Round 1
        pred_r2_mid = pred_r1_mid
        gt_r2_mid = gt_r1_mid
        print("  Warning: No Round 2 data, using Round 1 for both")

    # 保存为NPZ
    npz_path = spectral_dir / 'predictions_for_spectral.npz'
    np.savez(
        npz_path,
        pred_r1=pred_r1_mid,
        gt_r1=gt_r1_mid,
        pred_r2=pred_r2_mid,
        gt_r2=gt_r2_mid,
        channels=np.array(channels)
    )

    print(f"  Saved predictions for spectral analysis to {npz_path}")
    return str(npz_path)


def run_spectral_analysis(npz_path, output_dir, channels):
    """
    运行频谱误差分析

    Args:
        npz_path: 保存的NPZ文件路径
        output_dir: 输出目录
        channels: 通道名称列表
    """
    try:
        from analyze_spectral_error import run_spectral_analysis_from_npz
        results_r1, results_r2, phase_r1, phase_r2 = run_spectral_analysis_from_npz(
            npz_path, output_dir, channels
        )
        return True
    except Exception as e:
        print(f"  Warning: Spectral analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def plot_ar_error_curves(results, save_dir):
    """绘制AR误差累积曲线"""
    plots_dir = save_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    channels = results['channels']
    num_rounds = results['num_ar_rounds']

    if num_rounds < 2:
        print("  Only 1 AR round, skipping error curve plot")
        return

    rounds = list(range(1, num_rounds + 1))

    # RMSE随AR轮数变化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Normalized RMSE
    ax = axes[0]
    for c, ch in enumerate(channels):
        rmse_values = [results['per_round'][r]['rmse_normalized'][c] for r in rounds]
        ax.plot(rounds, rmse_values, 'o-', label=ch.upper(), linewidth=2, markersize=8)

    ax.set_xlabel('AR Round', fontsize=12)
    ax.set_ylabel('RMSE (Normalized)', fontsize=12)
    ax.set_title('Error Accumulation vs AR Rounds', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(rounds)

    # Physical RMSE
    ax = axes[1]
    for c, ch in enumerate(channels):
        rmse_values = [results['per_round'][r]['rmse_physical'][c] for r in rounds]
        ax.plot(rounds, rmse_values, 'o-', label=ch.upper(), linewidth=2, markersize=8)

    ax.set_xlabel('AR Round', fontsize=12)
    ax.set_ylabel('RMSE (Physical)', fontsize=12)
    ax.set_title('Error Accumulation vs AR Rounds', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(rounds)

    plt.tight_layout()
    plt.savefig(plots_dir / 'ar_error_accumulation.png', dpi=150)
    plt.close()

    print(f"  Saved error curve to {plots_dir / 'ar_error_accumulation.png'}")


def main():
    parser = argparse.ArgumentParser(description='Autoregressive Test for Turbulence Models')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, required=True, choices=['swin', 'fno', 'dual_residual'],
                        help='Model type: swin, fno, or dual_residual')
    parser.add_argument('--config_name', type=str, default='mid',
                        help='Config name for Swin models (legacy DNS)')
    parser.add_argument('--dns_config', type=str, default='baseline',
                        choices=['baseline', 'baseline_v2', 'gradient', 'spectral_mid', 'spectral_v2', 'spectral_gradient', 'lite', 'large'],
                        help='DNS config name: baseline, baseline_v2, gradient, spectral_mid, spectral_v2, spectral_gradient, etc.')
    parser.add_argument('--jhu_config', type=str, default='baseline',
                        choices=['baseline', 'physical', 'lite', 'full', 'large',
                                 'spectral_mid', 'spectral_high', 'spectral_band', 'dual_residual',
                                 'curriculum', 'ns5x', 'spectral_progressive',
                                 'spectral_progressive_large', 'spectral_progressive_giant',
                                 'pde_refiner_large', 'pde_refiner_xl'],
                        help='JHU config name')
    parser.add_argument('--data_source', type=str, required=True, choices=['dns', 'jhu'],
                        help='Data source: dns or jhu')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: checkpoint_dir/test_ar_results)')
    parser.add_argument('--num_ar_rounds', type=int, default=2,
                        help='Number of autoregressive rounds')
    parser.add_argument('--sparse_z_interval', type=int, default=8,
                        help='Sparse z step')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of test samples')
    parser.add_argument('--spectral_analysis', action='store_true',
                        help='Run spectral/frequency error analysis')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"Autoregressive Test")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Model: {args.model_type}")
    print(f"Data: {args.data_source}")
    print(f"AR Rounds: {args.num_ar_rounds}")
    print(f"Checkpoint: {args.checkpoint}")

    # 加载配置
    sampling_config = load_sampling_config(args.data_source)
    norm_stats = load_normalization_stats(args.data_source)

    # 数据集参数 - 两种数据源都用4通道 (u, v, w, p)
    channels = ['u', 'v', 'w', 'p']
    if args.data_source == 'dns':
        dense_z_size = 64  # 原始65层，在数据加载时截断为64层
    else:
        dense_z_size = 128

    # 加载模型
    print(f"\nLoading model...")
    model, config = load_model(
        args.model_type, args.config_name,
        args.checkpoint, device, args.data_source,
        jhu_config=args.jhu_config,
        dns_config=args.dns_config
    )

    # 创建数据加载器
    print(f"\nLoading test data...")
    test_loader, test_dataset = create_dataloader(
        args.data_source,
        split='test',
        batch_size=1,
        sparse_z_interval=args.sparse_z_interval
    )

    # 设置输出目录
    if args.output_dir:
        save_dir = Path(args.output_dir)
    else:
        save_dir = Path(args.checkpoint).parent / 'test_ar_results'
    save_dir.mkdir(exist_ok=True, parents=True)

    # 运行AR测试
    results, vis_data = autoregressive_test(
        model=model,
        model_type=args.model_type,
        test_loader=test_loader,
        num_ar_rounds=args.num_ar_rounds,
        channels=channels,
        sparse_z_interval=args.sparse_z_interval,
        dense_z_size=dense_z_size,
        device=device,
        sampling_config=sampling_config,
        norm_stats=norm_stats,
        save_dir=save_dir,
        data_source=args.data_source
    )

    # 保存结果
    results_path = save_dir / 'ar_evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to: {results_path}")

    # 打印结果摘要
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")

    for ar_round in range(1, args.num_ar_rounds + 1):
        if ar_round in results['per_round']:
            r = results['per_round'][ar_round]
            print(f"\nRound {ar_round}:")
            for c, ch in enumerate(channels):
                print(f"  {ch.upper()}: RMSE(norm)={r['rmse_normalized'][c]:.6f}, "
                      f"RMSE(phys)={r['rmse_physical'][c]:.6f}")
            if r.get('divergence_error_mean', 0) > 0:
                print(f"  Divergence Error: {r['divergence_error_mean']:.6e}")

    # 生成可视化
    print(f"\nGenerating visualizations...")
    save_visualizations(vis_data, sampling_config, channels, save_dir)
    plot_ar_error_curves(results, save_dir)

    # 频谱分析
    if args.spectral_analysis:
        print(f"\nRunning spectral/frequency error analysis...")
        npz_path = save_predictions_for_spectral_analysis(vis_data, save_dir, channels)
        if npz_path:
            spectral_output_dir = save_dir / 'spectral_analysis'
            success = run_spectral_analysis(npz_path, spectral_output_dir, channels)
            if success:
                print(f"  Spectral analysis completed. Results in: {spectral_output_dir}")

    print(f"\n{'='*80}")
    print("Test completed!")
    print(f"{'='*80}")
    print(f"Results saved to: {save_dir}")

    return results


if __name__ == "__main__":
    main()
