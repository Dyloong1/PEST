"""
JHU Swin模型3轮AR评估脚本

评估指标：
- Per-channel RMSE (normalized & physical units)
- Per-channel SSIM (if available)
- Divergence error

Usage:
    python evaluate_swin_jhu_ar.py \
        --checkpoint Checkpoints/swin_jhu_spectral_patch2x2x2/best_model.pt \
        --config spectral_small_patch \
        --num_ar_rounds 3
"""

import sys
import os

# Add paths
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, os.path.join(CODE_DIR, 'Model'))
sys.path.insert(0, os.path.join(CODE_DIR, 'Dataset'))

import torch
import numpy as np
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

from Dataset.jhu_dataset import SparseJHUDataset
from Model.configs.swin_jhu_config import get_jhu_swin_config


def compute_divergence_3d(u, v, w, dx=1.0, dy=1.0, dz=1.0):
    """计算3D散度"""
    # u: (B, T, Z, H, W), v: (B, T, Z, H, W), w: (B, T, Z, H, W)
    # 使用中心差分
    du_dx = (torch.roll(u, -1, dims=-1) - torch.roll(u, 1, dims=-1)) / (2 * dx)
    dv_dy = (torch.roll(v, -1, dims=-2) - torch.roll(v, 1, dims=-2)) / (2 * dy)
    dw_dz = (torch.roll(w, -1, dims=-3) - torch.roll(w, 1, dims=-3)) / (2 * dz)

    div = du_dx + dv_dy + dw_dz
    return torch.abs(div).mean().item()


def evaluate_ar_rounds(model, dataset, num_rounds, device, stats, channels):
    """评估多轮AR"""

    results = {
        'per_round': {},
        'num_ar_rounds': num_rounds,
        'channels': channels
    }

    model.eval()

    # 获取测试样本
    test_samples = []
    for i in range(len(dataset)):
        sample = dataset[i]
        test_samples.append(sample)

    print(f"\nTotal test samples: {len(test_samples)}")

    # 对每一轮AR进行评估
    for round_num in range(1, num_rounds + 1):
        print(f"\n{'='*60}")
        print(f"Evaluating AR Round {round_num}")
        print(f"{'='*60}")

        round_metrics = defaultdict(list)
        valid_samples = 0

        # 需要确保有连续的样本对
        max_sample_idx = len(test_samples) - (round_num - 1)

        for sample_idx in tqdm(range(max_sample_idx), desc=f'Round {round_num}'):
            try:
                # 获取当前样本和后续样本（用于多轮AR）
                current_samples = [test_samples[sample_idx + i] for i in range(round_num)]

                # 第一轮：使用GT输入
                input_dense = current_samples[0]['input_dense'].unsqueeze(0).to(device)
                gt_output = current_samples[0]['output_dense'].unsqueeze(0).to(device)
                output_sparse = current_samples[0]['output_sparse'].unsqueeze(0).to(device)

                sparse_z_indices = torch.arange(0, 128, 8, dtype=torch.long)  # [0, 8, 16, ..., 120]

                # 执行AR轮次
                for r in range(round_num):
                    with torch.no_grad():
                        model_input = {
                            'input_dense': input_dense,
                            'output_sparse_z': output_sparse,
                            'sparse_indices': {'z': sparse_z_indices}
                        }

                        model_output = model(model_input)
                        if isinstance(model_output, dict):
                            pred_dense = model_output['output_full']
                        else:
                            pred_dense = model_output

                    # 如果不是最后一轮，准备下一轮的输入
                    if r < round_num - 1:
                        input_dense = pred_dense
                        output_sparse = current_samples[r+1]['output_sparse'].unsqueeze(0).to(device)
                        gt_output = current_samples[r+1]['output_dense'].unsqueeze(0).to(device)

                # 计算最后一轮的误差
                pred_np = pred_dense.cpu().numpy()
                gt_np = gt_output.cpu().numpy()

                # Per-channel MSE (normalized)
                mse_norm = np.mean((pred_np - gt_np) ** 2, axis=(0, 1, 3, 4, 5))  # [C,]
                round_metrics['mse_normalized'].append(mse_norm)

                # Per-channel RMSE (normalized)
                rmse_norm = np.sqrt(mse_norm)
                round_metrics['rmse_normalized'].append(rmse_norm)

                # 反归一化到物理空间
                pred_phys = pred_np.copy()
                gt_phys = gt_np.copy()

                for c_idx, ch_name in enumerate(channels):
                    mean_val = stats[ch_name]['mean']
                    std_val = stats[ch_name]['std']
                    pred_phys[:, :, c_idx] = pred_np[:, :, c_idx] * std_val + mean_val
                    gt_phys[:, :, c_idx] = gt_np[:, :, c_idx] * std_val + mean_val

                # Per-channel MSE & RMSE (physical)
                mse_phys = np.mean((pred_phys - gt_phys) ** 2, axis=(0, 1, 3, 4, 5))  # [C,]
                rmse_phys = np.sqrt(mse_phys)

                round_metrics['mse_physical'].append(mse_phys)
                round_metrics['rmse_physical'].append(rmse_phys)

                # 计算散度误差
                pred_tensor = torch.from_numpy(pred_phys).to(device)
                u_pred = pred_tensor[:, :, 0]  # (B, T, Z, H, W)
                v_pred = pred_tensor[:, :, 1]
                w_pred = pred_tensor[:, :, 2]

                div_error = compute_divergence_3d(u_pred, v_pred, w_pred)
                round_metrics['divergence_error'].append(div_error)

                valid_samples += 1

            except Exception as e:
                print(f"Warning: Sample {sample_idx} failed: {e}")
                continue

        # 聚合当前轮的指标
        if valid_samples > 0:
            results['per_round'][str(round_num)] = {
                'mse_normalized': np.mean(round_metrics['mse_normalized'], axis=0).tolist(),
                'mse_physical': np.mean(round_metrics['mse_physical'], axis=0).tolist(),
                'rmse_normalized': np.mean(round_metrics['rmse_normalized'], axis=0).tolist(),
                'rmse_physical': np.mean(round_metrics['rmse_physical'], axis=0).tolist(),
                'divergence_error_mean': np.mean(round_metrics['divergence_error']),
                'divergence_error_std': np.std(round_metrics['divergence_error']),
                'num_samples': valid_samples
            }

            # 打印当前轮结果
            print(f"\nRound {round_num} Results ({valid_samples} samples):")
            print(f"  RMSE (Physical): {results['per_round'][str(round_num)]['rmse_physical']}")
            print(f"  RMSE (Normalized): {results['per_round'][str(round_num)]['rmse_normalized']}")
            print(f"  Divergence: {results['per_round'][str(round_num)]['divergence_error_mean']:.6f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='JHU Swin AR Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--config', type=str, required=True, help='JHU config name')
    parser.add_argument('--data_dir', type=str, required=True, help='JHU data directory')
    parser.add_argument('--split_config', type=str, required=True, help='Data split JSON')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--num_ar_rounds', type=int, default=3, help='Number of AR rounds')
    parser.add_argument('--sparse_z_interval', type=int, default=8, help='Sparse z interval')
    parser.add_argument('--model_name', type=str, default='model', help='Model name for output')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load config
    config = get_jhu_swin_config(args.config)
    channels = ['u', 'v', 'w', 'p']

    # Load dataset
    print(f"\nLoading dataset from {args.data_dir}")
    dataset = SparseJHUDataset(
        data_dir=args.data_dir,
        split='test',
        split_config_path=args.split_config,
        indicators=channels,
        sparse_z_interval=args.sparse_z_interval,
        normalize=True
    )

    print(f"Test samples: {len(dataset)}")

    # Load model
    print(f"\nLoading model from {args.checkpoint}")
    from Model.swin_transformer_model_patched import SwinTransformerPhysics

    model = SwinTransformerPhysics(config).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print("Model loaded successfully")

    # Run evaluation
    results = evaluate_ar_rounds(
        model, dataset, args.num_ar_rounds, device,
        dataset.norm_stats, channels
    )

    # Add metadata
    results['model_type'] = 'swin'
    results['config'] = args.config
    results['checkpoint'] = args.checkpoint
    results['sparse_z_interval'] = args.sparse_z_interval
    results['dense_z_size'] = 128
    results['num_sparse_layers'] = 128 // args.sparse_z_interval
    results['num_missing_layers'] = 128 - (128 // args.sparse_z_interval)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / 'ar_evaluation_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to {output_file}")

    # Print summary
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Model: {args.config}")
    print(f"AR Rounds: {args.num_ar_rounds}")
    print(f"\nPer-Channel RMSE (Physical Units):")
    print(f"{'Round':<10} {'u':<12} {'v':<12} {'w':<12} {'p':<12}")
    print("-" * 60)

    for round_num in range(1, args.num_ar_rounds + 1):
        if str(round_num) in results['per_round']:
            rmse = results['per_round'][str(round_num)]['rmse_physical']
            print(f"{round_num:<10} {rmse[0]:<12.6f} {rmse[1]:<12.6f} {rmse[2]:<12.6f} {rmse[3]:<12.6f}")

    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
