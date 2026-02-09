#!/usr/bin/env python3
"""
Training Script for Swin-DiT with PDE-Refiner Configuration

This combines:
- Swin Transformer backbone
- PDE-Refiner's diffusion setup (K=3 steps, exponential sigma schedule)
- Sparse observation interpolation as prior

Usage:
    python train_swin_dit_pde_refiner.py --config default --epochs 150
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
import argparse
import json
from datetime import datetime
from tqdm import tqdm
import csv
import sys

# Add paths
CODE_DIR = Path(__file__).parent
sys.path.insert(0, str(CODE_DIR))

from Model.swin_dit_pde_refiner import SwinDiTPDERefiner, get_swin_dit_pde_refiner_config
from Dataset.jhu_dataset import SparseJHUDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Train Swin-DiT with PDE-Refiner Config')

    # Model
    parser.add_argument('--config', type=str, default='giant',
                        choices=['default', 'giant'],
                        help='Model size: default(15M), giant(350M for 32GB GPU)')

    # Data
    parser.add_argument('--data_source', type=str, default='jhu',
                        choices=['jhu'], help='Data source (only JHU supported)')
    parser.add_argument('--data_dir', type=str,
                        default='/media/ydai17/T7 Shield/JHU_data/DNS128',
                        help='Data directory')
    parser.add_argument('--split_config', type=str, default=None,
                        help='Split configuration file')
    parser.add_argument('--sparse_z_interval', type=int, default=8,
                        help='Sparse z interval')

    # Training
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--use_amp', action='store_true', help='Use mixed precision training')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile for speedup (PyTorch 2.0+)')
    parser.add_argument('--grad_checkpointing', action='store_true', help='Use gradient checkpointing to save memory')

    # Physics loss (only applied at low noise stages k >= threshold)
    parser.add_argument('--use_physics_loss', action='store_true',
                        help='Enable physics loss at low noise stages')
    parser.add_argument('--div_weight', type=float, default=1e-4,
                        help='Divergence loss weight (default: 1e-4)')
    parser.add_argument('--ns_weight', type=float, default=1e-5,
                        help='NS residual loss weight (default: 1e-5)')
    parser.add_argument('--physics_threshold', type=int, default=2,
                        help='Apply physics loss when k >= threshold (default: 2)')
    parser.add_argument('--physics_warmup_epochs', type=int, default=50,
                        help='Number of warmup epochs before enabling physics loss (default: 50)')
    parser.add_argument('--physics_rampup_epochs', type=int, default=20,
                        help='Number of epochs to ramp up physics loss weight (default: 20)')

    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Checkpoint directory')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')

    # Other
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Data loader workers (default=0 for JHU large data to avoid memory issues)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_periodic', action='store_true',
                        help='Save periodic checkpoints (every 50 epochs)')

    args = parser.parse_args()

    # Set default paths
    if args.split_config is None:
        args.split_config = str(CODE_DIR / 'Dataset' / 'jhu_data_splits_ar.json')

    if args.checkpoint_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.checkpoint_dir = str(CODE_DIR / 'Checkpoints' / f'swin_dit_pde_refiner_{args.config}_{timestamp}')

    return args


def load_norm_stats(data_source='jhu'):
    """Load normalization statistics."""
    if data_source == 'jhu':
        stats_path = CODE_DIR / 'Data' / 'jhu_normalization_stats.json'
    else:
        stats_path = CODE_DIR / 'Data' / 'normalization_stats.json'

    if stats_path.exists():
        with open(stats_path) as f:
            raw_stats = json.load(f)

        indicators = raw_stats.get('indicators', ['u', 'v', 'w', 'p'])
        norm_stats = {}
        for i, ind in enumerate(indicators):
            norm_stats[ind] = {
                'mean': raw_stats['mean'][i],
                'std': raw_stats['std'][i]
            }
        return norm_stats
    else:
        print(f"WARNING: Normalization stats not found at {stats_path}")
        return None


def create_dataloaders(args, norm_stats):
    """Create train and validation data loaders."""
    train_dataset = SparseJHUDataset(
        data_dir=args.data_dir,
        split='train',
        split_config_path=args.split_config,
        indicators=['u', 'v', 'w', 'p'],
        sparse_z_interval=args.sparse_z_interval,
        normalize=True,
        norm_stats=norm_stats,
        dense_z_size=128
    )

    val_dataset = SparseJHUDataset(
        data_dir=args.data_dir,
        split='val',
        split_config_path=args.split_config,
        indicators=['u', 'v', 'w', 'p'],
        sparse_z_interval=args.sparse_z_interval,
        normalize=True,
        norm_stats=norm_stats,
        dense_z_size=128
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


def train_epoch(model, loader, optimizer, scaler, device, use_amp=False,
                use_physics_loss=False, physics_config=None):
    """Train for one epoch using diffusion training loss."""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(loader, desc='Training')
    for batch in pbar:
        input_dense = batch['input_dense'].to(device)
        output_sparse = batch['output_sparse'].to(device)
        output_dense = batch['output_dense'].to(device)
        sparse_z_indices = batch['sparse_z_indices'].to(device)

        optimizer.zero_grad()

        if use_amp:
            with autocast('cuda'):
                loss = model.get_training_loss(
                    input_dense, output_dense, output_sparse, sparse_z_indices,
                    use_physics_loss=use_physics_loss, physics_config=physics_config
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = model.get_training_loss(
                input_dense, output_dense, output_sparse, sparse_z_indices,
                use_physics_loss=use_physics_loss, physics_config=physics_config
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})

    return total_loss / num_batches


@torch.no_grad()
def validate(model, loader, device):
    """Validate by running full inference (K-step denoising)."""
    model.eval()
    total_loss = 0
    num_batches = 0

    for batch in tqdm(loader, desc='Validation'):
        input_dense = batch['input_dense'].to(device)
        output_sparse = batch['output_sparse'].to(device)
        output_dense = batch['output_dense'].to(device)
        sparse_z_indices = batch['sparse_z_indices'].to(device)
        missing_z_indices = batch['missing_z_indices'].to(device)

        # Run full inference (K-step denoising)
        pred = model(input_dense, output_sparse, sparse_z_indices, missing_z_indices)

        # Compute loss on missing layers only
        B = input_dense.shape[0]
        loss = 0
        for b in range(B):
            missing_idx = missing_z_indices[b]
            pred_missing = pred[b, :, :, missing_idx, :, :]
            gt_missing = output_dense[b, :, :, missing_idx, :, :]
            loss = loss + nn.functional.mse_loss(pred_missing, gt_missing)
        loss = loss / B

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def main():
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Enable cuDNN benchmark for faster training (when input sizes are fixed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for faster matmul
        torch.backends.cudnn.allow_tf32 = True

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    config = get_swin_dit_pde_refiner_config(args.config)
    print(f"\nModel: Swin-DiT + PDE-Refiner ({args.config})")
    print(f"  Refine steps: {config.REFINER_STEPS}")
    print(f"  Min noise std: {config.MIN_NOISE_STD}")
    print(f"  Embed dim: {config.EMBED_DIM}")
    print(f"  Depths: {config.DEPTHS}")

    # Load normalization stats
    norm_stats = load_norm_stats(args.data_source)

    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader = create_dataloaders(args, norm_stats)
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")

    # Create model
    print("\nCreating model...")
    model = SwinDiTPDERefiner(config)
    model = model.to(device)

    # Move scheduler to device
    model.scheduler.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Gradient checkpointing (save memory at cost of some speed)
    if args.grad_checkpointing:
        print("Enabling gradient checkpointing...")
        model.gradient_checkpointing_enable()

    # torch.compile for speedup (PyTorch 2.0+)
    if args.compile:
        print("Compiling model with torch.compile...")
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print("  Model compiled successfully!")
        except Exception as e:
            print(f"  Warning: torch.compile failed: {e}")
            print("  Continuing without compilation...")

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # AMP scaler
    scaler = GradScaler('cuda', enabled=args.use_amp)

    # Resume if specified
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume:
        print(f"\nResuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        # Load AMP scaler state if available
        if args.use_amp and ckpt.get('scaler_state_dict') is not None:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"  Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.6f}")

    # Training log (append if resuming, otherwise create new)
    log_path = checkpoint_dir / 'training_log.csv'
    if args.resume and log_path.exists():
        log_file = open(log_path, 'a', newline='')
        log_writer = csv.writer(log_file)
    else:
        log_file = open(log_path, 'w', newline='')
        log_writer = csv.writer(log_file)
        log_writer.writerow(['epoch', 'train_loss', 'val_loss', 'best_val_loss', 'lr'])

    # Save config
    config_path = checkpoint_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump({
            'model': 'swin_dit_pde_refiner',
            'config': args.config,
            'args': vars(args),
            'total_params': total_params,
            'refine_steps': config.REFINER_STEPS,
            'min_noise_std': config.MIN_NOISE_STD,
            'embed_dim': config.EMBED_DIM,
            'depths': config.DEPTHS,
            'use_physics_loss': args.use_physics_loss,
            'div_weight': args.div_weight if args.use_physics_loss else None,
            'ns_weight': args.ns_weight if args.use_physics_loss else None,
            'physics_threshold': args.physics_threshold if args.use_physics_loss else None,
            'physics_warmup_epochs': args.physics_warmup_epochs if args.use_physics_loss else None,
            'physics_rampup_epochs': args.physics_rampup_epochs if args.use_physics_loss else None
        }, f, indent=2)

    # Physics loss configuration (with warmup and ramp-up)
    physics_config_base = None
    if args.use_physics_loss:
        # Extract norm_mean and norm_std as lists for physics loss
        if norm_stats is not None:
            norm_mean = [norm_stats[ind]['mean'] for ind in ['u', 'v', 'w', 'p']]
            norm_std = [norm_stats[ind]['std'] for ind in ['u', 'v', 'w', 'p']]
        else:
            norm_mean = None
            norm_std = None

        physics_config_base = {
            'div_weight': args.div_weight,
            'ns_weight': args.ns_weight,
            'physics_threshold': args.physics_threshold,
            'norm_mean': norm_mean,
            'norm_std': norm_std,
        }
        print(f"\nPhysics loss enabled (with warmup):")
        print(f"  Warmup epochs: {args.physics_warmup_epochs}")
        print(f"  Ramp-up epochs: {args.physics_rampup_epochs}")
        print(f"  Final div weight: {args.div_weight}")
        print(f"  Final NS weight: {args.ns_weight}")
        print(f"  Threshold: k >= {args.physics_threshold}")

    # Training loop
    print(f"\nStarting training from epoch {start_epoch}...")
    print(f"Using AMP: {args.use_amp}")

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Compute physics loss config with warmup and ramp-up
        use_physics_this_epoch = False
        physics_config = None

        if args.use_physics_loss and physics_config_base is not None:
            if epoch >= args.physics_warmup_epochs:
                use_physics_this_epoch = True

                # Ramp-up factor: 0 -> 1 over rampup_epochs
                rampup_progress = min(1.0, (epoch - args.physics_warmup_epochs) / max(1, args.physics_rampup_epochs))

                physics_config = {
                    'div_weight': physics_config_base['div_weight'] * rampup_progress,
                    'ns_weight': physics_config_base['ns_weight'] * rampup_progress,
                    'physics_threshold': physics_config_base['physics_threshold'],
                    'norm_mean': physics_config_base['norm_mean'],
                    'norm_std': physics_config_base['norm_std'],
                }

                if rampup_progress < 1.0:
                    print(f"  Physics loss ramp-up: {rampup_progress*100:.1f}% "
                          f"(div={physics_config['div_weight']:.2e}, ns={physics_config['ns_weight']:.2e})")
                else:
                    print(f"  Physics loss: FULL "
                          f"(div={physics_config['div_weight']:.2e}, ns={physics_config['ns_weight']:.2e})")
            else:
                print(f"  Physics loss: WARMUP ({epoch + 1}/{args.physics_warmup_epochs})")

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler, device,
            use_amp=args.use_amp,
            use_physics_loss=use_physics_this_epoch,
            physics_config=physics_config
        )

        # Validate
        val_loss = validate(model, val_loader, device)

        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Log
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
              f"Best Val Loss: {best_val_loss:.6f}, LR: {current_lr:.2e}")

        log_writer.writerow([epoch + 1, train_loss, val_loss, best_val_loss, current_lr])
        log_file.flush()

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict() if args.use_amp else None,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
        }

        # Save latest
        torch.save(checkpoint, checkpoint_dir / 'latest_model.pt')

        # Save best
        if is_best:
            torch.save(checkpoint, checkpoint_dir / 'best_model.pt')
            print("  -> New best model saved!")

        # Save periodic checkpoint
        if args.save_periodic and (epoch + 1) % 50 == 0:
            torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch_{epoch + 1}.pt')

    log_file.close()
    print(f"\nTraining completed! Best validation loss: {best_val_loss:.6f}")
    print(f"Results saved to: {checkpoint_dir}")


if __name__ == '__main__':
    main()
