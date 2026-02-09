"""数据损失函数 (修复版本)"""
import torch
import torch.nn as nn


def mse_loss(pred, target, missing_indices=None, **kwargs):
    """
    MSE损失 - 支持masked计算
    
    Args:
        pred: (B, T, C, Z, H, W) - 预测
        target: (B, T, C, Z, H, W) - 真值
        missing_indices: (B, n_missing) - 需要预测的z层索引
            如果为None，则对所有层计算loss
    
    Returns:
        loss: scalar
    """
    # FIXED: 如果提供了missing_indices，只对missing层计算loss
    if missing_indices is not None:
        return mse_loss_masked(pred, target, missing_indices)
    else:
        return nn.functional.mse_loss(pred, target)


def mse_loss_masked(pred, target, missing_indices):
    """
    只对missing层计算MSE损失
    
    避免对sparse层(直接复制的GT)计算loss，防止loss被稀释
    
    Args:
        pred: (B, T, C, Z, H, W)
        target: (B, T, C, Z, H, W)
        missing_indices: (B, n_missing)
    """
    B = pred.shape[0]
    losses = []
    
    for b in range(B):
        # 提取当前样本的missing层
        missing_z = missing_indices[b]  # (n_missing,)
        
        # 提取pred和target在missing层的数据
        pred_missing = pred[b, :, :, missing_z, :, :]    # (T, C, n_missing, H, W)
        target_missing = target[b, :, :, missing_z, :, :]  # (T, C, n_missing, H, W)
        
        # 计算这个样本的loss
        loss_b = nn.functional.mse_loss(pred_missing, target_missing)
        losses.append(loss_b)
    
    # 对batch取平均
    return torch.stack(losses).mean()


def l1_loss(pred, target, missing_indices=None, **kwargs):
    """L1损失 - 支持masked计算"""
    if missing_indices is not None:
        return l1_loss_masked(pred, target, missing_indices)
    else:
        return nn.functional.l1_loss(pred, target)


def l1_loss_masked(pred, target, missing_indices):
    """只对missing层计算L1损失"""
    B = pred.shape[0]
    losses = []
    
    for b in range(B):
        missing_z = missing_indices[b]
        pred_missing = pred[b, :, :, missing_z, :, :]
        target_missing = target[b, :, :, missing_z, :, :]
        loss_b = nn.functional.l1_loss(pred_missing, target_missing)
        losses.append(loss_b)
    
    return torch.stack(losses).mean()


def huber_loss(pred, target, missing_indices=None, delta=1.0, **kwargs):
    """Huber损失 - 支持masked计算"""
    if missing_indices is not None:
        return huber_loss_masked(pred, target, missing_indices, delta)
    else:
        return nn.functional.smooth_l1_loss(pred, target, beta=delta)


def huber_loss_masked(pred, target, missing_indices, delta=1.0):
    """只对missing层计算Huber损失"""
    B = pred.shape[0]
    losses = []
    
    for b in range(B):
        missing_z = missing_indices[b]
        pred_missing = pred[b, :, :, missing_z, :, :]
        target_missing = target[b, :, :, missing_z, :, :]
        loss_b = nn.functional.smooth_l1_loss(pred_missing, target_missing, beta=delta)
        losses.append(loss_b)
    
    return torch.stack(losses).mean()


def relative_l2_loss(pred, target, missing_indices=None, eps=1e-8, **kwargs):
    """相对L2损失 - 支持masked计算"""
    if missing_indices is not None:
        return relative_l2_loss_masked(pred, target, missing_indices, eps)
    else:
        diff = pred - target
        rel_error = torch.norm(diff.reshape(diff.shape[0], -1), dim=1) / \
                    (torch.norm(target.reshape(target.shape[0], -1), dim=1) + eps)
        return rel_error.mean()


def relative_l2_loss_masked(pred, target, missing_indices, eps=1e-8):
    """只对missing层计算相对L2损失"""
    B = pred.shape[0]
    losses = []
    
    for b in range(B):
        missing_z = missing_indices[b]
        pred_missing = pred[b, :, :, missing_z, :, :]
        target_missing = target[b, :, :, missing_z, :, :]
        
        diff = pred_missing - target_missing
        rel_error = torch.norm(diff.reshape(-1)) / (torch.norm(target_missing.reshape(-1)) + eps)
        losses.append(rel_error)
    
    return torch.stack(losses).mean()


def get_data_loss(loss_type='mse'):
    """
    获取数据损失函数
    
    Args:
        loss_type: 'mse', 'l1', 'huber', 'relative_l2'
    
    Returns:
        loss_fn: 损失函数
    """
    losses = {
        'mse': mse_loss,
        'l1': l1_loss,
        'huber': huber_loss,
        'relative_l2': relative_l2_loss
    }
    if loss_type not in losses:
        raise ValueError(f"Unknown data loss type: {loss_type}")
    return losses[loss_type]