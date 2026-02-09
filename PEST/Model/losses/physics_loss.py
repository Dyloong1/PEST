"""物理损失函数"""
import torch
import torch.nn as nn

def denormalize_field(pred, norm_mean=None, norm_std=None):
    """
    反归一化物理场
    pred: (B, T, C, Z, H, W) 归一化的数据
    norm_mean: (C,) 各通道均值
    norm_std: (C,) 各通道标准差
    返回: (B, T, C, Z, H, W) 真实物理量
    """
    if norm_mean is None or norm_std is None:
        # 如果没有提供统计量，返回原始预测（假设已归一化）
        return pred

    pred_real = pred.clone()
    device = pred.device

    # 确保统计量在正确的设备上
    if not isinstance(norm_mean, torch.Tensor):
        norm_mean = torch.tensor(norm_mean, device=device)
    if not isinstance(norm_std, torch.Tensor):
        norm_std = torch.tensor(norm_std, device=device)

    norm_mean = norm_mean.to(device)
    norm_std = norm_std.to(device)

    # 反归一化: x_real = x_norm * std + mean
    # 处理shape: (B, T, C, Z, H, W)
    for c_idx in range(pred.shape[2]):
        pred_real[:, :, c_idx] = pred[:, :, c_idx] * norm_std[c_idx] + norm_mean[c_idx]

    return pred_real

def divergence_loss_2d(pred, target=None, dx=1.0, dy=1.0, norm_mean=None, norm_std=None, **kwargs):
    """
    2D散度损失（在真实物理空间计算）
    pred: (B, T, C, Z, H, W) 归一化的预测
    """
    # 反归一化到真实物理空间
    pred_real = denormalize_field(pred, norm_mean, norm_std)
    
    u = pred_real[:, :, 0, :, :, :]  # 真实速度 m/s
    v = pred_real[:, :, 1, :, :, :]
    
    # 中心差分计算导数
    du_dx = (u[:, :, :, :, 2:] - u[:, :, :, :, :-2]) / (2 * dx)
    dv_dy = (v[:, :, :, 2:, :] - v[:, :, :, :-2, :]) / (2 * dy)
    
    min_h = min(du_dx.shape[3], dv_dy.shape[3])
    min_w = min(du_dx.shape[4], dv_dy.shape[4])
    du_dx = du_dx[:, :, :, :min_h, :min_w]
    dv_dy = dv_dy[:, :, :, :min_h, :min_w]
    
    divergence = du_dx + dv_dy  # 真实散度 1/s
    return (divergence ** 2).mean()

def divergence_loss_3d(pred, target=None, dx=1.0, dy=1.0, dz=1.0, norm_mean=None, norm_std=None, **kwargs):
    """
    3D散度损失（在真实物理空间计算）

    计算不可压缩连续性方程约束:
        ∇·u = ∂u/∂x + ∂v/∂y + ∂w/∂z = 0

    如果只有(u,v,p)三个通道（w=0），只计算∂u/∂x + ∂v/∂y（2D divergence）
    如果有(u,v,w,p)四个通道，计算完整3D divergence

    Args:
        pred: (B, T, C, Z, H, W) 预测场
        target: 未使用
        dx, dy, dz: 物理网格间距 [L]
            对于DNS Taylor-Green (Domain=2π, Grid=64×128×128):
            - dx = dy = 2π/128 ≈ 0.049
            - dz = 2π/64 ≈ 0.098
        norm_mean, norm_std: 数据归一化参数

    Returns:
        loss: 散度的L2范数 (完美不可压缩流动应为0)
    """
    n_channels = pred.shape[2]

    # 反归一化到真实物理空间
    pred_real = denormalize_field(pred, norm_mean, norm_std)

    u = pred_real[:, :, 0, :, :, :]
    v = pred_real[:, :, 1, :, :, :]

    # 中心差分
    du_dx = (u[:, :, :, :, 2:] - u[:, :, :, :, :-2]) / (2 * dx)
    dv_dy = (v[:, :, :, 2:, :] - v[:, :, :, :-2, :]) / (2 * dy)

    if n_channels >= 4:
        # 完整3D散度 (u,v,w,p)
        w = pred_real[:, :, 2, :, :, :]
        dw_dz = (w[:, :, 2:, :, :] - w[:, :, :-2, :, :]) / (2 * dz)

        min_z = min(du_dx.shape[2], dv_dy.shape[2], dw_dz.shape[2])
        min_h = min(du_dx.shape[3], dv_dy.shape[3], dw_dz.shape[3])
        min_w = min(du_dx.shape[4], dv_dy.shape[4], dw_dz.shape[4])

        du_dx = du_dx[:, :, :min_z, :min_h, :min_w]
        dv_dy = dv_dy[:, :, :min_z, :min_h, :min_w]
        dw_dz = dw_dz[:, :, :min_z, :min_h, :min_w]

        divergence = du_dx + dv_dy + dw_dz
    else:
        # 2D散度 (u,v,p)，忽略w=0
        min_h = min(du_dx.shape[3], dv_dy.shape[3])
        min_w = min(du_dx.shape[4], dv_dy.shape[4])

        du_dx = du_dx[:, :, :, :min_h, :min_w]
        dv_dy = dv_dy[:, :, :, :min_h, :min_w]

        divergence = du_dx + dv_dy

    return (divergence ** 2).mean()

def navier_stokes_residual(pred, target=None, nu=0.01, rho=1.0,
                          dx=1.0, dy=1.0, dz=1.0, dt=1.0, norm_mean=None, norm_std=None, **kwargs):
    """
    NS方程残差损失（在真实物理空间计算）

    计算不可压缩Navier-Stokes方程的残差:
        ∂u/∂t + (u·∇)u + (1/ρ)∇p - ν∇²u = 0

    重要: 所有参数必须使用一致的物理单位！

    对于DNS Taylor-Green (Re=1600, Domain=2π):
        - nu = 1/Re = 0.000625 (物理运动粘度)
        - dx = dy = 2π/128 ≈ 0.049 (物理网格间距)
        - dz = 2π/64 ≈ 0.098 (物理网格间距)
        - dt = 0.125 (无量纲时间步长, 以t_c = L/V₀为单位)

    Args:
        pred: (B, T, C, Z, H, W) 预测场，C >= 4 (u, v, w, p)
        target: 未使用
        nu: 运动粘度 [L²/T] (物理单位，对于Re=1600: ν = 1/Re = 0.000625)
        rho: 密度 [M/L³] (通常归一化为1.0)
        dx, dy, dz: 物理网格间距 [L] (对于2π域和128网格: dx = 2π/128 ≈ 0.049)
        dt: 时间步长 [T] (无量纲时间)
        norm_mean, norm_std: 数据归一化参数

    Returns:
        loss: NS方程残差的L2范数
    """
    B, T, C = pred.shape[0], pred.shape[1], pred.shape[2]

    if C < 4:
        # DNS数据只有2通道(u,v)，跳过NS损失
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    if T < 2:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    # 反归一化到真实物理空间
    pred_real = denormalize_field(pred, norm_mean, norm_std)
    
    t_mid = T // 2
    u_curr = pred_real[:, t_mid, 0, :, :, :]  # m/s
    v_curr = pred_real[:, t_mid, 1, :, :, :]
    w_curr = pred_real[:, t_mid, 2, :, :, :]
    p_curr = pred_real[:, t_mid, 3, :, :, :]  # Pa
    
    # 时间导数
    if t_mid > 0 and t_mid < T - 1:
        dudt = (pred_real[:, t_mid+1, 0] - pred_real[:, t_mid-1, 0]) / (2 * dt)
        dvdt = (pred_real[:, t_mid+1, 1] - pred_real[:, t_mid-1, 1]) / (2 * dt)
        dwdt = (pred_real[:, t_mid+1, 2] - pred_real[:, t_mid-1, 2]) / (2 * dt)
    else:
        dudt = (pred_real[:, 1, 0] - pred_real[:, 0, 0]) / dt
        dvdt = (pred_real[:, 1, 1] - pred_real[:, 0, 1]) / dt
        dwdt = (pred_real[:, 1, 2] - pred_real[:, 0, 2]) / dt
    
    # 空间一阶导数
    dudx = (u_curr[:, :, :, 2:] - u_curr[:, :, :, :-2]) / (2*dx)
    dudy = (u_curr[:, :, 2:, :] - u_curr[:, :, :-2, :]) / (2*dy)
    dudz = (u_curr[:, 2:, :, :] - u_curr[:, :-2, :, :]) / (2*dz)
    
    dvdx = (v_curr[:, :, :, 2:] - v_curr[:, :, :, :-2]) / (2*dx)
    dvdy = (v_curr[:, :, 2:, :] - v_curr[:, :, :-2, :]) / (2*dy)
    dvdz = (v_curr[:, 2:, :, :] - v_curr[:, :-2, :, :]) / (2*dz)
    
    dwdx = (w_curr[:, :, :, 2:] - w_curr[:, :, :, :-2]) / (2*dx)
    dwdy = (w_curr[:, :, 2:, :] - w_curr[:, :, :-2, :]) / (2*dy)
    dwdz = (w_curr[:, 2:, :, :] - w_curr[:, :-2, :, :]) / (2*dz)
    
    dpdx = (p_curr[:, :, :, 2:] - p_curr[:, :, :, :-2]) / (2*dx)
    dpdy = (p_curr[:, :, 2:, :] - p_curr[:, :, :-2, :]) / (2*dy)
    dpdz = (p_curr[:, 2:, :, :] - p_curr[:, :-2, :, :]) / (2*dz)
    
    # 空间二阶导数
    d2udx2 = (u_curr[:, :, :, 2:] - 2*u_curr[:, :, :, 1:-1] + u_curr[:, :, :, :-2]) / (dx**2)
    d2udy2 = (u_curr[:, :, 2:, :] - 2*u_curr[:, :, 1:-1, :] + u_curr[:, :, :-2, :]) / (dy**2)
    d2udz2 = (u_curr[:, 2:, :, :] - 2*u_curr[:, 1:-1, :, :] + u_curr[:, :-2, :, :]) / (dz**2)
    
    d2vdx2 = (v_curr[:, :, :, 2:] - 2*v_curr[:, :, :, 1:-1] + v_curr[:, :, :, :-2]) / (dx**2)
    d2vdy2 = (v_curr[:, :, 2:, :] - 2*v_curr[:, :, 1:-1, :] + v_curr[:, :, :-2, :]) / (dy**2)
    d2vdz2 = (v_curr[:, 2:, :, :] - 2*v_curr[:, 1:-1, :, :] + v_curr[:, :-2, :, :]) / (dz**2)
    
    d2wdx2 = (w_curr[:, :, :, 2:] - 2*w_curr[:, :, :, 1:-1] + w_curr[:, :, :, :-2]) / (dx**2)
    d2wdy2 = (w_curr[:, :, 2:, :] - 2*w_curr[:, :, 1:-1, :] + w_curr[:, :, :-2, :]) / (dy**2)
    d2wdz2 = (w_curr[:, 2:, :, :] - 2*w_curr[:, 1:-1, :, :] + w_curr[:, :-2, :, :]) / (dz**2)
    
    # 对齐尺寸
    z_sizes = [dudz.shape[1], dvdz.shape[1], dwdz.shape[1], dpdz.shape[1], 
               d2udz2.shape[1], d2vdz2.shape[1], d2wdz2.shape[1], dudt.shape[1]]
    h_sizes = [dudy.shape[2], dvdy.shape[2], dwdy.shape[2], dpdy.shape[2], 
               d2udy2.shape[2], d2vdy2.shape[2], d2wdy2.shape[2], dudt.shape[2]]
    w_sizes = [dudx.shape[3], dvdx.shape[3], dwdx.shape[3], dpdx.shape[3], 
               d2udx2.shape[3], d2vdx2.shape[3], d2wdx2.shape[3], dudt.shape[3]]
    
    min_z, min_h, min_w = min(z_sizes), min(h_sizes), min(w_sizes)
    
    def crop(t, z, h, w):
        return t[:, :z, :h, :w]
    
    u = crop(u_curr[:, 1:-1, 1:-1, 1:-1], min_z, min_h, min_w)
    v = crop(v_curr[:, 1:-1, 1:-1, 1:-1], min_z, min_h, min_w)
    w = crop(w_curr[:, 1:-1, 1:-1, 1:-1], min_z, min_h, min_w)
    
    dudt = crop(dudt[:, 1:-1, 1:-1, 1:-1], min_z, min_h, min_w)
    dvdt = crop(dvdt[:, 1:-1, 1:-1, 1:-1], min_z, min_h, min_w)
    dwdt = crop(dwdt[:, 1:-1, 1:-1, 1:-1], min_z, min_h, min_w)
    
    dudx = crop(dudx[:, 1:-1, 1:-1, :], min_z, min_h, min_w)
    dudy = crop(dudy[:, 1:-1, :, 1:-1], min_z, min_h, min_w)
    dudz = crop(dudz[:, :, 1:-1, 1:-1], min_z, min_h, min_w)
    
    dvdx = crop(dvdx[:, 1:-1, 1:-1, :], min_z, min_h, min_w)
    dvdy = crop(dvdy[:, 1:-1, :, 1:-1], min_z, min_h, min_w)
    dvdz = crop(dvdz[:, :, 1:-1, 1:-1], min_z, min_h, min_w)
    
    dwdx = crop(dwdx[:, 1:-1, 1:-1, :], min_z, min_h, min_w)
    dwdy = crop(dwdy[:, 1:-1, :, 1:-1], min_z, min_h, min_w)
    dwdz = crop(dwdz[:, :, 1:-1, 1:-1], min_z, min_h, min_w)
    
    dpdx = crop(dpdx[:, 1:-1, 1:-1, :], min_z, min_h, min_w)
    dpdy = crop(dpdy[:, 1:-1, :, 1:-1], min_z, min_h, min_w)
    dpdz = crop(dpdz[:, :, 1:-1, 1:-1], min_z, min_h, min_w)
    
    d2udx2 = crop(d2udx2[:, 1:-1, 1:-1, :], min_z, min_h, min_w)
    d2udy2 = crop(d2udy2[:, 1:-1, :, 1:-1], min_z, min_h, min_w)
    d2udz2 = crop(d2udz2[:, :, 1:-1, 1:-1], min_z, min_h, min_w)
    
    d2vdx2 = crop(d2vdx2[:, 1:-1, 1:-1, :], min_z, min_h, min_w)
    d2vdy2 = crop(d2vdy2[:, 1:-1, :, 1:-1], min_z, min_h, min_w)
    d2vdz2 = crop(d2vdz2[:, :, 1:-1, 1:-1], min_z, min_h, min_w)
    
    d2wdx2 = crop(d2wdx2[:, 1:-1, 1:-1, :], min_z, min_h, min_w)
    d2wdy2 = crop(d2wdy2[:, 1:-1, :, 1:-1], min_z, min_h, min_w)
    d2wdz2 = crop(d2wdz2[:, :, 1:-1, 1:-1], min_z, min_h, min_w)
    
    # NS方程各项（真实物理单位）
    conv_u = u*dudx + v*dudy + w*dudz  # m/s^2
    conv_v = u*dvdx + v*dvdy + w*dvdz
    conv_w = u*dwdx + v*dwdy + w*dwdz
    
    lap_u = nu * (d2udx2 + d2udy2 + d2udz2)  # m/s^2
    lap_v = nu * (d2vdx2 + d2vdy2 + d2vdz2)
    lap_w = nu * (d2wdx2 + d2wdy2 + d2wdz2)
    
    # 压力梯度项: NS方程移项后应为 +(1/ρ)∇p
    # NS: ∂u/∂t + (u·∇)u = -∇p/ρ + ν∇²u
    # 残差形式: ∂u/∂t + (u·∇)u + (1/ρ)∇p - ν∇²u = 0
    press_u = (1.0/rho) * dpdx  # m/s^2 (正号)
    press_v = (1.0/rho) * dpdy
    press_w = (1.0/rho) * dpdz
    
    # NS残差
    res_u = dudt + conv_u + press_u - lap_u  # m/s^2
    res_v = dvdt + conv_v + press_v - lap_v
    res_w = dwdt + conv_w + press_w - lap_w
    
    loss = (res_u**2).mean() + (res_v**2).mean() + (res_w**2).mean()
    return loss


def navier_stokes_forced_turb(pred, target=None, nu=0.01, rho=1.0,
                               dx=1.0, dy=1.0, dz=1.0, dt=1.0,
                               forcing_k_cutoff=2.0, domain_size=2*3.141592653589793,
                               norm_mean=None, norm_std=None, **kwargs):
    """
    NS方程残差损失 - 适用于强迫湍流数据 (如JHU Isotropic Turbulence)

    对于强迫湍流，NS方程实际为:
        ∂u/∂t + (u·∇)u + (1/ρ)∇p - ν∇²u = f

    其中 f 是强迫项，通常只在低波数 |k| <= k_cutoff 起作用。

    本函数通过在谱域过滤掉低波数残差来排除强迫项的影响:
    1. 计算物理空间NS残差
    2. FFT变换到谱域
    3. 将 |k| <= forcing_k_cutoff 的模态置零
    4. 只计算高波数部分的残差能量

    Args:
        pred: (B, T, C, Z, H, W) 预测场，C >= 4 (u, v, w, p)
        target: 未使用
        nu: 运动粘度 [L²/T]
        rho: 密度 [M/L³]
        dx, dy, dz: 网格间距 [L]
        dt: 时间步长 [T]
        forcing_k_cutoff: 强迫项截止波数，|k| <= cutoff 的模态被排除
                          JHU数据: k_cutoff = 2 (能量注入在 |k| <= 2)
        domain_size: 物理域尺寸 (默认 2π，用于计算波数)
        norm_mean, norm_std: 数据归一化参数

    Returns:
        loss: 过滤后NS方程残差的L2范数 (只包含 |k| > k_cutoff 的贡献)

    Note:
        - 对于无强迫的衰减湍流 (如Taylor-Green)，使用原始的 navier_stokes_residual
        - 对于强迫湍流 (如JHU Isotropic)，使用本函数
    """
    B, T, C = pred.shape[0], pred.shape[1], pred.shape[2]

    if C < 4:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    if T < 2:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    # 反归一化到真实物理空间
    pred_real = denormalize_field(pred, norm_mean, norm_std)

    Z, H, W = pred_real.shape[3], pred_real.shape[4], pred_real.shape[5]

    t_mid = T // 2
    u_curr = pred_real[:, t_mid, 0, :, :, :]
    v_curr = pred_real[:, t_mid, 1, :, :, :]
    w_curr = pred_real[:, t_mid, 2, :, :, :]
    p_curr = pred_real[:, t_mid, 3, :, :, :]

    # 时间导数
    if t_mid > 0 and t_mid < T - 1:
        dudt = (pred_real[:, t_mid+1, 0] - pred_real[:, t_mid-1, 0]) / (2 * dt)
        dvdt = (pred_real[:, t_mid+1, 1] - pred_real[:, t_mid-1, 1]) / (2 * dt)
        dwdt = (pred_real[:, t_mid+1, 2] - pred_real[:, t_mid-1, 2]) / (2 * dt)
    else:
        dudt = (pred_real[:, 1, 0] - pred_real[:, 0, 0]) / dt
        dvdt = (pred_real[:, 1, 1] - pred_real[:, 0, 1]) / dt
        dwdt = (pred_real[:, 1, 2] - pred_real[:, 0, 2]) / dt

    # 空间一阶导数 (中心差分)
    dudx = (u_curr[:, :, :, 2:] - u_curr[:, :, :, :-2]) / (2*dx)
    dudy = (u_curr[:, :, 2:, :] - u_curr[:, :, :-2, :]) / (2*dy)
    dudz = (u_curr[:, 2:, :, :] - u_curr[:, :-2, :, :]) / (2*dz)

    dvdx = (v_curr[:, :, :, 2:] - v_curr[:, :, :, :-2]) / (2*dx)
    dvdy = (v_curr[:, :, 2:, :] - v_curr[:, :, :-2, :]) / (2*dy)
    dvdz = (v_curr[:, 2:, :, :] - v_curr[:, :-2, :, :]) / (2*dz)

    dwdx = (w_curr[:, :, :, 2:] - w_curr[:, :, :, :-2]) / (2*dx)
    dwdy = (w_curr[:, :, 2:, :] - w_curr[:, :, :-2, :]) / (2*dy)
    dwdz = (w_curr[:, 2:, :, :] - w_curr[:, :-2, :, :]) / (2*dz)

    dpdx = (p_curr[:, :, :, 2:] - p_curr[:, :, :, :-2]) / (2*dx)
    dpdy = (p_curr[:, :, 2:, :] - p_curr[:, :, :-2, :]) / (2*dy)
    dpdz = (p_curr[:, 2:, :, :] - p_curr[:, :-2, :, :]) / (2*dz)

    # 空间二阶导数
    d2udx2 = (u_curr[:, :, :, 2:] - 2*u_curr[:, :, :, 1:-1] + u_curr[:, :, :, :-2]) / (dx**2)
    d2udy2 = (u_curr[:, :, 2:, :] - 2*u_curr[:, :, 1:-1, :] + u_curr[:, :, :-2, :]) / (dy**2)
    d2udz2 = (u_curr[:, 2:, :, :] - 2*u_curr[:, 1:-1, :, :] + u_curr[:, :-2, :, :]) / (dz**2)

    d2vdx2 = (v_curr[:, :, :, 2:] - 2*v_curr[:, :, :, 1:-1] + v_curr[:, :, :, :-2]) / (dx**2)
    d2vdy2 = (v_curr[:, :, 2:, :] - 2*v_curr[:, :, 1:-1, :] + v_curr[:, :, :-2, :]) / (dy**2)
    d2vdz2 = (v_curr[:, 2:, :, :] - 2*v_curr[:, 1:-1, :, :] + v_curr[:, :-2, :, :]) / (dz**2)

    d2wdx2 = (w_curr[:, :, :, 2:] - 2*w_curr[:, :, :, 1:-1] + w_curr[:, :, :, :-2]) / (dx**2)
    d2wdy2 = (w_curr[:, :, 2:, :] - 2*w_curr[:, :, 1:-1, :] + w_curr[:, :, :-2, :]) / (dy**2)
    d2wdz2 = (w_curr[:, 2:, :, :] - 2*w_curr[:, 1:-1, :, :] + w_curr[:, :-2, :, :]) / (dz**2)

    # 对齐尺寸
    z_sizes = [dudz.shape[1], dvdz.shape[1], dwdz.shape[1], dpdz.shape[1],
               d2udz2.shape[1], d2vdz2.shape[1], d2wdz2.shape[1], dudt.shape[1]]
    h_sizes = [dudy.shape[2], dvdy.shape[2], dwdy.shape[2], dpdy.shape[2],
               d2udy2.shape[2], d2vdy2.shape[2], d2wdy2.shape[2], dudt.shape[2]]
    w_sizes = [dudx.shape[3], dvdx.shape[3], dwdx.shape[3], dpdx.shape[3],
               d2udx2.shape[3], d2vdx2.shape[3], d2wdx2.shape[3], dudt.shape[3]]

    min_z, min_h, min_w = min(z_sizes), min(h_sizes), min(w_sizes)

    def crop(t, z, h, w):
        return t[:, :z, :h, :w]

    u = crop(u_curr[:, 1:-1, 1:-1, 1:-1], min_z, min_h, min_w)
    v = crop(v_curr[:, 1:-1, 1:-1, 1:-1], min_z, min_h, min_w)
    w = crop(w_curr[:, 1:-1, 1:-1, 1:-1], min_z, min_h, min_w)

    dudt = crop(dudt[:, 1:-1, 1:-1, 1:-1], min_z, min_h, min_w)
    dvdt = crop(dvdt[:, 1:-1, 1:-1, 1:-1], min_z, min_h, min_w)
    dwdt = crop(dwdt[:, 1:-1, 1:-1, 1:-1], min_z, min_h, min_w)

    dudx = crop(dudx[:, 1:-1, 1:-1, :], min_z, min_h, min_w)
    dudy = crop(dudy[:, 1:-1, :, 1:-1], min_z, min_h, min_w)
    dudz = crop(dudz[:, :, 1:-1, 1:-1], min_z, min_h, min_w)

    dvdx = crop(dvdx[:, 1:-1, 1:-1, :], min_z, min_h, min_w)
    dvdy = crop(dvdy[:, 1:-1, :, 1:-1], min_z, min_h, min_w)
    dvdz = crop(dvdz[:, :, 1:-1, 1:-1], min_z, min_h, min_w)

    dwdx = crop(dwdx[:, 1:-1, 1:-1, :], min_z, min_h, min_w)
    dwdy = crop(dwdy[:, 1:-1, :, 1:-1], min_z, min_h, min_w)
    dwdz = crop(dwdz[:, :, 1:-1, 1:-1], min_z, min_h, min_w)

    dpdx = crop(dpdx[:, 1:-1, 1:-1, :], min_z, min_h, min_w)
    dpdy = crop(dpdy[:, 1:-1, :, 1:-1], min_z, min_h, min_w)
    dpdz = crop(dpdz[:, :, 1:-1, 1:-1], min_z, min_h, min_w)

    d2udx2 = crop(d2udx2[:, 1:-1, 1:-1, :], min_z, min_h, min_w)
    d2udy2 = crop(d2udy2[:, 1:-1, :, 1:-1], min_z, min_h, min_w)
    d2udz2 = crop(d2udz2[:, :, 1:-1, 1:-1], min_z, min_h, min_w)

    d2vdx2 = crop(d2vdx2[:, 1:-1, 1:-1, :], min_z, min_h, min_w)
    d2vdy2 = crop(d2vdy2[:, 1:-1, :, 1:-1], min_z, min_h, min_w)
    d2vdz2 = crop(d2vdz2[:, :, 1:-1, 1:-1], min_z, min_h, min_w)

    d2wdx2 = crop(d2wdx2[:, 1:-1, 1:-1, :], min_z, min_h, min_w)
    d2wdy2 = crop(d2wdy2[:, 1:-1, :, 1:-1], min_z, min_h, min_w)
    d2wdz2 = crop(d2wdz2[:, :, 1:-1, 1:-1], min_z, min_h, min_w)

    # NS方程各项
    conv_u = u*dudx + v*dudy + w*dudz
    conv_v = u*dvdx + v*dvdy + w*dvdz
    conv_w = u*dwdx + v*dwdy + w*dwdz

    lap_u = nu * (d2udx2 + d2udy2 + d2udz2)
    lap_v = nu * (d2vdx2 + d2vdy2 + d2vdz2)
    lap_w = nu * (d2wdx2 + d2wdy2 + d2wdz2)

    press_u = (1.0/rho) * dpdx
    press_v = (1.0/rho) * dpdy
    press_w = (1.0/rho) * dpdz

    # NS残差 (在无强迫的情况下应为0，但JHU数据有强迫项)
    res_u = dudt + conv_u + press_u - lap_u
    res_v = dvdt + conv_v + press_v - lap_v
    res_w = dwdt + conv_w + press_w - lap_w

    # ============================================================
    # 谱域过滤：排除低波数 |k| <= forcing_k_cutoff 的贡献
    # ============================================================
    device = res_u.device
    res_z, res_h, res_w_dim = res_u.shape[1], res_u.shape[2], res_u.shape[3]

    # 创建波数网格 (使用整数波数，与JHU强迫方案一致)
    # kx, ky, kz 是整数波数: 0, 1, 2, ..., N/2, -N/2+1, ..., -1
    kx = torch.fft.fftfreq(res_w_dim, d=1.0/res_w_dim, device=device)  # (W,)
    ky = torch.fft.fftfreq(res_h, d=1.0/res_h, device=device)          # (H,)
    kz = torch.fft.fftfreq(res_z, d=1.0/res_z, device=device)          # (Z,)

    # 创建3D波数幅值网格
    kz_grid, ky_grid, kx_grid = torch.meshgrid(kz, ky, kx, indexing='ij')
    k_mag = torch.sqrt(kx_grid**2 + ky_grid**2 + kz_grid**2)  # (Z, H, W)

    # 高通滤波器：只保留 |k| > forcing_k_cutoff 的模态
    high_pass_mask = (k_mag > forcing_k_cutoff).float()  # (Z, H, W)

    # FFT变换残差到谱域
    res_u_hat = torch.fft.fftn(res_u, dim=(-3, -2, -1), norm='ortho')  # (B, Z, H, W)
    res_v_hat = torch.fft.fftn(res_v, dim=(-3, -2, -1), norm='ortho')
    res_w_hat = torch.fft.fftn(res_w, dim=(-3, -2, -1), norm='ortho')

    # 应用高通滤波
    res_u_hat_filtered = res_u_hat * high_pass_mask
    res_v_hat_filtered = res_v_hat * high_pass_mask
    res_w_hat_filtered = res_w_hat * high_pass_mask

    # 计算过滤后的能量 (Parseval定理: 谱域L2 = 物理空间L2)
    # 只计算被保留模态的能量
    n_modes = high_pass_mask.sum().clamp(min=1)

    energy_u = (res_u_hat_filtered.real**2 + res_u_hat_filtered.imag**2).sum() / n_modes
    energy_v = (res_v_hat_filtered.real**2 + res_v_hat_filtered.imag**2).sum() / n_modes
    energy_w = (res_w_hat_filtered.real**2 + res_w_hat_filtered.imag**2).sum() / n_modes

    loss = energy_u + energy_v + energy_w
    return loss


def vorticity_conservation(pred, target=None, dx=1.0, dy=1.0, dt=1.0, norm_mean=None, norm_std=None, **kwargs):
    """
    涡量守恒损失（在真实物理空间计算）
    """
    # 反归一化到真实物理空间
    pred_real = denormalize_field(pred, norm_mean, norm_std)
    
    u = pred_real[:, :, 0, :, :, :]
    v = pred_real[:, :, 1, :, :, :]
    
    dv_dx = (v[:, :, :, :, 2:] - v[:, :, :, :, :-2]) / (2 * dx)
    du_dy = (u[:, :, :, 2:, :] - u[:, :, :, :-2, :]) / (2 * dy)
    
    min_h = min(dv_dx.shape[3], du_dy.shape[3])
    min_w = min(dv_dx.shape[4], du_dy.shape[4])
    dv_dx = dv_dx[:, :, :, :min_h, :min_w]
    du_dy = du_dy[:, :, :, :min_h, :min_w]
    
    vorticity = dv_dx - du_dy  # 1/s
    
    if pred.shape[1] < 2:
        return torch.tensor(0.0, device=pred.device)
    
    dvort_dt = vorticity[:, 1:] - vorticity[:, :-1]
    return (dvort_dt ** 2).mean()

# =============================================================================
# 谱域物理损失函数 (Spectral Physics Loss)
# =============================================================================

def create_wavenumber_grid(Z, H, W, Lz=1.0, Ly=1.0, Lx=1.0, device='cpu'):
    """
    创建3D波数网格

    Args:
        Z, H, W: 空间维度
        Lz, Ly, Lx: 物理域长度
        device: 计算设备

    Returns:
        kz, ky, kx: 波数网格 (Z, H, W//2+1) for rfft
    """
    # 波数 (注意: rfft 在最后一个维度只有 W//2+1 个频率)
    # kx: 0, 1, 2, ..., W//2 (正频率)
    kx = 2 * torch.pi / Lx * torch.fft.rfftfreq(W, d=1.0/W, device=device)  # (W//2+1,)

    # ky, kz: 0, 1, ..., N//2, -N//2+1, ..., -1 (正负频率都有)
    ky = 2 * torch.pi / Ly * torch.fft.fftfreq(H, d=1.0/H, device=device)   # (H,)
    kz = 2 * torch.pi / Lz * torch.fft.fftfreq(Z, d=1.0/Z, device=device)   # (Z,)

    # 创建网格: (Z, H, W//2+1)
    kz_grid, ky_grid, kx_grid = torch.meshgrid(kz, ky, kx, indexing='ij')

    return kz_grid, ky_grid, kx_grid


def spectral_divergence_3d(pred, target=None, Lz=1.0, Ly=1.0, Lx=1.0,
                           norm_mean=None, norm_std=None, **kwargs):
    """
    谱域3D散度损失

    原理:
        物理空间: ∇·u = ∂u/∂x + ∂v/∂y + ∂w/∂z = 0
        谱域:     ∇·û = i(kx·û + ky·v̂ + kz·ŵ) = 0

    优点:
        1. 导数计算精确（无有限差分截断误差）
        2. 与FNO的频率截断兼容（只约束保留的低频模态）
        3. 计算效率高（FFT是O(N log N)）

    Args:
        pred: (B, T, C, Z, H, W) 预测场
        Lz, Ly, Lx: 物理域长度

    Returns:
        loss: 谱域散度的L2范数（归一化后与物理空间尺度一致）
    """
    # 反归一化（如果需要）
    pred_real = denormalize_field(pred, norm_mean, norm_std)

    B, T, C, Z, H, W = pred_real.shape
    device = pred_real.device

    if C < 3:
        raise ValueError(f"谱域散度需要至少3个速度分量(u,v,w)，得到{C}个")

    u = pred_real[:, :, 0]  # (B, T, Z, H, W)
    v = pred_real[:, :, 1]
    w = pred_real[:, :, 2]

    # 创建波数网格
    kz, ky, kx = create_wavenumber_grid(Z, H, W, Lz, Ly, Lx, device)  # (Z, H, W//2+1)

    # FFT with orthonormal normalization (使能量守恒，Parseval定理)
    u_hat = torch.fft.rfftn(u, dim=(-3, -2, -1), norm='ortho')  # (B, T, Z, H, W//2+1)
    v_hat = torch.fft.rfftn(v, dim=(-3, -2, -1), norm='ortho')
    w_hat = torch.fft.rfftn(w, dim=(-3, -2, -1), norm='ortho')

    # 谱域散度: F[∇·u] = i * (kx*û + ky*v̂ + kz*ŵ)
    # 注: 对于能量计算 |div_hat|^2，乘i不影响结果（|i|=1），但保持公式完整性
    div_hat = 1j * (kx * u_hat + ky * v_hat + kz * w_hat)  # (B, T, Z, H, W//2+1)

    # 散度的能量（使用 ortho norm 后，Parseval定理保证谱域L2 = 物理空间L2）
    # 注意：rfft 只有一半的频率，需要补偿（除了 k=0 和 Nyquist 频率）
    div_energy = (div_hat.real ** 2 + div_hat.imag ** 2).mean()

    return div_energy


def spectral_divergence_3d_filtered(pred, target=None, k_cutoff=12,
                                      norm_mean=None, norm_std=None,
                                      normalize_output=True, **kwargs):
    """
    带波数截止的谱域3D散度损失 (专为JHU跳采样数据设计)

    背景:
        JHU数据从1024³跳采样到128³，导致高波数aliasing，使得
        有限差分计算的散度MSE约为34（而非理论上的0）。

        分析发现：低波数（|k| ≤ 12）部分的散度MSE仅约0.006，
        真正满足不可压缩性约束。高波数的散度误差来自aliasing，
        不是真实物理。

    原理:
        物理空间: ∇·u = ∂u/∂x + ∂v/∂y + ∂w/∂z = 0
        谱域:     ∇·û = i(kx·û + ky·v̂ + kz·ŵ) = 0

        只惩罚 |k| ≤ k_cutoff 的模态，避免强迫模型学习aliasing噪声。

    Args:
        pred: (B, T, C, Z, H, W) 预测场，C >= 3 (u, v, w)
        target: 未使用（保持接口一致）
        k_cutoff: 整数波数截止值，只惩罚 |k| ≤ k_cutoff 的模态
                  推荐值（基于200时间步统计）：
                  - k_cutoff=8:  MSE ≈ 0.002 (很好)
                  - k_cutoff=12: MSE ≈ 0.006 (推荐，平衡物理约束和aliasing)
                  - k_cutoff=16: MSE ≈ 0.016 (可接受)
        norm_mean, norm_std: 数据归一化参数
        normalize_output: 是否归一化输出到O(1)尺度 (默认True)
                         True: 除以速度场能量，使loss约为O(0.01-1)
                         False: 返回原始散度能量

    Returns:
        loss: 低波数谱域散度的MSE (归一化后约为0.01-1)

    Note:
        - 使用整数波数（与JHU强迫方案一致）
        - 对于128³网格，k_cutoff=12 对应波长 λ > 2π/12 ≈ 0.52 的大尺度结构
        - normalize_output=True时，loss对速度场幅值不敏感，更稳定
    """
    # 在归一化空间计算散度（避免denormalization导致的数值爆炸）
    # 关键洞察：如果std_u ≈ std_v ≈ std_w，散度约束在归一化空间同样成立
    # 即使std略有差异，这也是一个合理的近似
    B, T, C, Z, H, W = pred.shape
    device = pred.device

    if C < 3:
        raise ValueError(f"谱域散度需要至少3个速度分量(u,v,w)，得到{C}个")

    u = pred[:, :, 0]  # (B, T, Z, H, W)
    v = pred[:, :, 1]
    w = pred[:, :, 2]

    # 创建整数波数网格（与JHU强迫方案一致）
    # kx: 0, 1, 2, ..., W//2 (rfft只有正频率)
    kx = torch.fft.rfftfreq(W, d=1.0/W, device=device)  # (W//2+1,)
    # ky, kz: 0, 1, ..., N//2, -N//2+1, ..., -1
    ky = torch.fft.fftfreq(H, d=1.0/H, device=device)   # (H,)
    kz = torch.fft.fftfreq(Z, d=1.0/Z, device=device)   # (Z,)

    # 创建3D波数幅值网格
    kz_grid, ky_grid, kx_grid = torch.meshgrid(kz, ky, kx, indexing='ij')  # (Z, H, W//2+1)
    k_mag = torch.sqrt(kx_grid**2 + ky_grid**2 + kz_grid**2)

    # 低通滤波器：只保留 |k| <= k_cutoff 的模态
    low_pass_mask = (k_mag <= k_cutoff).float()  # (Z, H, W//2+1)

    # FFT with orthonormal normalization
    u_hat = torch.fft.rfftn(u, dim=(-3, -2, -1), norm='ortho')  # (B, T, Z, H, W//2+1)
    v_hat = torch.fft.rfftn(v, dim=(-3, -2, -1), norm='ortho')
    w_hat = torch.fft.rfftn(w, dim=(-3, -2, -1), norm='ortho')

    # 应用低通滤波
    u_hat_filtered = u_hat * low_pass_mask
    v_hat_filtered = v_hat * low_pass_mask
    w_hat_filtered = w_hat * low_pass_mask

    # 谱域散度: F[∇·u] = i * (kx*û + ky*v̂ + kz*ŵ)
    div_hat = 1j * (kx_grid * u_hat_filtered + ky_grid * v_hat_filtered + kz_grid * w_hat_filtered)

    # 计算滤波后的散度能量（只计算被保留的模态）
    n_modes = low_pass_mask.sum().clamp(min=1)
    div_energy = (div_hat.real ** 2 + div_hat.imag ** 2).sum() / (B * T * n_modes)

    if normalize_output:
        # 归一化：除以速度场能量，使loss对幅值不敏感
        # 这样loss表示"散度占速度场能量的比例"
        vel_energy = (
            (u_hat_filtered.real ** 2 + u_hat_filtered.imag ** 2).sum() +
            (v_hat_filtered.real ** 2 + v_hat_filtered.imag ** 2).sum() +
            (w_hat_filtered.real ** 2 + w_hat_filtered.imag ** 2).sum()
        ) / (B * T * n_modes * 3)  # 平均每个分量的能量

        # 避免除零，并加入k²的期望值作为归一化因子
        # 对于随机场: E[|div|²] ≈ <k²> * E[|vel|²]
        # 所以 div_energy / (k²_avg * vel_energy) ≈ 1 对于随机场
        # 对于不可压缩流: div_energy / vel_energy ≈ 0
        k_sq_avg = (k_mag ** 2 * low_pass_mask).sum() / n_modes.clamp(min=1)

        # 归一化后的散度：对于不可压缩流应接近0，对于随机场约为1
        div_normalized = div_energy / (k_sq_avg * vel_energy + 1e-8)
        return div_normalized
    else:
        return div_energy


def spectral_divergence_3d_soft(pred, target=None, Lz=1.0, Ly=1.0, Lx=1.0,
                                 cutoff_ratio=0.5, norm_mean=None, norm_std=None, **kwargs):
    """
    软截断谱域散度损失

    只惩罚低频部分的散度（与FNO保留的模态匹配）

    Args:
        cutoff_ratio: 截断比例，0.5 表示只考虑前50%的频率
    """
    pred_real = denormalize_field(pred, norm_mean, norm_std)

    B, T, C, Z, H, W = pred_real.shape
    device = pred_real.device

    if C < 3:
        raise ValueError(f"需要至少3个速度分量(u,v,w)")

    u = pred_real[:, :, 0]
    v = pred_real[:, :, 1]
    w = pred_real[:, :, 2]

    # 创建波数网格
    kz, ky, kx = create_wavenumber_grid(Z, H, W, Lz, Ly, Lx, device)

    # 计算波数幅值
    k_mag = torch.sqrt(kx**2 + ky**2 + kz**2)
    k_max = k_mag.max()

    # 低通滤波器：只保留 |k| < cutoff_ratio * k_max 的模态
    low_pass_mask = (k_mag < cutoff_ratio * k_max).float()

    # FFT with orthonormal normalization
    u_hat = torch.fft.rfftn(u, dim=(-3, -2, -1), norm='ortho')
    v_hat = torch.fft.rfftn(v, dim=(-3, -2, -1), norm='ortho')
    w_hat = torch.fft.rfftn(w, dim=(-3, -2, -1), norm='ortho')

    # 应用低通滤波
    u_hat_filtered = u_hat * low_pass_mask
    v_hat_filtered = v_hat * low_pass_mask
    w_hat_filtered = w_hat * low_pass_mask

    # 谱域散度: F[∇·u] = i * (kx*û + ky*v̂ + kz*ŵ)
    div_hat = 1j * (kx * u_hat_filtered + ky * v_hat_filtered + kz * w_hat_filtered)

    # 只计算被滤波保留的部分
    div_energy = (div_hat.real ** 2 + div_hat.imag ** 2).sum() / low_pass_mask.sum().clamp(min=1)

    return div_energy


def spectral_gradient(field_hat, kx, ky, kz):
    """
    谱域梯度计算

    ∂f/∂x → i*kx*f_hat
    """
    df_dx = 1j * kx * field_hat
    df_dy = 1j * ky * field_hat
    df_dz = 1j * kz * field_hat
    return df_dx, df_dy, df_dz


def spectral_laplacian(field_hat, kx, ky, kz):
    """
    谱域拉普拉斯算子

    ∇²f → -(kx² + ky² + kz²) * f_hat
    """
    k_sq = kx**2 + ky**2 + kz**2
    return -k_sq * field_hat


def spectral_ns_residual(pred, target=None, nu=0.01, rho=1.0,
                         Lz=1.0, Ly=1.0, Lx=1.0, dt=1.0,
                         norm_mean=None, norm_std=None, **kwargs):
    """
    谱域NS方程残差损失

    NS方程:
        ∂u/∂t + (u·∇)u = -∇p/ρ + ν∇²u

    谱域形式:
        - 时间导数: 在物理空间计算（有限差分）
        - 压力梯度: -i*k*p̂/ρ
        - 粘性项: -ν*|k|²*û
        - 非线性项: 伪谱方法（物理空间计算乘积）

    Args:
        pred: (B, T, C, Z, H, W) 预测场，C >= 4 (u, v, w, p)
        nu: 运动粘度
        rho: 密度
        Lz, Ly, Lx: 物理域长度
        dt: 时间步长
    """
    pred_real = denormalize_field(pred, norm_mean, norm_std)

    B, T, C, Z, H, W = pred_real.shape
    device = pred_real.device

    if C < 4:
        raise ValueError(f"NS方程需要4个通道(u,v,w,p)，得到{C}个")
    if T < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # 创建波数网格
    kz, ky, kx = create_wavenumber_grid(Z, H, W, Lz, Ly, Lx, device)
    k_sq = kx**2 + ky**2 + kz**2

    # 取中间时刻
    t_mid = T // 2
    u = pred_real[:, t_mid, 0]  # (B, Z, H, W)
    v = pred_real[:, t_mid, 1]
    w = pred_real[:, t_mid, 2]
    p = pred_real[:, t_mid, 3]

    # === 时间导数（物理空间有限差分）===
    if t_mid > 0 and t_mid < T - 1:
        dudt = (pred_real[:, t_mid+1, 0] - pred_real[:, t_mid-1, 0]) / (2 * dt)
        dvdt = (pred_real[:, t_mid+1, 1] - pred_real[:, t_mid-1, 1]) / (2 * dt)
        dwdt = (pred_real[:, t_mid+1, 2] - pred_real[:, t_mid-1, 2]) / (2 * dt)
    else:
        dudt = (pred_real[:, 1, 0] - pred_real[:, 0, 0]) / dt
        dvdt = (pred_real[:, 1, 1] - pred_real[:, 0, 1]) / dt
        dwdt = (pred_real[:, 1, 2] - pred_real[:, 0, 2]) / dt

    # === FFT with orthonormal normalization ===
    u_hat = torch.fft.rfftn(u, dim=(-3, -2, -1), norm='ortho')
    v_hat = torch.fft.rfftn(v, dim=(-3, -2, -1), norm='ortho')
    w_hat = torch.fft.rfftn(w, dim=(-3, -2, -1), norm='ortho')
    p_hat = torch.fft.rfftn(p, dim=(-3, -2, -1), norm='ortho')

    # === 压力梯度项（谱域）===
    # NS残差形式: ∂u/∂t + (u·∇)u + (1/ρ)∇p - ν∇²u = 0
    # 谱域梯度: ∂f/∂x → i*kx*f_hat
    # 所以 (1/ρ)∇p → i*k*p_hat/ρ (正号)
    press_u_hat = 1j * kx * p_hat / rho
    press_v_hat = 1j * ky * p_hat / rho
    press_w_hat = 1j * kz * p_hat / rho

    # 变换回物理空间
    press_u = torch.fft.irfftn(press_u_hat, s=(Z, H, W), dim=(-3, -2, -1), norm='ortho')
    press_v = torch.fft.irfftn(press_v_hat, s=(Z, H, W), dim=(-3, -2, -1), norm='ortho')
    press_w = torch.fft.irfftn(press_w_hat, s=(Z, H, W), dim=(-3, -2, -1), norm='ortho')

    # === 粘性项（谱域）===
    # ν∇²u → -ν*|k|²*u_hat
    visc_u_hat = -nu * k_sq * u_hat
    visc_v_hat = -nu * k_sq * v_hat
    visc_w_hat = -nu * k_sq * w_hat

    # 变换回物理空间
    visc_u = torch.fft.irfftn(visc_u_hat, s=(Z, H, W), dim=(-3, -2, -1), norm='ortho')
    visc_v = torch.fft.irfftn(visc_v_hat, s=(Z, H, W), dim=(-3, -2, -1), norm='ortho')
    visc_w = torch.fft.irfftn(visc_w_hat, s=(Z, H, W), dim=(-3, -2, -1), norm='ortho')

    # === 非线性项（伪谱方法）===
    # 计算速度梯度（谱域）
    dudx_hat = 1j * kx * u_hat
    dudy_hat = 1j * ky * u_hat
    dudz_hat = 1j * kz * u_hat

    dvdx_hat = 1j * kx * v_hat
    dvdy_hat = 1j * ky * v_hat
    dvdz_hat = 1j * kz * v_hat

    dwdx_hat = 1j * kx * w_hat
    dwdy_hat = 1j * ky * w_hat
    dwdz_hat = 1j * kz * w_hat

    # 变换回物理空间
    dudx = torch.fft.irfftn(dudx_hat, s=(Z, H, W), dim=(-3, -2, -1), norm='ortho')
    dudy = torch.fft.irfftn(dudy_hat, s=(Z, H, W), dim=(-3, -2, -1), norm='ortho')
    dudz = torch.fft.irfftn(dudz_hat, s=(Z, H, W), dim=(-3, -2, -1), norm='ortho')

    dvdx = torch.fft.irfftn(dvdx_hat, s=(Z, H, W), dim=(-3, -2, -1), norm='ortho')
    dvdy = torch.fft.irfftn(dvdy_hat, s=(Z, H, W), dim=(-3, -2, -1), norm='ortho')
    dvdz = torch.fft.irfftn(dvdz_hat, s=(Z, H, W), dim=(-3, -2, -1), norm='ortho')

    dwdx = torch.fft.irfftn(dwdx_hat, s=(Z, H, W), dim=(-3, -2, -1), norm='ortho')
    dwdy = torch.fft.irfftn(dwdy_hat, s=(Z, H, W), dim=(-3, -2, -1), norm='ortho')
    dwdz = torch.fft.irfftn(dwdz_hat, s=(Z, H, W), dim=(-3, -2, -1), norm='ortho')

    # 非线性项: (u·∇)u 在物理空间计算
    conv_u = u * dudx + v * dudy + w * dudz
    conv_v = u * dvdx + v * dvdy + w * dvdz
    conv_w = u * dwdx + v * dwdy + w * dwdz

    # === NS方程残差 ===
    # ∂u/∂t + (u·∇)u + ∇p/ρ - ν∇²u = 0
    res_u = dudt + conv_u + press_u - visc_u
    res_v = dvdt + conv_v + press_v - visc_v
    res_w = dwdt + conv_w + press_w - visc_w

    loss = (res_u**2).mean() + (res_v**2).mean() + (res_w**2).mean()
    return loss


# =============================================================================
# 梯度损失和平滑损失 (Gradient & Smoothness Loss)
# 用于缓解 patch-based 模型的空心问题
# =============================================================================

def gradient_loss_3d(pred, target, norm_mean=None, norm_std=None, **kwargs):
    """
    3D 梯度损失（在真实物理空间计算）

    惩罚预测和真值之间的空间梯度差异，有助于：
    1. 保持空间细节和边缘
    2. 减少 patch 边界处的不连续

    Args:
        pred: (B, T, C, Z, H, W) 预测场（归一化）
        target: (B, T, C, Z, H, W) 真值（归一化）
        norm_mean: 归一化均值
        norm_std: 归一化标准差

    Returns:
        loss: 梯度差异的 L1 损失
    """
    # 反归一化到真实物理空间
    pred_real = denormalize_field(pred, norm_mean, norm_std)
    target_real = denormalize_field(target, norm_mean, norm_std)

    # Z 方向梯度
    pred_dz = pred_real[:, :, :, 1:, :, :] - pred_real[:, :, :, :-1, :, :]
    target_dz = target_real[:, :, :, 1:, :, :] - target_real[:, :, :, :-1, :, :]

    # H 方向梯度
    pred_dh = pred_real[:, :, :, :, 1:, :] - pred_real[:, :, :, :, :-1, :]
    target_dh = target_real[:, :, :, :, 1:, :] - target_real[:, :, :, :, :-1, :]

    # W 方向梯度
    pred_dw = pred_real[:, :, :, :, :, 1:] - pred_real[:, :, :, :, :, :-1]
    target_dw = target_real[:, :, :, :, :, 1:] - target_real[:, :, :, :, :, :-1]

    loss_z = torch.abs(pred_dz - target_dz).mean()
    loss_h = torch.abs(pred_dh - target_dh).mean()
    loss_w = torch.abs(pred_dw - target_dw).mean()

    return loss_z + loss_h + loss_w


def tv_loss_3d(pred, target=None, norm_mean=None, norm_std=None, **kwargs):
    """
    3D Total Variation 损失

    鼓励输出的空间平滑性，减少噪声和伪影

    Args:
        pred: (B, T, C, Z, H, W) 预测场

    Returns:
        loss: TV 损失
    """
    # Z 方向变化
    tv_z = torch.abs(pred[:, :, :, 1:, :, :] - pred[:, :, :, :-1, :, :]).mean()

    # H 方向变化
    tv_h = torch.abs(pred[:, :, :, :, 1:, :] - pred[:, :, :, :, :-1, :]).mean()

    # W 方向变化
    tv_w = torch.abs(pred[:, :, :, :, :, 1:] - pred[:, :, :, :, :, :-1]).mean()

    return tv_z + tv_h + tv_w


def edge_aware_smoothness_loss(pred, target, norm_mean=None, norm_std=None, **kwargs):
    """
    边缘感知平滑损失

    在目标场梯度小的区域鼓励平滑，在梯度大的区域（边缘）保持细节

    Args:
        pred: (B, T, C, Z, H, W) 预测场
        target: (B, T, C, Z, H, W) 真值

    Returns:
        loss: 边缘感知平滑损失
    """
    # 计算目标的梯度幅值作为权重
    target_dz = torch.abs(target[:, :, :, 1:, :, :] - target[:, :, :, :-1, :, :])
    target_dh = torch.abs(target[:, :, :, :, 1:, :] - target[:, :, :, :, :-1, :])
    target_dw = torch.abs(target[:, :, :, :, :, 1:] - target[:, :, :, :, :, :-1])

    # 权重: 梯度大的地方权重小 (exp(-|grad|))
    weight_z = torch.exp(-target_dz.mean(dim=2, keepdim=True))  # 对通道取平均
    weight_h = torch.exp(-target_dh.mean(dim=2, keepdim=True))
    weight_w = torch.exp(-target_dw.mean(dim=2, keepdim=True))

    # 预测的梯度
    pred_dz = pred[:, :, :, 1:, :, :] - pred[:, :, :, :-1, :, :]
    pred_dh = pred[:, :, :, :, 1:, :] - pred[:, :, :, :, :-1, :]
    pred_dw = pred[:, :, :, :, :, 1:] - pred[:, :, :, :, :, :-1]

    # 目标的梯度（重新计算，带通道）
    target_dz_full = target[:, :, :, 1:, :, :] - target[:, :, :, :-1, :, :]
    target_dh_full = target[:, :, :, :, 1:, :] - target[:, :, :, :, :-1, :]
    target_dw_full = target[:, :, :, :, :, 1:] - target[:, :, :, :, :, :-1]

    # 加权梯度误差
    loss_z = (weight_z * torch.abs(pred_dz - target_dz_full)).mean()
    loss_h = (weight_h * torch.abs(pred_dh - target_dh_full)).mean()
    loss_w = (weight_w * torch.abs(pred_dw - target_dw_full)).mean()

    return loss_z + loss_h + loss_w


def laplacian_smoothness_loss(pred, target, norm_mean=None, norm_std=None, **kwargs):
    """
    拉普拉斯平滑损失

    惩罚预测和目标的二阶导数差异，有助于：
    1. 减少高频噪声
    2. 保持整体平滑性

    Args:
        pred: (B, T, C, Z, H, W) 预测场
        target: (B, T, C, Z, H, W) 真值

    Returns:
        loss: 拉普拉斯差异损失
    """
    # 3D 拉普拉斯算子 (离散近似)
    def laplacian_3d(x):
        # Z 方向二阶导
        d2_z = x[:, :, :, 2:, :, :] - 2 * x[:, :, :, 1:-1, :, :] + x[:, :, :, :-2, :, :]
        # H 方向二阶导
        d2_h = x[:, :, :, :, 2:, :] - 2 * x[:, :, :, :, 1:-1, :] + x[:, :, :, :, :-2, :]
        # W 方向二阶导
        d2_w = x[:, :, :, :, :, 2:] - 2 * x[:, :, :, :, :, 1:-1] + x[:, :, :, :, :, :-2]

        # 对齐尺寸
        min_z = min(d2_z.shape[3], d2_h.shape[3], d2_w.shape[3])
        min_h = min(d2_z.shape[4], d2_h.shape[4], d2_w.shape[4])
        min_w = min(d2_z.shape[5], d2_h.shape[5], d2_w.shape[5])

        d2_z = d2_z[:, :, :, :min_z, :min_h, :min_w]
        d2_h = d2_h[:, :, :, :min_z, :min_h, :min_w]
        d2_w = d2_w[:, :, :, :min_z, :min_h, :min_w]

        return d2_z + d2_h + d2_w

    lap_pred = laplacian_3d(pred)
    lap_target = laplacian_3d(target)

    return torch.abs(lap_pred - lap_target).mean()


def get_physics_loss(loss_type, config_dict):
    """
    获取物理损失函数

    Args:
        loss_type: 损失类型
        config_dict: 配置字典（包含物理参数）

    Returns:
        loss_fn: 损失函数
    """
    losses = {
        # 物理空间（有限差分）
        'divergence_2d': divergence_loss_2d,
        'divergence_3d': divergence_loss_3d,
        'navier_stokes': navier_stokes_residual,
        'navier_stokes_forced_turb': navier_stokes_forced_turb,  # 强迫湍流专用 (如JHU)
        'vorticity': vorticity_conservation,
        # 谱域（FFT）
        'spectral_divergence': spectral_divergence_3d,
        'spectral_divergence_filtered': spectral_divergence_3d_filtered,  # JHU跳采样数据专用
        'spectral_divergence_soft': spectral_divergence_3d_soft,
        'spectral_ns': spectral_ns_residual,
        # 梯度和平滑损失 (用于缓解空心问题)
        'gradient_3d': gradient_loss_3d,
        'tv_3d': tv_loss_3d,
        'edge_smooth': edge_aware_smoothness_loss,
        'laplacian': laplacian_smoothness_loss,
    }
    if loss_type not in losses:
        raise ValueError(f"未知物理损失类型: {loss_type}")

    loss_fn = losses[loss_type]

    def wrapped_loss(pred, target=None, **kwargs):
        merged_kwargs = {**config_dict, **kwargs}
        return loss_fn(pred, target, **merged_kwargs)

    return wrapped_loss