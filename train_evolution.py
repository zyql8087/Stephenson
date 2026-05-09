import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from kinematics import StephensonIII
import copy

def load_target_data(filepath):
    data = np.loadtxt(filepath, delimiter=',', skiprows=1)
    phase = torch.tensor(data[:, 0], dtype=torch.float32)
    foot_x = torch.tensor(data[:, 1], dtype=torch.float32)
    foot_y = torch.tensor(data[:, 2], dtype=torch.float32)
    theta2_seq = phase * 2 * np.pi
    return theta2_seq, foot_x, foot_y

def calculate_loss(model, pred_x, pred_y, target_x, target_y, pred_angles, epoch=0):
    pred_x = pred_x.view(-1)
    pred_y = pred_y.view(-1)
    target_x = target_x.view(-1)
    target_y = target_y.view(-1)

    min_len = min(len(pred_x), len(target_x))
    p_x, t_x = pred_x[:min_len], target_x[:min_len]
    p_y, t_y = pred_y[:min_len], target_y[:min_len]

    # 1. 基础 NaN 防御
    if torch.isnan(p_x).any() or torch.isnan(p_y).any():
        return torch.tensor(1000.0, requires_grad=True)

    # 2. 算出 E_foot (足端轨迹误差，支撑相 5 倍加权)
    y_mean = torch.mean(t_y)
    weights = torch.ones_like(t_y)
    weights[t_y < y_mean] = 5.0
    mse_x = torch.mean(weights * (p_x - t_x) ** 2)
    mse_y = torch.mean(weights * (p_y - t_y) ** 2)
    E_foot = mse_x + mse_y

    # 提取内部角度
    t3 = pred_angles[:min_len, 0]
    t4 = pred_angles[:min_len, 1]
    t6 = pred_angles[:min_len, 2]
    t7 = pred_angles[:min_len, 3]

    # 3. 算出 E_knee (膝角姿态防反弯误差)
    alpha = model.get_actual_lengths()[7]
    knee_angle = (t3 + alpha) - t6
    E_knee = torch.mean(torch.relu(torch.cos(knee_angle) - 0.95)) * 10.0

    # 4. 算出 E_ankle (踝角/脚掌平稳度误差)
    stance_phase_mask = t_y < y_mean
    if stance_phase_mask.any():
        E_ankle = torch.var(t6[stance_phase_mask])
    else:
        E_ankle = torch.tensor(0.0)

    # 5. 算出 Penalty (传动角物理死点惩罚)
    cos_gamma1 = torch.abs(torch.cos(t3 - t4))
    cos_gamma2 = torch.abs(torch.cos(t6 - t7))
    trans_penalty1 = torch.sum(torch.relu(cos_gamma1 - 0.866)) * 500.0
    trans_penalty2 = torch.sum(torch.relu(cos_gamma2 - 0.866)) * 500.0
    Penalty = trans_penalty1 + trans_penalty2

    # ==========================================
    # 分阶段动态惩罚权重 (Curriculum Learning)
    # ==========================================
    # 假设总训练 500 轮：
    if epoch < 100:
        penalty_ramp = 0.001   # 前 100 轮：几乎关闭物理限制，让它专心降轨迹 Loss
    elif epoch < 300:
        penalty_ramp = 0.001 + 0.999 * ((epoch - 100) / 200.0)  # 100~300 轮：惩罚项像温水煮青蛙一样慢慢加码
    else:
        penalty_ramp = 1.0     # 300 轮以后：施加 100% 严苛约束，锁死最终安全边界

    # 严格套用师兄的评价体系公式
    w1, w2, w3 = 0.7, 0.2, 0.1
    total_loss = w1 * E_foot + w2 * E_knee + w3 * E_ankle + (penalty_ramp * Penalty)

    return total_loss

def enforce_physical_bounds(model):
    with torch.no_grad():
        model.L_raw[1].clamp_(-5.0, -1.5)  # 锁死曲柄尺寸
        if hasattr(model, 'X_fix'):
            model.X_fix.clamp_(0.5, 1.5)

def de_objective(x, model, theta2_seq, target_x, target_y):
    with torch.no_grad():
        model.L_raw.copy_(torch.tensor(x, dtype=torch.float32))
        enforce_physical_bounds(model)
        pred_x, pred_y, pred_angles = model(theta2_seq)
        loss = calculate_loss(model, pred_x, pred_y, target_x, target_y, pred_angles)
        return loss.item()

def plot_final_results(loss_history, pred_x, pred_y, target_x, target_y):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(target_x, target_y, 'g--', label='Target Gait', linewidth=2)
    plt.plot(pred_x, pred_y, 'b-', label='Optimized Stephenson III', linewidth=2)
    plt.scatter([0], [0], color='red', marker='x', s=100, label='Hip Joint')
    plt.title("Phase 3: Final Result")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.axis('equal')

    plt.subplot(1, 2, 2)
    plt.plot(loss_history, 'k-', linewidth=1.5)
    plt.title("Adam Convergence")
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss")
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    plt.show()


def run_optimization():
    theta2_seq, target_x, target_y = load_target_data('target_gait.csv')
    model = StephensonIII()

    print("阶段 A：启动差分进化算法，全局搜索最优初值...")
    bounds = [
        (0.5, 1.5), (0.1, 0.5), (0.5, 2.0), (0.5, 2.0),
        (1.0, 3.0), (0.5, 2.0), (0.5, 2.0), (0.1, 1.5)
    ]

    # 恢复稳定的 DE 参数，不再降维
    result = differential_evolution(
        de_objective, bounds, args=(model, theta2_seq, target_x, target_y),
        maxiter=15, popsize=5, disp=True, workers=1, updating='immediate',
        mutation=(0.5, 1.0), recombination=0.7, polish=False
    )
    print(f"\n 进化搜索完成！初始 Loss: {result.fun:.4f}")

    # 存档功能保留
    np.savetxt("phaseA_best_params.txt", result.x, delimiter=",", fmt="%.6f")
    print("💾 [保险生效] 阶段A参数已备份。")

    print("阶段 B：切换为 Adam 优化器，开始局部平滑精雕...")
    # 把 DE 算出来的值赋给模型
    with torch.no_grad():
        model.L_raw.copy_(torch.tensor(result.x, dtype=torch.float32))

    # ==========================================
    # 优化器设置 (在 no_grad 之外)
    # ==========================================
    # 步长缩小到 0.0005，训练轮数拉长到 1000 轮
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

    epochs = 1000  # 锁定 1000 轮，给动态权重留足时间
    loss_history = []
    best_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred_x, pred_y, pred_angles = model(theta2_seq)

        # 🚨 极其关键：必须传入 epoch=epoch，才能触发动态权重 (Curriculum Learning)
        loss = calculate_loss(model, pred_x, pred_y, target_x, target_y, pred_angles, epoch=epoch)

        loss.backward()
        optimizer.step()
        scheduler.step()

        enforce_physical_bounds(model)
        loss_val = loss.item()
        loss_history.append(loss_val)

        # 记录最佳状态
        if loss_val < best_loss and not np.isnan(loss_val):
            best_loss = loss_val
            best_model_state = copy.deepcopy(model.state_dict())

        # 每 50 轮打印一次日志
        if epoch % 50 == 0:
            actual_L = model.get_actual_lengths()
            print(f"Adam Epoch {epoch:03d} | Loss: {loss_val:.4f} | L2(曲柄): {actual_L[1].item():.3f}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n 已恢复巅峰状态，最终最佳 Loss: {best_loss:.4f}")

    with torch.no_grad():
        final_x, final_y, final_angles = model(theta2_seq)

    # 绘图函数已按要求注释掉，保证直接跑完出数据
    # plot_final_results(loss_history, final_x.numpy(), final_y.numpy(), target_x.numpy(), target_y.numpy())

    return loss_history, model

if __name__ == "__main__":
    run_optimization()