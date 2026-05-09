import matplotlib.pyplot as plt
import torch
import numpy as np
from train_evolution import run_optimization


def print_optimized_params(model):
    """
    自动提取并格式化打印 Stephenson III 的最优结构参数
    """
    print("\n" + "=" * 50)
    print(" 最优 Stephenson III 机构参数与建模表 ")
    print("=" * 50)

    # 从模型中获取真实杆长 (经过 softplus 处理后的值)
    L = model.get_actual_lengths().detach().numpy()
    X_fix = model.X_fix.item()
    X0 = model.X0.item()
    Y0 = model.Y0.item()

    print("【1. 连杆尺寸参数】")
    print(f"  L1 (四杆机架间距): {L[0]:.2f} mm")
    print(f"  L2 (驱动曲柄):     {L[1]:.2f} mm")
    print(f"  L3 (浮动连杆1):    {L[2]:.2f} mm")
    print(f"  L4 (摇杆1):        {L[3]:.2f} mm")
    print(f"  L5 (小腿延长杆):   {L[4]:.2f} mm")
    print(f"  L6 (足端连接杆):   {L[5]:.2f} mm")
    print(f"  L7 (摇杆2):        {L[6]:.2f} mm")
    print(f"  Alpha(内部夹角):   {L[7]:.2f} rad ({(L[7] * 180 / np.pi):.1f}°)")

    print("\n【2. 机架固定铰接点坐标 (SolidWorks 建基准点用)】")
    print(f"  曲柄旋转原点 (Joint A): ({X0:.2f}, {Y0:.2f})")
    print(f"  四杆右侧铰点 (Joint D): ({X0 + L[0]:.2f}, {Y0:.2f})")
    print(f"  外侧独立铰点 (Joint E): ({X0 + X_fix:.2f}, {Y0:.2f})")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    print("=== Stephenson III 仿生腿寻优工程启动 ===")

    # 注意：这里需要确保你的 run_optimization() 除了返回 losses，还能返回最终的最佳模型 best_model
    # 例如：losses, best_model = run_optimization()
    losses, best_model = run_optimization()

    # 1. 打印最终提取到的最佳尺寸图纸
    if best_model is not None:
        print_optimized_params(best_model)

    # 2. 画出 Loss 曲线
    if losses:
        plt.figure(figsize=(8, 5))
        plt.plot(losses)
        plt.title("Optimization Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.show()

    print("程序执行完毕。")