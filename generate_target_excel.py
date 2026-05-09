import numpy as np
from scipy.interpolate import interp1d


def process_h36m_gait(hip_xy, knee_xy, ankle_xy, foot_xy):
    """
    输入: 形状为 (N, 2) 的 numpy 数组，代表 N 帧的关节坐标 (x, y)
    输出: 形状为 (100, 5) 的优化器目标矩阵
    """
    N = len(hip_xy)


    # 1. 计算参考腿长 L0 (用于无量纲化)
    # 计算大腿长 + 小腿长的平均值，以减小单帧视角带来的透视误差
    thigh_len = np.mean(np.linalg.norm(knee_xy - hip_xy, axis=1))
    calf_len = np.mean(np.linalg.norm(ankle_xy - knee_xy, axis=1))
    L0 = thigh_len + calf_len

    # 2. 计算足端相对轨迹并无量纲化
    # 强制将髋关节作为 (0,0) 原点，并将坐标系等比缩放到 [-1, 1] 左右
    foot_rel_xy = (foot_xy - hip_xy) / L0
    # 将图像坐标系(Y向下)转换为直角坐标系(Y向上)
    foot_rel_xy[:, 1] = -foot_rel_xy[:, 1]
    # 3. 计算关节角度 (核心：使用向量法保证不出现象限反转突变)
    # 定义躯干/杆件向量
    vec_thigh = knee_xy - hip_xy
    vec_calf = ankle_xy - knee_xy
    vec_foot = foot_xy - ankle_xy

    # 使用 arctan2 计算绝对夹角，然后相减得到相对夹角
    knee_angle = np.arctan2(vec_calf[:, 1], vec_calf[:, 0]) - \
                 np.arctan2(vec_thigh[:, 1], vec_thigh[:, 0])

    ankle_angle = np.arctan2(vec_foot[:, 1], vec_foot[:, 0]) - \
                  np.arctan2(vec_calf[:, 1], vec_calf[:, 0])

    # 统一转换到 [-pi, pi] 区间 (极其重要，否则后续优化器算 Loss 会直接爆炸)
    knee_angle = (knee_angle + np.pi) % (2 * np.pi) - np.pi
    ankle_angle = (ankle_angle + np.pi) % (2 * np.pi) - np.pi

    # 4. 组装原始特征矩阵 (N, 4) -> [Foot_X, Foot_Y, Knee, Ankle]
    raw_features = np.column_stack((foot_rel_xy, knee_angle, ankle_angle))

    # 5. 三次样条插值 (Cubic Spline) 重采样至 100 个等距相位点
    old_time = np.linspace(0, 1, N)
    new_time = np.linspace(0, 1, 100)

    interpolator = interp1d(old_time, raw_features, axis=0, kind='cubic')
    resampled_features = interpolator(new_time)

    # 6. 拼装最终矩阵: 添加第一列代表相位 (Phase) [0, 0.01, ..., 0.99]
    phase = new_time.reshape(100, 1)
    target_matrix = np.hstack((phase, resampled_features))

    return target_matrix


# ==========================================
# 数据调用与执行部分
# ==========================================

# 1. 加载 CPN 微调后的 2D 坐标检测文件
data_path = 'data_2d_h36m_cpn_ft_h36m_dbb.npz'
dataset = np.load(data_path, allow_pickle=True)

# 提取核心的 2D 坐标字典
positions_2d = dataset['positions_2d'].item()

# 2. 定位受试者 S1 -> 动作 Walking 1 -> 第一个侧视摄像机 (索引0)
subject = 'S1'
action = 'Walking 1'
camera_idx = 0
video_frames = positions_2d[subject][action][camera_idx]

print(f"成功提取视频序列，共 {len(video_frames)} 帧。")

# 3. 提取右腿的 4 个关键点坐标 (N帧, 17关节, 2维坐标)
# Human3.6M 常用骨架索引: 1=右髋(Pelvis/Right Hip), 2=右膝(Right Knee), 3=右踝(Right Ankle), 4=右足尖(Right Foot)
hip_xy = video_frames[:, 1, :]
knee_xy = video_frames[:, 2, :]
ankle_xy = video_frames[:, 3, :]
foot_xy = video_frames[:, 3, :]

print("右侧髋、膝、踝、足坐标提取完毕，正在生成优化目标...")

# 4. 截取单步态周期 (重要前提：需要你先用 matplotlib 画图观察一下 foot_xy，挑选两个波谷之间的帧索引)
# 假设你观察到第 150 帧到第 220 帧是一个完整的步态周期：
start_frame = 150
end_frame = 220

# 5. 传入处理函数
target_data = process_h36m_gait(
    hip_xy[start_frame:end_frame],
    knee_xy[start_frame:end_frame],
    ankle_xy[start_frame:end_frame],
    foot_xy[start_frame:end_frame]
)
import matplotlib.pyplot as plt

# 画出整个视频的足端 Y 坐标轨迹
#plt.plot(foot_xy[:, 1])
#plt.title("Foot Y Trajectory")
#plt.xlabel("Frame")
#plt.ylabel("Y Pixel")
#plt.grid(True)
#plt.show()

# 运行后，看着弹出的图，找到两个相邻的“波峰”（因为图像Y向下，波峰代表脚最靠下）
# 比如图上显示第 125 帧到 190 帧是一个完整的波，你再把切片改成：
# start_frame = 125
# end_frame = 190
# 6. 保存为 CSV，供后续的 Stephenson III 优化器直接读取
np.savetxt('target_gait.csv', target_data, delimiter=',',
           header='Phase,Foot_X,Foot_Y,Knee_Angle,Ankle_Angle', comments='')



print("清洗完成！已输出为 target_gait.csv。第一阶段宣告结束。")