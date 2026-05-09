import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class StephensonDataset:
    def __init__(self, file_path, n_steps=100):
        """
        n_steps: 模型前向传播的点数，默认100
        """
        self.file_path = file_path
        self.n_steps = n_steps

        # 定义归一化器 (Min-Max 缩放到 [-1, 1])
        self.scaler_foot = MinMaxScaler(feature_range=(-1, 1))
        self.scaler_angle = MinMaxScaler(feature_range=(-1, 1))

        # 存储加载后的 Tensor
        self.target_foot = None  # [100, 2]
        self.target_angle = None  # [100, 2]
        self.raw_foot = None
        self.raw_angle = None

    def load_and_preprocess(self):
        print(f"[*] 正在从独立文件加载目标轨迹: {self.file_path}")

        try:
            # 1. 读取 Excel (默认读取第一个 Sheet)
            df = pd.read_excel(self.file_path)

            # 2. 如果 Excel 行数不是 100，进行线性插值对齐
            # 确保目标点数与模型输出的 n_steps 一致
            if len(df) != self.n_steps:
                print(f"[!] 警告: Excel 数据量({len(df)})与模型步数({self.n_steps})不符，正在进行重采样...")
                df_indices = np.linspace(0, len(df) - 1, self.n_steps)
                df = df.iloc[df_indices].reset_index(drop=True)

            # 3. 提取特征列
            # 假设列名: Foot_X, Foot_Y, Knee_Angle, Ankle_Angle
            foot_data = df[['Foot_X', 'Foot_Y']].values.astype(np.float32)
            angle_data = df[['Knee_Angle', 'Ankle_Angle']].values.astype(np.float32)

            self.raw_foot = torch.from_numpy(foot_data)
            self.raw_angle = torch.from_numpy(angle_data)

            # 4. 执行归一化 (让 Loss 计算时轨迹和角度的权重更平衡)
            foot_norm = self.scaler_foot.fit_transform(foot_data)
            angle_norm = self.scaler_angle.fit_transform(angle_data)

            self.target_foot = torch.from_numpy(foot_norm)
            self.target_angle = torch.from_numpy(angle_norm)

            print(f"[+] 数据处理完成: 足端轨迹与关节角已重采样为 {self.n_steps} 帧")
            return True

        except Exception as e:
            print(f"[X] 加载失败: {e}")
            return False

    def get_torch_targets(self, device='cpu'):
        """获取可直接用于 Loss 计算的 Tensor"""
        return {
            'foot': self.target_foot.to(device),
            'angle': self.target_angle.to(device)
        }

    def denormalize_foot(self, norm_data):
        """将模型输出的 [-1, 1] 映射回真实的米"""
        if isinstance(norm_data, torch.Tensor):
            norm_data = norm_data.detach().cpu().numpy()
        return self.scaler_foot.inverse_transform(norm_data)