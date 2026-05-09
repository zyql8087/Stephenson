import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class StephensonDataset:
    def __init__(self, file_path, n_steps=100):
        self.file_path = file_path
        self.n_steps = n_steps
        self.scaler_foot = MinMaxScaler(feature_range=(-1, 1))
        self.scaler_angle = MinMaxScaler(feature_range=(-1, 1))

        self.target_foot = None
        self.target_angle = None
        self.raw_foot = None
        self.raw_angle = None

    def load_and_preprocess(self):
        print(f"[*] 正在从 CSV 加载目标轨迹: {self.file_path}")
        try:
            df = pd.read_csv(self.file_path)
            # 对齐步数
            if len(df) != self.n_steps:
                indices = np.linspace(0, len(df) - 1, self.n_steps).astype(int)
                df = df.iloc[indices].reset_index(drop=True)

            # 提取列 (确保 CSV 中包含这些表头)
            foot_data = df[['Foot_X', 'Foot_Y']].values.astype(np.float32)
            angle_data = df[['Knee_Angle', 'Ankle_Angle']].values.astype(np.float32)

            self.raw_foot = torch.from_numpy(foot_data)
            self.raw_angle = torch.from_numpy(angle_data)

            # 归一化：用于平衡 Loss
            self.target_foot = torch.from_numpy(self.scaler_foot.fit_transform(foot_data))
            self.target_angle = torch.from_numpy(self.scaler_angle.fit_transform(angle_data))

            print(f"[+] 成功加载并对齐 {self.n_steps} 帧数据。")
            return True
        except Exception as e:
            print(f"[X] 加载失败: {e}")
            return False

    def get_torch_targets(self, device='cpu'):
        return {
            'foot': self.target_foot.to(device),
            'angle': self.target_angle.to(device),
            'raw_foot': self.raw_foot.to(device),
            'raw_angle': self.raw_angle.to(device)
        }