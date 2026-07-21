import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # 改为后台绘图避免弹窗卡死
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# --- 配置参数 ---
class Config:
    # 核心修改点 1：适配 3 秒时长
    TARGET_SAMPLING_RATE = 30
    TARGET_DURATION = 3  # 改为 3 秒
    NUM_TIMESTEPS = TARGET_SAMPLING_RATE * TARGET_DURATION  # 90 time points

    # CNN参数 (坚决不改，保持和 10 秒模型完全一致的架构)
    KERNEL_SIZE1 = 20
    KERNEL_SIZE2 = 5
    POOL_SIZE = 2
    FILTERS1 = 32
    FILTERS2 = 64
    FILTERS3 = 32
    FILTERS4 = 64
    OUTPUT_CHANNELS = 60
    DROPOUT_RATE = 0.2

    # 文件路径
    INPUT_FILE = r"E:\msb论文模型\数据\Processed_3s_32Subjects.xlsx"
    OUTPUT_DIR = r"E:\msb论文模型\数据\Output"


# 自定义数据集类
class ForceDataset(Dataset):
    def __init__(self, force_data):
        # 转换为PyTorch张量并添加通道维度
        self.force_data = torch.tensor(force_data, dtype=torch.float32).unsqueeze(2)  # (N, T, 1)

    def __len__(self):
        return len(self.force_data)

    def __getitem__(self, idx):
        return self.force_data[idx]


# CNN特征提取模型 (架构与原版保持 100% 一致)
class MultiCNNFeatureExtractor(nn.Module):
    def __init__(self, output_channels=Config.OUTPUT_CHANNELS):
        super(MultiCNNFeatureExtractor, self).__init__()
        self.output_channels = output_channels

        self.encoder1 = nn.Sequential(
            nn.Conv1d(1, Config.FILTERS1, kernel_size=Config.KERNEL_SIZE1, padding=Config.KERNEL_SIZE1 // 2),
            nn.ReLU(),
            nn.BatchNorm1d(Config.FILTERS1),
            nn.Conv1d(Config.FILTERS1, Config.FILTERS2, kernel_size=Config.KERNEL_SIZE2,
                      padding=Config.KERNEL_SIZE2 // 2),
            nn.ReLU(),
            nn.BatchNorm1d(Config.FILTERS2),
            nn.MaxPool1d(Config.POOL_SIZE)
        )

        self.encoder2 = nn.Sequential(
            nn.Conv1d(1, Config.FILTERS3, kernel_size=Config.KERNEL_SIZE1, padding=Config.KERNEL_SIZE1 // 2),
            nn.ReLU(),
            nn.BatchNorm1d(Config.FILTERS3),
            nn.MaxPool1d(Config.POOL_SIZE),
            nn.Conv1d(Config.FILTERS3, Config.FILTERS4, kernel_size=Config.KERNEL_SIZE2,
                      padding=Config.KERNEL_SIZE2 // 2),
            nn.ReLU(),
            nn.BatchNorm1d(Config.FILTERS4),
            nn.Conv1d(Config.FILTERS4, Config.FILTERS4, kernel_size=Config.KERNEL_SIZE2,
                      padding=Config.KERNEL_SIZE2 // 2),
            nn.ReLU(),
            nn.BatchNorm1d(Config.FILTERS4),
            nn.Conv1d(Config.FILTERS4, Config.FILTERS4, kernel_size=Config.KERNEL_SIZE2,
                      padding=Config.KERNEL_SIZE2 // 2),
            nn.ReLU(),
            nn.BatchNorm1d(Config.FILTERS4)
        )

        self.feature_fusion = nn.Sequential(
            nn.Conv1d(Config.FILTERS2 + Config.FILTERS4, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Conv1d(128, output_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(output_channels)
        )

        # 自适应池化层确保输出时间步数动态适配 (这里会自动变成 30 // 2 = 15)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(Config.NUM_TIMESTEPS // Config.POOL_SIZE)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        encoder1_output = self.encoder1(x)
        encoder2_output = self.encoder2(x)
        concatenated = torch.cat((encoder1_output, encoder2_output), dim=1)
        fused_features = self.feature_fusion(concatenated)
        fused_features = self.adaptive_pool(fused_features)
        fused_features = fused_features.permute(0, 2, 1)
        return fused_features


def extract_features(model, data_loader, device):
    model.eval()
    all_features = []
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            features = model(batch)
            all_features.append(features.cpu().numpy())
    return np.vstack(all_features)


def load_and_preprocess_data():
    """核心修改点 2：极其清爽的数据加载，因为数据已经被处理得很完美了"""
    print("正在加载已经对齐好的 32 人 3 秒数据...")
    df = pd.read_excel(Config.INPUT_FILE)

    # 提取除 'Time' 列之外的所有受试者数据 (一共32列)
    subject_cols = [col for col in df.columns if col != 'Time']
    processed_force = []

    for col in subject_cols:
        data = df[col].values
        # 确保正好是 30 个点
        if len(data) >= Config.NUM_TIMESTEPS:
            data = data[:Config.NUM_TIMESTEPS]
        else:
            data = np.pad(data, (0, Config.NUM_TIMESTEPS - len(data)), 'constant')
        processed_force.append(data)

    force_array = np.array(processed_force)
    print(f"捏力数据加载完毕！最终形状: {force_array.shape} (样本数, 时间步数)")
    return force_array


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 1. 加载数据
    force_data = load_and_preprocess_data()

    # 2. 数据加载器
    dataset = ForceDataset(force_data)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    # 3. 初始化模型
    model = MultiCNNFeatureExtractor().to(device)

    # 4. 提取特征
    fused_features = extract_features(model, dataloader, device)

    # 5. 打印并验证维度 (应该是 32样本 x 15时间步 x 60通道)
    print(f"\n提取特征维度: {fused_features.shape}")

    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    np.save(os.path.join(Config.OUTPUT_DIR, 'standardized_force_3s_32sub.npy'),
            force_data[:, :, np.newaxis])

    # 核心修改点 3：另存为一个新的名字，不要覆盖原 40 人的特征！
    output_filename = os.path.join(Config.OUTPUT_DIR, f'fused_features_32sub_3s_{Config.OUTPUT_CHANNELS}ch.npy')
    np.save(output_filename, fused_features)

    print(f"\n🎉 完美！32 人 (3秒) 的高维特征已保存至: {output_filename}")
