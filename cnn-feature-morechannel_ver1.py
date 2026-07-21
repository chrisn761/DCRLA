import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 配置参数
class Config:
    # 数据参数
    TARGET_SAMPLING_RATE = 30  # TIM acquisition rate (Hz)
    TARGET_DURATION = 10  # 目标持续时间 (秒)
    NUM_TIMESTEPS = TARGET_SAMPLING_RATE * TARGET_DURATION  # 300 time points

    # CNN参数 - 可调整这些参数以改变输出特征维度
    KERNEL_SIZE1 = 20
    KERNEL_SIZE2 = 5
    POOL_SIZE = 2
    FILTERS1 = 32  # 增加滤波器数量以获取更多特征
    FILTERS2 = 64  # 增加滤波器数量以获取更多特征
    FILTERS3 = 32  # 增加滤波器数量以获取更多特征
    FILTERS4 = 64  # 增加滤波器数量以获取更多特征
    OUTPUT_CHANNELS = 60  # 最终输出特征通道数（可根据需要调整）
    DROPOUT_RATE = 0.2


# 数据预处理函数
def preprocess_force_data(force_data):
    """
    Preserve the complete trial and standardize it to 10 s at 30 Hz.
    :param force_data: 原始捏力数据 (1D数组)
    :return: 预处理后的捏力数据 (长度为Config.NUM_TIMESTEPS)
    """
    force_data = np.asarray(force_data, dtype=np.float32)
    if len(force_data) >= Config.NUM_TIMESTEPS:
        return force_data[:Config.NUM_TIMESTEPS]
    return np.pad(force_data, (0, Config.NUM_TIMESTEPS - len(force_data)), mode='constant')


# 自定义数据集类
class ForceDataset(Dataset):
    def __init__(self, force_data):
        """
        初始化数据集
        :param force_data: 捏力数据 (num_samples, num_timesteps)
        """
        # 转换为PyTorch张量并添加通道维度
        self.force_data = torch.tensor(force_data, dtype=torch.float32).unsqueeze(2)  # (N, T, 1)

    def __len__(self):
        return len(self.force_data)

    def __getitem__(self, idx):
        return self.force_data[idx]


# CNN特征提取模型 - 修改为输出更多通道
class MultiCNNFeatureExtractor(nn.Module):
    def __init__(self, output_channels=Config.OUTPUT_CHANNELS):
        """
        初始化特征提取器
        :param output_channels: 输出特征的通道数（默认为60）
        """
        super(MultiCNNFeatureExtractor, self).__init__()
        self.output_channels = output_channels

        # 浅层特征提取 - 增加通道数
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

        # 深层特征提取 - 增加通道数
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

        # 特征融合层 - 使用拼接而不是乘法以保留更多信息
        self.feature_fusion = nn.Sequential(
            nn.Conv1d(Config.FILTERS2 + Config.FILTERS4, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Conv1d(128, output_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(output_channels)
        )

        # 自适应池化层确保输出时间步数为50
        self.adaptive_pool = nn.AdaptiveAvgPool1d(Config.NUM_TIMESTEPS // Config.POOL_SIZE)

    def forward(self, x):
        # 输入形状: (batch_size, timesteps, 1)
        # 转换为PyTorch Conv1d期望的形状: (batch_size, channels, timesteps)
        x = x.permute(0, 2, 1)

        # 提取浅层特征
        encoder1_output = self.encoder1(x)

        # 提取深层特征
        encoder2_output = self.encoder2(x)

        # 在通道维度拼接特征
        concatenated = torch.cat((encoder1_output, encoder2_output), dim=1)

        # 融合特征
        fused_features = self.feature_fusion(concatenated)

        # 应用自适应池化确保时间步数为50
        fused_features = self.adaptive_pool(fused_features)

        # 转换回 (batch_size, time_steps, channels) 形状
        fused_features = fused_features.permute(0, 2, 1)

        return fused_features


def extract_features(model, data_loader, device):
    """使用训练好的模型提取特征"""
    model.eval()
    all_features = []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            features = model(batch)
            all_features.append(features.cpu().numpy())

    return np.vstack(all_features)


def load_and_preprocess_data():
    """加载并预处理数据"""
    # 1. 加载脑卒中受试者捏力数据
    stroke_force = pd.read_excel(r"E:\111研究生\论文\论文2——\数据\Input\患者原始数据10秒.xlsx", index_col=0)
    # 2. 加载健康人捏力数据
    normal_force = pd.read_excel(r"E:\111研究生\论文\论文2——\数据\Input\正常人的原始数据10秒.xlsx", index_col=0)

    # 预处理捏力数据
    processed_force = []

    print("开始预处理捏力数据...")

    # 处理脑卒中受试者数据 (10Hz, 10-24秒)
    for i, col in enumerate(stroke_force.columns[0:21]):  # 第2列到第21列
        original_data = stroke_force[col].dropna().values  # 删除NaN值
        if len(original_data) == 0:
            print(f"警告: 脑卒中受试者 {i + 1} 无有效数据")
            continue

        original_duration = len(original_data) / Config.TARGET_SAMPLING_RATE
        preprocessed = preprocess_force_data(original_data)
        processed_force.append(preprocessed)
        print(f"处理脑卒中受试者 {i + 1}: 原始长度={len(original_data)}, 持续时间={original_duration:.2f}秒")

    # 处理健康人数据 (30Hz, 10-15秒)
    for i, col in enumerate(normal_force.columns[0:21]):  # 第2列到第21列
        original_data = normal_force[col].dropna().values  # 删除NaN值
        if len(original_data) == 0:
            print(f"警告: 健康受试者 {i + 1} 无有效数据")
            continue

        original_duration = len(original_data) / Config.TARGET_SAMPLING_RATE
        preprocessed = preprocess_force_data(original_data)
        processed_force.append(preprocessed)
        print(f"处理健康受试者 {i + 1}: 原始长度={len(original_data)}, 持续时间={original_duration:.2f}秒")

    # 转换为numpy数组
    force_array = np.array(processed_force)

    # 打印形状确认
    print(f"捏力数据形状: {force_array.shape} (样本数, 时间步数)")

    return force_array


if __name__ == "__main__":
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 1. 加载并预处理数据
    force_data = load_and_preprocess_data()

    # 2. 创建数据集和数据加载器
    dataset = ForceDataset(force_data)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    # 3. 初始化模型 - 可指定输出通道数
    output_channels = 60  # 可以修改这个值来改变输出通道数
    model = MultiCNNFeatureExtractor(output_channels=output_channels).to(device)

    # 打印模型结构
    print("特征提取器结构:")
    print(model)

    # 4. 提取特征
    fused_features = extract_features(model, dataloader, device)

    # 5. 打印特征维度
    print(f"\n提取特征维度: {fused_features.shape}")
    print(f"- 样本数: {fused_features.shape[0]}")
    print(f"- 时间步数: {fused_features.shape[1]}")
    print(f"- 特征维度: {fused_features.shape[2]}")

    # 6. 保存特征供后续使用
    output_dir = r"E:\111研究生\论文\论文2——\数据\Output"
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, 'standardized_force_10s.npy'), force_data[:, :, np.newaxis])

    # 保存特征和形状信息
    np.save(os.path.join(output_dir, f'fused_features_{output_channels}ch.npy'), fused_features)

    # 保存形状信息
    with open(os.path.join(output_dir, 'feature_shape_info.txt'), 'w') as f:
        f.write(f"样本数: {fused_features.shape[0]}\n")
        f.write(f"时间步数: {fused_features.shape[1]}\n")
        f.write(f"特征维度: {fused_features.shape[2]}\n")
        f.write(f"总特征数: {fused_features.shape[1] * fused_features.shape[2]}\n")

    print(f"特征已保存至: {os.path.join(output_dir, f'fused_features_{output_channels}ch.npy')}")

    # 7. 可视化特征提取过程
    plt.figure(figsize=(15, 10))

    # 随机选择一个样本进行可视化
    sample_idx = np.random.randint(0, fused_features.shape[0])

    # 原始捏力数据
    plt.subplot(3, 1, 1)
    plt.plot(force_data[sample_idx])
    plt.title(f"样本 {sample_idx} - 原始捏力数据")
    plt.xlabel("时间步")
    plt.ylabel("捏力值")

    # 提取的特征（前10个通道）
    plt.subplot(3, 1, 2)
    for i in range(min(10, output_channels)):
        plt.plot(fused_features[sample_idx, :, i], alpha=0.7, label=f"通道 {i + 1}")
    plt.title(f"样本 {sample_idx} - 提取特征 (前10个通道)")
    plt.xlabel("时间步")
    plt.ylabel("特征值")
    plt.legend()

    # 所有通道的平均特征
    plt.subplot(3, 1, 3)
    mean_features = np.mean(fused_features[sample_idx], axis=1)
    plt.plot(mean_features)
    plt.title(f"样本 {sample_idx} - 所有通道平均特征")
    plt.xlabel("时间步")
    plt.ylabel("特征值")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_visualization.png"))
    plt.show()

    print("特征提取完成并已保存!")
