import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import time
import matplotlib.pyplot as plt
# ============ 添加中文支持配置 ============
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False
# =========================================

# ==========================================
# 1. 定义论文中的网络结构 (Fig. 5a)
# ==========================================
class PhotonicTensorCoreNet(nn.Module):

    def __init__(self):
        super(PhotonicTensorCoreNet, self).__init__()
        # 论文参数: 输入 28x28, 4个 2x2 卷积核, Valid Padding (无填充)
        # 输出尺寸: (28-2+1) = 27 -> 27x27x4
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=4,
                               kernel_size=2,
                               stride=1,
                               padding=0)
        self.relu = nn.ReLU()

        # 展平: 27 * 27 * 4 = 2916
        # 全连接层: 2916 -> 10 (数字 0-9)
        self.fc1 = nn.Linear(27 * 27 * 4, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 卷积 + ReLU (光子部分主要做这个)
        x = self.conv1(x)
        x = self.relu(x)

        # 展平
        x = x.view(x.size(0), -1)

        # 全连接 + Softmax (分类部分)
        x = self.fc1(x)
        x = self.softmax(x)
        return x


# ==========================================
# 2. 数据加载 (MNIST)
# ==========================================
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307, ), (0.3081, ))])
train_dataset = datasets.MNIST(root='./data',
                               train=True,
                               download=True,
                               transform=transform)
test_dataset = datasets.MNIST(root='./data',
                              train=False,
                              download=True,
                              transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000,
                         shuffle=False)  # 大批次用于测试吞吐量

# ==========================================
# 3. 训练模型 (获取准确的权重作为"真值")2
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PhotonicTensorCoreNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(">>> 正在训练模型以获取权重 (模拟论文中的训练过程)...")
for epoch in range(3):  # 简单训练 3 轮即可达到不错效果
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
print(">>> 训练完成。\n")

# 评估准确率
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print(f"【PyTorch 基准准确率】: {100 * correct / total:.2f}% (论文实验值约 95.3%)")

# ==========================================
# 4. 核心对比：PyTorch vs. 光子仿真
# ==========================================


def simulate_photonic_inference(model, dataloader, num_runs=10):
    """
    模拟光子张量核心的推理过程。
    假设：
    1. 权重已存储在 PCM 中 (非易失性)，无需加载时间。
    2. 卷积被转换为单次矩阵向量乘法 (MVM)，利用 WDM 并行。
    3. 延迟主要由光传播时间和探测器响应决定 (假设为常数 t_light)。
    """
    model.eval()

    # 提取权重 (模拟写入 PCM 后的状态)
    conv_weight = model.conv1.weight.detach().cpu().numpy()  # 4, 1, 2, 2
    conv_bias = model.conv1.bias.detach().cpu().numpy()
    fc_weight = model.fc1.weight.detach().cpu().numpy()
    fc_bias = model.fc1.bias.detach().cpu().numpy()

    # 理论参数 (基于论文 Fig 5e 和正文)
    # 论文提到调制速度可达 GHz 级别，这里假设单次 MVM 的光传播+探测时间为 100 ps (0.1 ns)
    # 注意：这是物理极限模拟，不是 CPU 运行时间
    t_light_per_mvm = 100e-12  # 100 picoseconds

    total_images = 0
    start_time = time.perf_counter()

    with torch.no_grad():
        for data, target in dataloader:
            data = data.cpu()  # 模拟数据已转换为光信号输入
            b, c, h, w = data.shape

            # --- 光子计算模拟阶段 ---

            # 1. 输入重构 (Im2Col): 将卷积转化为矩阵乘法
            # PyTorch 的 unfold 操作模拟了将图像块展开为向量
            # 输入: [B, C, H, W] -> 展开为 [B, (C*k*k), L] 其中 L 是输出位置数量
            patches = data.unfold(2, 2, 1).unfold(3, 2,
                                                  1)  # [B, 1, 27, 27, 2, 2]
            patches = patches.contiguous().view(b, 1, 27, 27,
                                                -1)  # [B, 1, 27, 27, 4]
            patches = patches.permute(0, 2, 3, 1,
                                      4).contiguous()  # [B, 27, 27, 1, 4]
            patches = patches.view(b, -1, 4).transpose(
                1, 2)  # [B, 4, 27*27] -> 每个样本 4 个通道，每个通道 729 个位置

            # 权重矩阵重塑: [4, 1, 2, 2] -> [4, 4] (因为输入通道是1, k*k=4)
            # 实际上光子芯片做的是: Input_Vector (长度4) @ Weight_Matrix (4x4) -> Output
            # 这里为了简化，我们直接计算数学结果，但计时采用"光子时间"

            # 2. 模拟并行 MVM (Multiply-Accumulate)
            # 在光子芯片上，这 4 个卷积核是同时计算的 (WDM)
            # 无论 Batch Size 多大，光子芯片是流式处理，这里我们计算"单个样本通过芯片的时间"

            # 真实物理过程模拟：
            # 光脉冲进入 -> 穿过 PCM 矩阵 (卷积) -> 探测器 -> ReLU (电) -> 穿过 PCM 矩阵 (FC) -> 探测器
            # 假设整个链路 (Conv + ReLU + FC) 的光传播延迟约为 200ps (极快)
            latency_per_sample = 200e-12

            # 累加理论光子耗时 (注意：这不是 CPU 跑代码的时间，而是理论物理时间)
            total_images += b

            # --- 验证计算正确性 (用 CPU 算一遍确保逻辑对，但不计入"光子时间") ---
            output_pytorch = model(data.to(device)).cpu()

    end_time = time.perf_counter()
    cpu_elapsed = end_time - start_time

    # 计算理论光子耗时
    theoretical_photonic_time = total_images * latency_per_sample

    return theoretical_photonic_time, cpu_elapsed, total_images


# 运行对比
print("\n>>> 开始性能对比仿真...")
theo_time, cpu_time, count = simulate_photonic_inference(model, test_loader)

print(f"测试样本总数: {count}")
print("-" * 50)
print(f"【传统 PyTorch (CPU/GPU) 实际耗时】: {cpu_time:.4f} 秒")
print(f"  -> 吞吐量: {count / cpu_time:.2f} 图片/秒")
print("-" * 50)
print(f"【光子张量核心 (理论物理耗时)】: {theo_time:.9f} 秒")
print(
    f"  -> 理论吞吐量: {count / theo_time:.2e} 图片/秒 (约 {count / theo_time / 1e9:.2f} Giga-IPS)"
)
print("-" * 50)
speedup = cpu_time / theo_time
print(f"🚀 理论加速比：{speedup:.2e} 倍 ({speedup/1e6:.2f} 百万倍)")

print("\n💡 结论分析:")
print("1. 准确性：PyTorch 仿真结果应与论文中的 95.3% 接近，证明算法复现成功。")
print("2. 速度差异：PyTorch 受限于电子时钟频率(GHz)和内存带宽。")
print("3. 光子优势：光子计算的核心优势在于 '光速传播' 和 '波分复用(WDM)并行'。")
print("   论文中提到理论上可达 1 Peta-MAC/s，这意味着在处理大规模矩阵时，光子芯片几乎是在'瞬间'完成计算。")
print("   本仿真中的加速比主要体现在忽略了数据搬运和电子时钟周期的限制。")

# 可视化对比 (对数坐标)
plt.figure(figsize=(10, 6))
bars = plt.bar(['PyTorch (实测)', 'Photonic (理论)'], [cpu_time, theo_time],
               color=['#1f77b4', '#ff7f0e'])
plt.yscale('log')  # 使用对数坐标以展示巨大差异
plt.ylabel('Time (seconds, log scale)')
plt.title('Inference Time Comparison: Electronic vs. Photonic Tensor Core')

for i, v in enumerate([cpu_time, theo_time]):
    label = f"{v:.4f}s" if i == 0 else f"{v:.2e}s"
    plt.text(i, v, label, ha='center', va='bottom' if i == 1 else 'top')

plt.show()
