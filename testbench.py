import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import time
import matplotlib.pyplot as plt
# ============ 添加中文支持配置 ============
plt.rcParams['font.sans-serif'] = [
    'Arial Unicode MS', 'PingFang SC', 'Heiti TC'
]
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
        x = x.view(x.shape[0], -1)

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
# 3. 训练模型 (获取准确的权重作为"真值")
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


def photonic_mvm_forward(data_cpu, conv_weight_cpu, conv_bias_cpu,
                         fc_weight_cpu, fc_bias_cpu):
    """
    用显式 MVM 方式实现 Conv+ReLU+FC+Softmax：
    1) Conv2d(1->4, k=2, s=1, p=0) 通过 unfold + MVM 实现
    2) FC 通过矩阵乘法实现
    """
    b = data_cpu.shape[0]

    # [B, 1, 28, 28] -> [B, 4, 729]，4 = C*k*k，729 = 27*27
    patches = F.unfold(data_cpu, kernel_size=2, stride=1)

    # [4, 1, 2, 2] -> [4, 4]
    conv_w = conv_weight_cpu.view(4, -1)

    # MVM: [4,4] x [B,4,729] -> [B,4,729]
    conv_out = torch.einsum('oc,bcl->bol', conv_w, patches)
    conv_out = conv_out + conv_bias_cpu.view(1, 4, 1)
    conv_out = conv_out.view(b, 4, 27, 27)
    conv_relu = torch.relu(conv_out)

    # FC MVM: [B,2916] x [2916,10] -> [B,10]
    fc_in = conv_relu.reshape(b, -1)
    logits = fc_in @ fc_weight_cpu.t() + fc_bias_cpu
    probs = torch.softmax(logits, dim=1)

    return conv_relu, probs


def simulate_photonic_inference(model, dataloader, num_runs=None):
    """
    模拟光子张量核心的推理过程。
    假设：
    1. 权重已存储在 PCM 中 (非易失性)，无需加载时间。
    2. 卷积被转换为单次矩阵向量乘法 (MVM)，利用 WDM 并行。
    3. 延迟主要由光传播时间和探测器响应决定 (假设为常数 t_light)。
    """
    model.eval()

    # 提取权重 (模拟写入 PCM 后的状态)
    conv_weight = model.conv1.weight.detach().cpu()
    conv_bias = model.conv1.bias.detach().cpu()
    fc_weight = model.fc1.weight.detach().cpu()
    fc_bias = model.fc1.bias.detach().cpu()

    # 理论延迟模型（可按器件参数调整）
    # 说明：按你的要求，光学路径不再只算 MVM，而是补上后续电学步骤延迟。
    timing_model = {
        # 光学 MVM（卷积）延迟
        "t_mvm_optical_conv": 100e-12,  # 100 ps
        # 后续电学链路延迟（示例假设值）
        "t_elec_nonlinearity": 1.0e-9,  # ReLU 等非线性
        "t_elec_fc": 5.0e-9,  # 全连接
        "t_elec_softmax_argmax": 1.0e-9,  # Softmax + 判决
        "t_elec_verify_target": 1.0e-9,  # 与 target 对比
    }

    latency_per_sample_mvm_only = timing_model["t_mvm_optical_conv"]
    latency_per_sample_hybrid = (
        timing_model["t_mvm_optical_conv"] +
        timing_model["t_elec_nonlinearity"] + timing_model["t_elec_fc"] +
        timing_model["t_elec_softmax_argmax"] +
        timing_model["t_elec_verify_target"])

    total_images = 0
    start_time = time.perf_counter()
    electronic_elapsed = 0.0

    # 正确性验证统计
    conv_max_abs_err = 0.0
    conv_sum_abs_err = 0.0
    conv_numel = 0

    out_max_abs_err = 0.0
    out_sum_abs_err = 0.0
    out_numel = 0

    pred_match = 0
    mvm_correct_vs_target = 0
    pytorch_correct_vs_target = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            if num_runs is not None and batch_idx >= num_runs:
                break

            data = data.cpu()  # 模拟数据已转换为光信号输入
            b, c, h, w = data.shape

            # --- 光子计算模拟阶段 ---

            # 1) 显式 MVM 前向：Conv(MVM)+ReLU+FC(MVM)+Softmax
            mvm_conv_relu, mvm_output = photonic_mvm_forward(
                data, conv_weight, conv_bias, fc_weight, fc_bias)

            # 累加样本数
            total_images += b

            # 2) PyTorch 基准前向（电学路径时间：前向 + 与target对比）
            data_device = data.to(device)
            t_elec_start = time.perf_counter()
            output_pytorch = model(data_device)
            pytorch_pred = output_pytorch.argmax(dim=1)
            pytorch_correct_vs_target += (pytorch_pred == target.to(device)).sum().item()
            electronic_elapsed += time.perf_counter() - t_elec_start

            # 下述为正确性验证开销，不计入电学前向基准时间
            ref_conv_relu = model.relu(model.conv1(data_device)).cpu()
            output_pytorch = output_pytorch.cpu()

            # 3) 误差统计（验证 MVM 正确性）
            conv_abs_err = (mvm_conv_relu - ref_conv_relu).abs()
            out_abs_err = (mvm_output - output_pytorch).abs()

            conv_max_abs_err = max(conv_max_abs_err, conv_abs_err.max().item())
            out_max_abs_err = max(out_max_abs_err, out_abs_err.max().item())

            conv_sum_abs_err += conv_abs_err.sum().item()
            out_sum_abs_err += out_abs_err.sum().item()
            conv_numel += conv_abs_err.numel()
            out_numel += out_abs_err.numel()

            mvm_pred = mvm_output.argmax(dim=1)
            pytorch_pred = output_pytorch.argmax(dim=1)

            pred_match += (mvm_pred == pytorch_pred).sum().item()
            mvm_correct_vs_target += (mvm_pred == target).sum().item()

    end_time = time.perf_counter()
    wall_elapsed = end_time - start_time

    # 计算理论光子耗时
    # 新口径：光学 MVM + 后续电学步骤
    theoretical_photonic_time = total_images * latency_per_sample_hybrid
    theoretical_photonic_time_mvm_only = total_images * latency_per_sample_mvm_only

    validation = {
        "conv_max_abs_err": conv_max_abs_err,
        "conv_mean_abs_err": conv_sum_abs_err / max(conv_numel, 1),
        "out_max_abs_err": out_max_abs_err,
        "out_mean_abs_err": out_sum_abs_err / max(out_numel, 1),
        "pred_match_acc": pred_match / max(total_images, 1),
        "mvm_acc_vs_target": mvm_correct_vs_target / max(total_images, 1),
        "pytorch_acc_vs_target":
        pytorch_correct_vs_target / max(total_images, 1),
        "latency_per_sample_mvm_only_ps": latency_per_sample_mvm_only * 1e12,
        "latency_per_sample_hybrid_ps": latency_per_sample_hybrid * 1e12,
        "theoretical_photonic_time_mvm_only": theoretical_photonic_time_mvm_only,
        "timing_model": timing_model,
        "wall_elapsed_all_checks": wall_elapsed,
        "checked_images": total_images,
    }

    return theoretical_photonic_time, electronic_elapsed, total_images, validation


# 运行对比
print("\n>>> 开始性能对比仿真...")
theo_time, cpu_time, count, val = simulate_photonic_inference(
    model, test_loader)

print(f"测试样本总数: {count}")
print("-" * 50)
print(f"【传统 PyTorch (CPU/GPU) 实际耗时】: {cpu_time:.4f} 秒 (前向+与target对比)")
print(f"  -> 吞吐量: {count / cpu_time:.2f} 图片/秒")
print("-" * 50)
print(f"【光子张量核心 (理论物理耗时, 新口径)】: {theo_time:.9f} 秒")
print(
    f"  -> 理论吞吐量: {count / theo_time:.2e} 图片/秒 (约 {count / theo_time / 1e9:.2f} Giga-IPS)"
)
print(f"  -> 其中 MVM-only 旧口径耗时: {val['theoretical_photonic_time_mvm_only']:.9f} 秒")
print(
    f"  -> 单样本延迟: MVM-only={val['latency_per_sample_mvm_only_ps']:.1f} ps, "
    f"新口径(含后续电学)= {val['latency_per_sample_hybrid_ps']:.1f} ps")
print("-" * 50)
speedup = cpu_time / theo_time
print(f"🚀 理论加速比：{speedup:.2e} 倍 ({speedup/1e6:.2f} 百万倍)")

print("\n>>> MVM 正确性验证结果 (光学MVM vs PyTorch):")
print(f"验证样本数: {val['checked_images']}")
print(f"Conv+ReLU 最大绝对误差: {val['conv_max_abs_err']:.3e}")
print(f"Conv+ReLU 平均绝对误差: {val['conv_mean_abs_err']:.3e}")
print(f"最终输出最大绝对误差: {val['out_max_abs_err']:.3e}")
print(f"最终输出平均绝对误差: {val['out_mean_abs_err']:.3e}")
print(f"MVM 与 target 准确率: {val['mvm_acc_vs_target']*100:.2f}%")
print(f"PyTorch 与 target 准确率: {val['pytorch_acc_vs_target']*100:.2f}%")
print(f"MVM 与 PyTorch 预测一致率: {val['pred_match_acc']*100:.2f}%")

tol_max = 1e-5
tol_mean = 1e-6
passed = (val["out_max_abs_err"] < tol_max
          and val["out_mean_abs_err"] < tol_mean
          and val["pred_match_acc"] == 1.0)
print(f"MVM 正确性判定: {'通过' if passed else '未通过'} "
      f"(阈值: max<{tol_max:.0e}, mean<{tol_mean:.0e}, 一致率=100%)")

print("\n💡 结论分析:")
print("1. 准确性：PyTorch 仿真结果应与论文中的 95.3% 接近，证明算法复现成功。")
print("2. MVM正确性：新增的显式 MVM 验证用于确保光学映射与 PyTorch 数学等价。")
print("3. 速度差异：PyTorch 受限于电子时钟频率(GHz)和内存带宽。")
print("4. 光子优势：光子计算的核心优势在于 '光速传播' 和 '波分复用(WDM)并行'。")
print("   论文中提到理论上可达 1 Peta-MAC/s，这意味着在处理大规模矩阵时，光子芯片几乎是在'瞬间'完成计算。")
print("   本仿真中的加速比主要体现在忽略了数据搬运和电子时钟周期的限制。")

# 可视化对比 (对数坐标)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 子图1：耗时对比 (对数坐标)
labels = ['PyTorch (实测)', 'Photonic (理论)']
times = [cpu_time, theo_time]
colors = ['#1f77b4', '#ff7f0e']

bars1 = ax1.bar(labels, times, color=colors)
ax1.set_yscale('log')
ax1.set_ylabel('推理耗时 (秒, 对数坐标)')
ax1.set_title('电子计算 vs 光子计算 - 耗时对比')

# 添加数值标签
for i, v in enumerate(times):
    label = f"{v:.4f}s" if i == 0 else f"{v:.2e}s"
    ax1.text(i,
             v,
             label,
             ha='center',
             va='bottom' if i == 1 else 'top',
             fontsize=10)

# 子图2：准确率对比
acc_labels = ['PyTorch 准确率', '光子MVM准确率']
acc_values = [
    val['pytorch_acc_vs_target'] * 100, val['mvm_acc_vs_target'] * 100
]
bars2 = ax2.bar(acc_labels, acc_values, color=['#1f77b4', '#ff7f0e'])
ax2.set_ylabel('准确率 (%)')
ax2.set_title('电子计算 vs 光子计算 - 准确率对比')
ax2.set_ylim(90, 100)  # 固定y轴范围，突出微小差异

# 添加数值标签
for i, v in enumerate(acc_values):
    ax2.text(i, v + 0.1, f"{v:.2f}%", ha='center', va='bottom', fontsize=10)

# 整体标题
fig.suptitle('光子张量核心 vs 传统电子计算 - 性能与准确率对比', fontsize=14, fontweight='bold')

# 调整布局
plt.tight_layout()
plt.show()
