import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from matplotlib.ticker import MultipleLocator
from GetF1test import get_f1

# 设置风格和字体 - 与第一个图统一
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.linewidth'] = 1.5  # 稍微加粗坐标轴

# 六个真实网络的名称及其对应的文件名映射
network_file_mapping = {
    "Hypertext2009": "hy.txt",
    "Thiers12": "Th.txt",
    "InVS15": "IN.txt",
    "LyonSchool": "ly.txt",
    "SIS": "Ising.txt",
    "Ising": "SIS.txt",
}

# 每个网络的最优迭代次数
optimal_iterations = {
    "Hypertext2009": 70,
    "Thiers12": 60,
    "InVS15": 80,
    "LyonSchool": 50,
    "SIS":80,
    "Ising":80
}

# 时间序列长度列表
T_list = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000,
          11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000]

# 其他两种方法的F1分数（示例数据，请替换为您的实际数据）
method1_scores = {
    "Hypertext2009": [0.431, 0.512, 0.53, 0.623, 0.641, 0.7123, 0.7832, 0.775, 0.792,
                      0.874, 0.882, 0.86, 0.875, 0.873, 0.863, 0.865, 0.868,
                      0.872, 0.879, 0.89],
    "Thiers12": [0.441, 0.61, 0.76, 0.87, 0.91, 0.95, 0.93, 0.946, 0.956,
                 0.961, 0.958, 0.962, 0.9624, 0.959, 0.948, 0.972, 0.974,
                 0.977, 0.984, 0.987],
    "InVS15": [0.032, 0.132, 0.245, 0.2745, 0.357, 0.423, 0.443, 0.4624, 0.5105,
               0.532, 0.563, 0.583, 0.604, 0.632, 0.674, 0.693, 0.682,
               0.725, 0.714, 0.743],
    "LyonSchool": [0.026, 0.214, 0.444, 0.5194, 0.546, 0.68, 0.732, 0.775, 0.8132,
                   0.802, 0.857, 0.862, 0.916, 0.8875, 0.935, 0.9135, 0.924,
                   0.9533, 0.9432, 0.968],
    "SIS": [0.10, 0.182, 0.532, 0.7684, 0.8783, 0.923, 0.914, 0.934, 0.932,
                       0.931, 0.933, 0.9234, 0.9315, 0.9345, 0.9289, 0.9343, 0.9324,
                       0.9365, 0.9332, 0.9321],
    "Ising": [0.026, 0.214, 0.444, 0.5194, 0.546, 0.68, 0.732, 0.775, 0.8132,
              0.802, 0.817, 0.822, 0.816, 0.8175, 0.835, 0.8135, 0.824,
              0.8233, 0.8132, 0.8268],
}

method2_scores = {
    "Hypertext2009": [0.249, 0.278, 0.346, 0.423, 0.506, 0.523, 0.589, 0.643, 0.654,
                      0.678, 0.694, 0.718, 0.734, 0.743, 0.763, 0.774, 0.794,
                      0.822, 0.842, 0.862],
    "Thiers12": [0.367, 0.476, 0.588, 0.664, 0.757, 0.7723, 0.794, 0.845, 0.867,
                 0.873, 0.866, 0.879, 0.884, 0.885, 0.897, 0.894, 0.873,
                 0.889, 0.896, 0.897],
    "InVS15": [0.038, 0.104, 0.225, 0.2445, 0.265, 0.294, 0.331, 0.350, 0.382,
               0.425, 0.453, 0.485, 0.539, 0.541, 0.561, 0.582, 0.623,
               0.651, 0.643, 0.647],
    "LyonSchool": [0.076, 0.151, 0.231, 0.342, 0.353, 0.420, 0.513, 0.684, 0.745,
                   0.732, 0.754, 0.783, 0.832, 0.843, 0.856, 0.846, 0.867,
                   0.861, 0.871, 0.872],
    "SIS": [0.016, 0.14, 0.344, 0.4194, 0.546, 0.682, 0.7321, 0.775, 0.8332,
            0.802, 0.847, 0.862, 0.846, 0.8575, 0.8635, 0.8635, 0.8624,
            0.8533, 0.852, 0.868],
    "Ising": [0.06, 0.16, 0.256, 0.34, 0.446, 0.49, 0.632, 0.675, 0.7432,
              0.802, 0.812, 0.812, 0.836, 0.8275, 0.835, 0.8335, 0.834,
              0.833, 0.8292, 0.8368]
}

# 初始化存储结果的字典
your_method_scores = {}

# 检查真实网络文件是否存在
true_networks = {}
for network_name, file_name in network_file_mapping.items():
    if os.path.exists(file_name):
        true_networks[network_name] = file_name
        print(f"成功找到真实网络文件: {network_name} -> {file_name}")
    else:
        print(f"错误: 无法找到真实网络文件 {file_name} 对应网络 {network_name}")

# 对于每个网络，读取不同时间序列长度的结果文件并计算F1分数
for network in network_file_mapping.keys():
    if network not in true_networks:
        print(f"跳过网络 {network}，因为没有真实网络文件")
        continue

    scores = []
    optimal_W = optimal_iterations[network]

    for T in T_list:
        base_dir = r"J:\pycharm\MyProject\HighOrder\TEST\RealNet_result"
        file_pattern = os.path.join(base_dir, network, f"T={T}", f"{network}_T{T}_W{optimal_W}.txt")

        matching_files = glob.glob(file_pattern)

        if not matching_files:
            print(f"警告: 未找到文件 {file_pattern}")
            scores.append(0)
            continue

        file_path = matching_files[0]

        try:
            # 直接传入文件路径给 get_f1
            f1_score = get_f1(true_networks[network], file_path)
            scores.append(f1_score)
            print(f"成功计算 {network} T={T} 的F1分数: {f1_score}")

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            scores.append(0)

    your_method_scores[network] = scores

# 绘制六个子图，每个网络一个 - 2行3列布局
fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=400)
axes = axes.flatten()

# 网络显示名称映射
display_names = {
    "Hypertext2009": "Hypertext2009",
    "Thiers12": "Thiers12",
    "InVS15": "InVS15",
    "LyonSchool": "LyonSchool",
    "SIS": "ER",
    "Ising": "BA"
}

for i, network in enumerate(network_file_mapping.keys()):
    if network not in your_method_scores:
        continue

    ax = axes[i]

    # 获取三种方法的分数
    your_scores = your_method_scores[network]
    m1_scores = method1_scores[network]
    m2_scores = method2_scores[network]

    # 绘制三条折线 - 使用新的格式
    ax.plot(T_list, your_scores, marker='D', markersize=8, linewidth=2.2,
            label='HRCK (Ours)', color='red')  # 红色，与第一个图一致
    ax.plot(T_list, m1_scores, marker='s', markersize=7, linewidth=2.0,
            label='HOR', color='black', linestyle=':')  # 黑色虚线，与第一个图一致
    ax.plot(T_list, m2_scores, marker='^', markersize=7, linewidth=2.0,
            label='DSR', color='brown', linestyle='-.')  # 棕色点划线，与第一个图一致

    # 设置标题和标签 - 使用显示名称
    ax.set_title(f'{display_names[network]}', fontsize=22, fontweight='bold')  # 放大标题
    ax.set_xlabel('T', fontsize=22)  # 放大横坐标标签
    ax.set_ylabel('F1', fontsize=22)  # 放大纵坐标标签

    # 设置横坐标范围
    ax.set_xlim(0, 20000)

    # 横坐标主刻度 - 放大刻度线和刻度值
    x_major_ticks = [0, 5000, 10000, 15000, 20000]
    ax.set_xticks(x_major_ticks)
    ax.tick_params(axis='x', which='major', length=12, width=1.5, direction='in', top=True, labelsize=18)

    # 横坐标次刻度 - 放大刻度线
    ax.xaxis.set_minor_locator(MultipleLocator(1000))  # 每1000一个次刻度
    ax.tick_params(axis='x', which='minor', length=6, width=1.2, direction='in', top=True)

    # 设置纵坐标范围
    ax.set_ylim(0, 1.0)

    # 纵坐标主刻度 - 放大刻度线和刻度值
    y_major_ticks = np.arange(0, 1.01, 0.2)
    ax.set_yticks(y_major_ticks)
    ax.tick_params(axis='y', which='major', length=12, width=1.5, direction='in', right=True, labelsize=18)

    # 纵坐标次刻度 - 放大刻度线
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))  # 每0.05一个次刻度
    ax.tick_params(axis='y', which='minor', length=6, width=1.2, direction='in', right=True)

    # 图例 - 放大字体和标记
    legend = ax.legend(
        loc='lower right',
        fontsize=18,  # 放大图例字体
        frameon=True,
        fancybox=False,
        framealpha=1,
        edgecolor="black",
        bbox_to_anchor=(0.90, 0.12),
        markerscale=1.2  # 放大图例中的标记
    )
    legend.get_frame().set_linewidth(1.5)
    legend.get_frame().set_boxstyle('Square')

# 六个子图间距稍大 - 保持相同的布局设置
plt.tight_layout(h_pad=2.0, w_pad=2.0)

# 保存图片
plt.savefig('comparison_plot2.png', dpi=400, bbox_inches='tight')
pdf_path = 'comparison_plot2.pdf'
plt.savefig(pdf_path, dpi=400, bbox_inches='tight')
print(f"图片已保存为 {pdf_path}")
plt.show()