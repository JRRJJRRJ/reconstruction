import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from GetF1test import get_f1

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 六个网络的名称及其真实文件映射（请根据实际路径修改真实网络文件名）
network_file_mapping = {
    "Hypertext2009": "hy.txt",
    "Thiers12": "Th.txt",
    "InVS15": "IN.txt",
    "LyonSchool": "ly.txt",
    "SIS": "sis.txt",
    "Ising": "ising.txt"
}

# 翻转比例（0.05 -> 5%）
flip_ratios = [5, 10, 15, 20, 25, 30]

# 存放结果
flip_results = {}

# 基础目录（你存放翻转文件的目录）
base_dir = r"J:\pycharm\MyProject\HighOrder\TEST\pic\flip_F1\flip_data\flip_results"

# 检查真实网络文件是否存在
true_networks = {}
for network_name, file_name in network_file_mapping.items():
    if os.path.exists(file_name):
        true_networks[network_name] = file_name
        print(f"成功找到真实网络文件: {network_name} -> {file_name}")
    else:
        print(f"警告: 未找到真实网络文件 {file_name} 对应网络 {network_name}")

# 遍历每个网络
for network in network_file_mapping.keys():
    if network not in true_networks:
        print(f"跳过网络 {network}，因为没有真实网络文件")
        continue

    flip_results[network] = {"0": [], "1": []}  # 分别存储翻转0和翻转1的结果

    for node_type in ["0", "1"]:  # 0节点翻转 / 1节点翻转
        scores = []

        for ratio in flip_ratios:
            pattern = os.path.join(base_dir, f"{network}_20000flip{node_type}-{ratio}.txt")
            matching_files = glob.glob(pattern)

            if not matching_files:
                print(f"警告: 未找到文件 {pattern}")
                scores.append(0)
                continue

            file_path = matching_files[0]

            try:
                f1_score = get_f1(true_networks[network], file_path)
                scores.append(f1_score)
                print(f"成功计算 {network} flip{node_type}-{ratio}% 的F1分数: {f1_score:.4f}")
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {e}")
                scores.append(0)

        flip_results[network][node_type] = scores

import matplotlib.pyplot as plt
import numpy as np

# 设置风格和字体 - 与之前代码统一
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.linewidth'] = 1.5  # 加粗坐标轴

fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=400)
axes = axes.flatten()

# 创建显示名称映射
display_names = {
    "Hypertext2009": "Hypertext2009",
    "Thiers12": "Thiers12",
    "InVS15": "InVS15",
    "LyonSchool": "LyonSchool",
    "SIS": "ER",  # 将SIS改为ER
    "Ising": "BA"  # 将Ising改为BA
}

for i, network in enumerate(flip_results.keys()):
    ax = axes[i]

    x = [r / 100 for r in flip_ratios]  # 转换为比例 0.05, 0.10, ...

    scores_0 = flip_results[network]["0"]
    scores_1 = flip_results[network]["1"]

    # 绘制两条折线 - 不使用标签，直接去掉图例
    ax.plot(x, scores_1, marker='D', markersize=8, linewidth=2.2,
            color='red')  # 红色，与之前一致
    ax.plot(x, scores_0, marker='s', markersize=7, linewidth=2.0,
            color='blue', linestyle='--')  # 蓝色虚线，与之前一致

    # 使用映射后的显示名称
    display_name = display_names.get(network, network)

    # 设置标题和标签 - 放大字体
    ax.set_title(f'{display_name}', fontsize=22, fontweight='bold')  # 放大标题
    ax.set_xlabel('M', fontsize=22)  # 放大横坐标标签
    ax.set_ylabel('F1', fontsize=22)  # 放大纵坐标标签

    # 设置坐标轴范围
    ax.set_ylim(0, 1.0)
    ax.set_xlim(0.03, 0.32)

    # 设置刻度
    ax.set_xticks([0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    ax.set_yticks(np.arange(0, 1.01, 0.2))

    # 放大刻度参数 - 与之前一致
    ax.tick_params(axis='x', which='major', length=12, width=1.5, direction='in', top=True, labelsize=18)
    ax.tick_params(axis='x', which='minor', length=6, width=1.2, direction='in', top=True)
    ax.tick_params(axis='y', which='major', length=12, width=1.5, direction='in', right=True, labelsize=18)
    ax.tick_params(axis='y', which='minor', length=6, width=1.2, direction='in', right=True)

    # 添加次刻度
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.01))  # 每0.01一个次刻度
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))  # 每0.05一个次刻度

plt.tight_layout()
plt.savefig('flip_comparison2.png', dpi=400, bbox_inches='tight')
plt.savefig('flip_comparison2.pdf', dpi=400, bbox_inches='tight')
plt.show()