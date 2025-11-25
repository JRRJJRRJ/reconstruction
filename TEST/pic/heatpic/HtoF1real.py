import numpy as np
import matplotlib.pyplot as plt
import os
from GetF1test import get_f1


def plot_heatmap(network: str, T_values: list, W_values: list, true_file: str, save_path: str = None):
    """
    绘制 F1 分数的热力图
    """
    base_dir = r"J:\pycharm\MyProject\HighOrder\TEST\RealNet_result"

    # 设置全局字体
    plt.rcParams['font.sans-serif'] = ['Arial']  # 高水平论文常用字体
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['axes.linewidth'] = 1.5  # 加粗坐标轴

    # 创建存放结果的矩阵，行=T，列=W
    F1_matrix = np.zeros((len(T_values), len(W_values)))

    for i, T in enumerate(T_values):
        for j, W in enumerate(W_values):
            folder = os.path.join(base_dir, network, f"T={T}")
            filename = f"{network}_T{T}_W{W}.txt"
            pred_file = os.path.join(folder, filename)
            print("Looking for:", pred_file)

            if os.path.exists(pred_file):
                f1 = get_f1(pred_file, true_file)
                F1_matrix[i, j] = f1
                print(f"{network}: T={T}, W={W}, F1={f1:.4f}")
            else:
                print(f"缺失文件: {pred_file}")
                F1_matrix[i, j] = np.nan

    # 绘制热力图
    plt.figure(figsize=(9, 7))
    im = plt.imshow(
        F1_matrix,
        aspect='auto',  # 保持矩形格子
        origin='lower',
        cmap='YlOrBr',  # 柔和的黄橙棕色，避免深红
        interpolation='none',  # 更锐利
        extent=[min(W_values), max(W_values), min(T_values), max(T_values)],
        vmin=0, vmax=1
    )

    # 颜色条（右边纵坐标 → F1）- 放大刻度线和刻度值
    cbar = plt.colorbar(im)
    cbar.set_label("F1", fontsize=22)  # 放大F1标签
    cbar.ax.tick_params(labelsize=18, length=12, width=1.5, direction='in')  # 放大刻度线和刻度值

    # 确保颜色条刻度与主图一致
    cbar.ax.tick_params(which='major', length=12, width=1.5)

    # 坐标轴与标题 - 放大
    plt.xlabel("W", fontsize=22)  # 放大横坐标标签
    plt.ylabel("T", fontsize=22)  # 放大左纵坐标标签
    plt.title(f"{network}", fontsize=26, weight='bold', pad=20)  # 放大标题

    # 放大坐标轴刻度线和刻度值
    plt.tick_params(axis='both', which='major', labelsize=18, length=12, width=1.5, direction='in')
    plt.tick_params(axis='both', which='minor', length=6, width=1.2, direction='in')

    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')  # dpi 提高锐利度
        print(f"热力图已保存到 {save_path}")
        pdf_path = os.path.splitext(save_path)[0] + ".pdf"
        plt.savefig(pdf_path, dpi=600, bbox_inches='tight')
        print(f"热力图已保存到 {pdf_path}")
    else:
        plt.show()


if __name__ == "__main__":
    network = "Hypertext2009" # 可选 "hypertext2009", "InVS15", "LyonSchool","Thiers12"
    T_values = list(range(1000, 20001, 1000))
    W_values = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    true_file = "hy.txt"

    plot_heatmap(network, T_values, W_values, true_file, save_path="F1_heatmap_hy.png")