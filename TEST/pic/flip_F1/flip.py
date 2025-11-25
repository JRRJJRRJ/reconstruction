import numpy as np
import pandas as pd
import os
import argparse


def flip_binary_data(data, flip_ratio, seed=None):
    """
    向二值时间序列数据中添加随机翻转噪声

    参数:
    data: 二维numpy数组，形状为(时间步数, 节点数)
    flip_ratio: 翻转比例，0到1之间的浮点数
    seed: 随机种子，用于确保结果可重现

    返回:
    添加噪声后的数据
    """
    if seed is not None:
        np.random.seed(seed)

    # 创建数据的副本
    noisy_data = data.copy()

    # 计算需要翻转的元素数量
    total_elements = data.size
    num_flips = int(total_elements * flip_ratio)

    # 随机选择要翻转的位置
    indices = np.random.choice(total_elements, num_flips, replace=False)

    # 将选中的位置进行翻转 (0->1, 1->0)
    for idx in indices:
        row = idx // data.shape[1]
        col = idx % data.shape[1]
        noisy_data[row, col] = 1 - noisy_data[row, col]

    return noisy_data


def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='为二值时间序列数据添加随机翻转噪声')
    parser.add_argument('./RealNet_data/hypertext2009/T=17000/hypertext2009_time_17000.csv', help='输入CSV文件路径')
    parser.add_argument('./TEST/pic/flip_F1', help='输出CSV文件路径')
    parser.add_argument('flip_ratio', type=float, help='翻转比例 (0.0-1.0)')
    parser.add_argument('--seed', type=int, default=None, help='随机种子 (可选)')

    args = parser.parse_args()

    # 读取输入数据
    try:
        data = pd.read_csv(args.input_file, header=None).values
    except Exception as e:
        print(f"错误: 无法读取文件 {args.input_file}: {e}")
        return

    # 检查数据是否为二值数据
    unique_values = np.unique(data)
    if not set(unique_values).issubset({0, 1}):
        print("警告: 输入数据包含非二值元素。确保数据只包含0和1。")

    # 添加噪声
    noisy_data = flip_binary_data(data, args.flip_ratio, args.seed)

    # 保存结果
    try:
        output_dir = os.path.dirname(args.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        pd.DataFrame(noisy_data).to_csv(args.output_file, index=False, header=False)
        print(f"成功: 已添加噪声并保存到 {args.output_file}")
    except Exception as e:
        print(f"错误: 无法保存文件 {args.output_file}: {e}")


if __name__ == "__main__":
    main()