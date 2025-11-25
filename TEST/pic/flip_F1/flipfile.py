import numpy as np
import pandas as pd
import os

# ========== 在这里填入你需要翻转的文件名 ==========
input_files = [
    "hypertext2009_time_20000.csv",
    "InVS15_20000.csv",
    "LySchool_20000.csv",
    "Thiers12_20000.csv",
    "sis_20000.csv",
    "ising_20000.csv"
    # 可以继续添加更多文件
]
# ================================================

# 输入文件所在目录
input_dir = "./flip_data"  # 如果文件在当前目录就用 "./"
# 输出目录
output_dir = os.path.join(input_dir, "flip_time_results")
os.makedirs(output_dir, exist_ok=True)


def flip_binary_data(data, flip_ratio, flip_type="both", seed=None):
    """
    分别对二值时间序列数据的0和1按比例进行随机翻转
    flip_type: "0" 只翻0->1, "1" 只翻1->0, "both" 同时翻转
    """
    if seed is not None:
        np.random.seed(seed)

    noisy_data = data.copy()
    zero_indices = np.argwhere(data == 0)
    one_indices = np.argwhere(data == 1)

    if flip_type in ("0", "both") and len(zero_indices) > 0:
        num_flips_zero = int(len(zero_indices) * flip_ratio)
        if num_flips_zero > 0:
            flip_zero_indices = zero_indices[np.random.choice(len(zero_indices), num_flips_zero, replace=False)]
            for row, col in flip_zero_indices:
                noisy_data[row, col] = 1  # 0 -> 1

    if flip_type in ("1", "both") and len(one_indices) > 0:
        num_flips_one = int(len(one_indices) * flip_ratio)
        if num_flips_one > 0:
            flip_one_indices = one_indices[np.random.choice(len(one_indices), num_flips_one, replace=False)]
            for row, col in flip_one_indices:
                noisy_data[row, col] = 0  # 1 -> 0

    return noisy_data


def process_files_separate(file_list, input_dir, output_dir, seed=None):
    flip_ratios = np.arange(0.05, 0.31, 0.05)  # 0.05 ~ 0.30

    for file_name in file_list:
        input_path = os.path.join(input_dir, file_name)

        # 读取数据
        try:
            data = pd.read_csv(input_path, header=None).values
        except Exception as e:
            print(f"❌ 错误: 无法读取 {input_path}: {e}")
            continue

        # 检查二值性
        unique_values = np.unique(data)
        if not set(unique_values).issubset({0, 1}):
            print(f"⚠️ 警告: {file_name} 包含非二值元素！")

        base_name, ext = os.path.splitext(file_name)
        network_name = base_name.split("_")[0]  # 提取网络名作为文件前缀

        # 循环翻转并保存结果（分别翻0或翻1）
        for r in flip_ratios:
            for flip_type in ["0", "1"]:
                noisy_data = flip_binary_data(data, r, flip_type=flip_type, seed=seed)

                ratio_str = str(int(r * 100))  # 比例转整数，例如5, 10, 15...
                output_name = f"{network_name}_flip_{flip_type}_{ratio_str}{ext}"
                output_path = os.path.join(output_dir, output_name)

                pd.DataFrame(noisy_data).to_csv(output_path, index=False, header=False)
                print(f"✅ 已保存: {output_path}")


if __name__ == "__main__":
    process_files_separate(input_files, input_dir, output_dir, seed=42)
