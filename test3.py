import random
import numpy as np
from typing import List, Set, Tuple


def generate_noise_triples(file_path: str, m: int, n: int) -> List[Tuple[int, int, int]]:
    """
    生成文件中不存在的三元组

    参数:
        file_path: 包含现有三元组的文件路径
        m: 需要生成的三元组数量
        n: 数值范围上限（所有数都在1到n之间）

    返回:
        包含m个唯一三元组的列表，每个三元组都是文件中不存在的
    """
    # 1. 读取文件中所有现有三元组
    existing_triples = set()
    with open(file_path, 'r') as f:
        for line in f:
            # 跳过空行
            if not line.strip():
                continue
            # 解析每行的三个整数
            a, b, c = map(int, line.strip().split())
            # 添加到集合（使用排序后的元组来忽略顺序）
            existing_triples.add(tuple(sorted((a, b, c))))

    # 2. 计算所有可能的三元组总数
    total_possible = n * (n - 1) * (n - 2) // 6  # 组合数 C(n, 3)
    existing_count = len(existing_triples)

    # 3. 检查是否有足够的新三元组
    if existing_count >= total_possible:
        raise ValueError(f"所有可能的三元组 ({total_possible}) 已存在于文件中")

    if existing_count + m > total_possible:
        raise ValueError(f"只有 {total_possible - existing_count} 个可用三元组，但请求了 {m} 个")

    # 4. 高效生成新三元组
    new_triples = set()
    max_attempts = m * 10  # 最大尝试次数

    # 预生成所有可能的数字
    all_numbers = list(range(1, n + 1))

    for _ in range(max_attempts):
        if len(new_triples) >= m:
            break

        # 随机选择三个不同的数字
        a, b, c = random.sample(all_numbers, 3)
        # 创建排序后的元组（忽略顺序）
        triple = tuple(sorted((a, b, c)))

        # 检查是否已存在
        if triple in existing_triples or triple in new_triples:
            continue

        new_triples.add(triple)

    # 5. 如果未能生成足够的三元组，使用系统方法
    if len(new_triples) < m:
        # 生成所有可能的三元组
        all_possible = set()
        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                for k in range(j + 1, n + 1):
                    all_possible.add((i, j, k))

        # 移除已存在的三元组
        available = all_possible - existing_triples
        # 随机选择所需数量
        new_triples = set(random.sample(list(available), m))

    return list(new_triples)


def save_triples_to_file(triples: List[Tuple[int, int, int]], output_path: str):
    """
    将三元组保存到文件

    参数:
        triples: 三元组列表
        output_path: 输出文件路径
    """
    with open(output_path, 'w') as f:
        for a, b, c in triples:
            f.write(f"{a} {b} {c}\n")


# 使用示例
if __name__ == "__main__":
    # 输入文件路径
    input_file = "hyT1000.txt"
    # 输出文件路径
    output_file = "HY_T1000.txt"
    # 数值范围上限
    n_value = 85
    # 需要生成的三元组数量
    m_value = 500

    try:
        # 生成噪声三元组
        noise_triples = generate_noise_triples(input_file, m_value, n_value)

        # 保存到文件
        save_triples_to_file(noise_triples, output_file)

        print(f"成功生成 {len(noise_triples)} 个三元组并保存到 {output_file}")

    except ValueError as e:
        print(f"错误: {e}")