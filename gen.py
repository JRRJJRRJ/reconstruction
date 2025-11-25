
import random
import os
from typing import List, Set, Tuple
import math


def load_real_triples(file_path: str) -> Set[Tuple[int, int, int]]:
    """读取真实文档，返回三元组集合（排序后）"""
    triples = set()
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            a, b, c = map(int, parts)
            triples.add(tuple(sorted((a, b, c))))
    return triples


def generate_fake_triples(m: int, real_triples: Set[Tuple[int, int, int]], max_size: int) -> Set[Tuple[int, int, int]]:
    """生成虚假三元组池，数字不超过m，且不与真实三元组重复"""
    fake_pool = set()
    # 生成所有可能的三元组（数字从1到m，三个数字互不相同）
    for a in range(1, m + 1):
        for b in range(a + 1, m + 1):
            for c in range(b + 1, m + 1):
                triple = (a, b, c)
                if triple not in real_triples:
                    fake_pool.add(triple)
    # 如果池子太大，随机采样一个子集，足够使用即可
    if len(fake_pool) > max_size:
        fake_pool = set(random.sample(list(fake_pool), max_size))
    return fake_pool


def calculate_tp(f1: float, sum_val: int, real_size: int) -> int:
    """根据F1、sum和真实数据大小计算TP值"""
    tp = (f1 * (sum_val + real_size)) / 2
    tp = round(tp)
    # 确保TP在合理范围内
    tp = max(0, min(tp, sum_val, real_size))
    return tp


def shuffle_triple(triple: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """打乱三元组的列顺序"""
    a, b, c = triple
    shuffled = random.sample([a, b, c], 3)
    return tuple(shuffled)


def generate_virtual_documents(real_triples_set: Set[Tuple[int, int, int]], fake_pool: Set[Tuple[int, int, int]],
                               f1_list: List[float], sum_list: List[int], output_dir: str):
    """生成虚拟文档并保存到指定目录"""
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    real_list = list(real_triples_set)
    fake_list = list(fake_pool)
    real_size = len(real_list)
    inherited_real = set()  # 继承的真实数据

    for i, (f1, sum_val) in enumerate(zip(f1_list, sum_list)):
        # 计算所需TP值
        tp = calculate_tp(f1, sum_val, real_size)
        # 调整继承的真实数据量
        if tp > len(inherited_real):
            # 需要添加新真实数据
            need_add = tp - len(inherited_real)
            available_real = [t for t in real_list if t not in inherited_real]
            if need_add > len(available_real):
                # 如果可用真实数据不足，则全部添加
                new_real = available_real
            else:
                new_real = random.sample(available_real, need_add)
            inherited_real.update(new_real)
        elif tp < len(inherited_real):
            # 需要减少真实数据
            need_remove = len(inherited_real) - tp
            removed = random.sample(list(inherited_real), need_remove)
            inherited_real.difference_update(removed)

        # 当前文档的真实数据
        current_real = list(inherited_real)
        # 当前文档的虚假数据量
        fp_count = sum_val - tp
        if fp_count > len(fake_list):
            # 如果虚假数据池不足，则重复使用（但应避免，但这里为了简单）
            fake_selected = fake_list * (fp_count // len(fake_list) + 1)
            fake_selected = fake_selected[:fp_count]
        else:
            fake_selected = random.sample(fake_list, fp_count)

        # 合并真实和虚假数据
        all_triples = current_real + fake_selected
        # 打乱每条三元组的列顺序
        shuffled_triples = [shuffle_triple(t) for t in all_triples]
        # 打乱行顺序
        random.shuffle(shuffled_triples)

        # 输出文档到指定目录，按指定命名规则
        output_file = os.path.join(output_dir, f"Ising_20000flip1-{i *5}.txt")
        with open(output_file, 'w') as f:
            for triple in shuffled_triples:
                f.write(f"{triple[0]} {triple[1]} {triple[2]}\n")

        # 验证F1分数
        pred_set = set(tuple(sorted(t)) for t in shuffled_triples)
        tp_val = len(real_triples_set & pred_set)
        fp_val = len(pred_set - real_triples_set)
        fn_val = len(real_triples_set - pred_set)
        precision = tp_val / (tp_val + fp_val) if tp_val + fp_val > 0 else 0
        recall = tp_val / (tp_val + fn_val) if tp_val + fn_val > 0 else 0
        f1_actual = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        print(
            f"文档{i + 1} (NY_T{i + 1}.txt): 目标F1={f1:.3f}, 实际F1={f1_actual:.3f}, TP目标={tp}, TP实际={tp_val}, sum={sum_val}")


def main():
    # 参数设置
    real_file = "hyT1000.txt"  # 真实文档路径
    m = 85  # 最大数字不超过m
    f1_list = [0.89,0.8,0.73,0.64,0.57,0.51]  # F1分数列表
    sum_list = [146,146,145,144,143,138]  # 总行数列表
    output_dir = "TEST/pic/flip_F1/flip_data/flip_results"  # 输出目录

    # 读取真实文档
    real_triples_set = load_real_triples(real_file)
    print(f"真实三元组数量: {len(real_triples_set)}")

    # 生成虚假三元组池（最大池子大小设为10000，可根据需要调整）
    fake_pool = generate_fake_triples(m, real_triples_set, max_size=10000)
    print(f"虚假三元组池大小: {len(fake_pool)}")

    # 生成虚拟文档
    generate_virtual_documents(real_triples_set, fake_pool, f1_list, sum_list, output_dir)
    print(f"所有文档已生成并保存到目录: {output_dir}")


if __name__ == "__main__":
    main()