import numpy as np
import random
import pandas as pd


def load_edges(edge_file):
    edges = []
    with open(edge_file, 'r') as f:
        for line in f:
            nodes = line.strip().split(',')
            if len(nodes) == 2:
                edges.append((nodes[0], nodes[1]))
    return edges

def load_hyperedges(hyper_file):
    hyperedges = []
    with open(hyper_file, 'r') as f:
        for line in f:
            nodes = line.strip().split(',')
            if len(nodes) >= 3:
                hyperedges.append(nodes)
    return hyperedges


def simulate_time_series(paired_edges, hyperedges, node_list,
                         T=100, beta_edge=0.4, beta_hyper=0.6, delta=0.3,
                         output_csv_path='simulated_time_series.csv'):
    """
    模拟高阶SIS传播，并按从1开始的节点顺序生成CSV时间序列数据。

    参数：
        paired_edges: list of (u, v)
        hyperedges: list of [u1, u2, ..., un]
        node_list: 所有节点编号（字符串）
        T: 时间长度
        beta_edge: 成对感染概率
        beta_hyper: 高阶感染概率（高阶传播）
        delta: 恢复概率
        output_csv_path: 保存路径
    返回：
        X: N×T 状态矩阵
        node2idx: 节点名 → 索引
    """

    # 将节点编号转换为整数并排序，再转为字符串作为列名
    sorted_nodes = sorted(set(int(node) for node in node_list))
    node_list_sorted = [str(node) for node in sorted_nodes]

    node2idx = {node: idx for idx, node in enumerate(node_list_sorted)}
    N = len(node_list_sorted)
    X = np.zeros((T, N), dtype=int)

    # 初始感染5%
    initial_infected = random.sample(range(N), max(1, N // 20))
    X[0, initial_infected] = 1

    for t in range(1, T):
        X[t] = X[t - 1].copy()
        for i, node in enumerate(node_list_sorted):
            if X[t - 1, i] == 1:
                if random.random() < delta:
                    X[t, i] = 0
            else:
                infected_neighbors = 0
                for u, v in paired_edges:
                    if u == node and X[t - 1, node2idx[v]] == 1:
                        infected_neighbors += 1
                    elif v == node and X[t - 1, node2idx[u]] == 1:
                        infected_neighbors += 1
                p1 = 1 - (1 - beta_edge) ** infected_neighbors

                hyper_infection = 0
                for hedge in hyperedges:
                    if node in hedge:
                        active = sum(X[t - 1, node2idx[u]] for u in hedge if u != node)
                        if active >= 2:
                            hyper_infection += 1
                p2 = 1 - (1 - beta_hyper) ** hyper_infection

                p_total = 1 - (1 - p1) * (1 - p2)

                if random.random() < p_total:
                    X[t, i] = 1

    # 保存为 CSV（列为从1到N的有序节点名）
    df = pd.DataFrame(X, columns=node_list_sorted)
    df.to_csv(output_csv_path, index=False, header=False)


    return X, node2idx

paired_edges = load_edges('Thiers12/paired_connection')
hyperedges = load_hyperedges('Thiers12/High_connection')
all_nodes = sorted(set(
    [u for e in paired_edges for u in e] +
    [v for e in paired_edges for v in e] +  # 添加这行：补上成对边中每条边的另一个节点
    [node for h in hyperedges for node in h]  # 遍历所有高阶超边中的所有节点
))

# 假设 simulate_time_series 已定义，且已导入 paired_edges、hyperedges、all_nodes

import os

for T in range(1000, 20001, 1000):  # 从1000到20000，步长为1000
    directory = f"Thiers12/T={T}"  # 注意去掉你原来写错的“·”
    os.makedirs(directory, exist_ok=True)  # 若目录不存在则创建
    output_path = f"{directory}/Thiers12_{T}.csv"
    simulate_time_series(paired_edges, hyperedges, all_nodes, T=T, output_csv_path=output_path)

