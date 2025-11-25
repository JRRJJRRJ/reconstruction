import networkx as nx
import numpy as np
import itertools
from collections import defaultdict
import os


def generate_network_structures():
    """生成并保存网络结构"""
    # 设置随机种子以确保可重复性
    np.random.seed(42)

    # 1. 生成ER网络 (用于SIS动力学)
    n_nodes = 300
    p_er = 0.02
    G_er = nx.erdos_renyi_graph(n_nodes, p_er, seed=42)

    # 2. 生成BA网络 (用于Ising动力学)
    m_ba = 3
    G_ba = nx.barabasi_albert_graph(n_nodes, m_ba, seed=42)

    # 提取成对相互作用（边）和三体相互作用（三角形）
    def extract_interactions(G):
        # 提取所有边（成对相互作用）
        pairwise_edges = list(G.edges())

        # 提取所有三角形（三体相互作用）
        triangles = set()
        for node in G.nodes():
            # 获取节点的所有邻居
            neighbors = list(G.neighbors(node))

            # 找出邻居之间的所有连接（形成三角形）
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    if G.has_edge(neighbors[i], neighbors[j]):
                        # 对节点排序以确保唯一表示
                        triangle = tuple(sorted([node, neighbors[i], neighbors[j]]))
                        triangles.add(triangle)

        return pairwise_edges, list(triangles)

    # 为两个网络提取相互作用
    er_edges, er_triangles = extract_interactions(G_er)
    ba_edges, ba_triangles = extract_interactions(G_ba)

    # 保存ER网络的相互作用
    with open('er_pairwise.txt', 'w') as f:
        for edge in er_edges:
            f.write(f"{edge[0]} {edge[1]}\n")

    with open('er_triadic.txt', 'w') as f:
        for triangle in er_triangles:
            f.write(f"{triangle[0]} {triangle[1]} {triangle[2]}\n")

    # 保存BA网络的相互作用
    with open('ba_pairwise.txt', 'w') as f:
        for edge in ba_edges:
            f.write(f"{edge[0]} {edge[1]}\n")

    with open('ba_triadic.txt', 'w') as f:
        for triangle in ba_triangles:
            f.write(f"{triangle[0]} {triangle[1]} {triangle[2]}\n")

    return er_edges, er_triangles, ba_edges, ba_triangles


# 生成网络结构
er_edges, er_triangles, ba_edges, ba_triangles = generate_network_structures()


def simulate_sis_dynamics(edges, triangles, n_nodes, total_steps, burn_in=200):
    """模拟SIS动力学"""
    # 初始化参数
    alpha = 0.3  # 两体相互作用强度参数
    beta = 0.5  # 三体相互作用强度参数
    sigma = 0.2  # 恢复率

    # 计算归一化因子
    k1 = len(edges) / n_nodes  # 平均度
    k2 = len(triangles) / n_nodes  # 平均三角形参与数

    gamma1 = alpha / k1
    gamma2 = beta / k2

    # 初始化状态：随机选择10%的节点作为初始感染
    states = np.zeros(n_nodes)
    initial_infected = np.random.choice(n_nodes, size=int(0.1 * n_nodes), replace=False)
    states[initial_infected] = 1

    # 创建邻接表和高阶相互作用表以提高效率
    adj_list = defaultdict(list)
    for u, v in edges:
        adj_list[u].append(v)
        adj_list[v].append(u)

    # 创建三角形查找表 (节点 -> 包含该节点的三角形)
    triangle_dict = defaultdict(list)
    for tri in triangles:
        for node in tri:
            triangle_dict[node].append([n for n in tri if n != node])

    # 存储时间序列
    time_series = np.zeros((total_steps, n_nodes))

    # 模拟动力学
    for step in range(total_steps):
        new_states = states.copy()

        for i in range(n_nodes):
            # 恢复过程
            if states[i] == 1 and np.random.rand() < sigma:
                new_states[i] = 0
                continue

            # 感染过程 (仅对易感节点)
            if states[i] == 0:
                # 计算两体相互作用贡献
                pair_contribution = 0
                for j in adj_list[i]:
                    if states[j] == 1:
                        pair_contribution += gamma1

                # 计算三体相互作用贡献
                triadic_contribution = 0
                for pair in triangle_dict[i]:
                    j, k = pair
                    if states[j] == 1 and states[k] == 1:
                        triadic_contribution += gamma2

                # 计算总感染概率
                infection_prob = min(1.0, pair_contribution + triadic_contribution)

                # 决定是否感染
                if np.random.rand() < infection_prob:
                    new_states[i] = 1

        states = new_states
        time_series[step] = states

    # 丢弃前burn_in步作为燃烧期
    time_series = time_series[burn_in:]

    return time_series


def simulate_ising_dynamics(edges, triangles, n_nodes, total_steps, burn_in=200):
    """模拟Ising动力学"""
    # 初始化参数
    J1 = 0.5  # 两体耦合强度
    J2 = 0.3  # 三体耦合强度
    T = 2.0  # 温度 (接近临界温度)

    # 初始化状态：随机自旋
    spins = np.random.choice([-1, 1], size=n_nodes)

    # 创建邻接表和高阶相互作用表以提高效率
    adj_list = defaultdict(list)
    for u, v in edges:
        adj_list[u].append(v)
        adj_list[v].append(u)

    # 创建三角形查找表 (节点 -> 包含该节点的三角形)
    triangle_dict = defaultdict(list)
    for tri in triangles:
        for node in tri:
            triangle_dict[node].append([n for n in tri if n != node])

    # 存储时间序列
    time_series = np.zeros((total_steps, n_nodes))

    # 模拟动力学 (使用Metropolis算法)
    for step in range(total_steps):
        for _ in range(n_nodes):  # 尝试翻转每个节点一次
            i = np.random.randint(n_nodes)

            # 计算当前能量贡献
            energy_current = 0
            # 两体相互作用
            for j in adj_list[i]:
                energy_current -= J1 * spins[i] * spins[j]

            # 三体相互作用
            for pair in triangle_dict[i]:
                j, k = pair
                energy_current -= J2 * spins[i] * spins[j] * spins[k]

            # 尝试翻转
            spins[i] = -spins[i]

            # 计算翻转后的能量贡献
            energy_flipped = 0
            # 两体相互作用
            for j in adj_list[i]:
                energy_flipped -= J1 * spins[i] * spins[j]

            # 三体相互作用
            for pair in triangle_dict[i]:
                j, k = pair
                energy_flipped -= J2 * spins[i] * spins[j] * spins[k]

            # 计算能量变化
            delta_energy = energy_flipped - energy_current

            # Metropolis准则：以概率min(1, exp(-ΔE/T))接受翻转
            if delta_energy > 0 and np.random.rand() >= np.exp(-delta_energy / T):
                spins[i] = -spins[i]  # 拒绝翻转

        time_series[step] = spins

    # 丢弃前burn_in步作为燃烧期
    time_series = time_series[burn_in:]

    # 将自旋值从{-1,1}映射到{0,1}以便与SIS保持一致
    time_series = (time_series + 1) / 2

    return time_series


def generate_time_series_datasets():
    """生成所有时间序列数据集"""
    n_nodes = 300

    # 创建输出目录
    os.makedirs('SIS', exist_ok=True)
    os.makedirs('Ising', exist_ok=True)

    # 生成不同长度的时间序列
    lengths = range(1000, 21000, 1000)  # 1000, 2000, ..., 20000

    print("Generating SIS time series...")
    for i, length in enumerate(lengths):
        print(f"  Generating SIS time series with length {length} ({i + 1}/20)")
        total_steps = length + 200  # 总步长 = 所需长度 + 燃烧期

        # 生成SIS时间序列
        sis_series = simulate_sis_dynamics(er_edges, er_triangles, n_nodes, total_steps)

        # 保存为CSV
        filename = f"SIS/sis_timeseries_{length}.csv"
        np.savetxt(filename, sis_series, delimiter=',', fmt='%d')

    print("Generating Ising time series...")
    for i, length in enumerate(lengths):
        print(f"  Generating Ising time series with length {length} ({i + 1}/20)")
        total_steps = length + 200  # 总步长 = 所需长度 + 燃烧期

        # 生成Ising时间序列
        ising_series = simulate_ising_dynamics(ba_edges, ba_triangles, n_nodes, total_steps)

        # 保存为CSV
        filename = f"Ising/ising_timeseries_{length}.csv"
        np.savetxt(filename, ising_series, delimiter=',', fmt='%d')

    print("All time series datasets generated successfully!")


# 生成所有时间序列数据集
generate_time_series_datasets()