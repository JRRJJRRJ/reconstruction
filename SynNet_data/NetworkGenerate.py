import networkx as nx
import numpy as np
import itertools
from collections import defaultdict


def generate_and_save_networks():
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
    with open('../SynNet_data/er_pairwise.txt', 'w') as f:
        for edge in er_edges:
            f.write(f"{edge[0]} {edge[1]}\n")

    with open('../SynNet_data/er_triadic.txt', 'w') as f:
        for triangle in er_triangles:
            f.write(f"{triangle[0]} {triangle[1]} {triangle[2]}\n")

    # 保存BA网络的相互作用Simplices_txt
    with open('../SynNet_data/ba_pairwise.txt', 'w') as f:
        for edge in ba_edges:
            f.write(f"{edge[0]} {edge[1]}\n")

    with open('../SynNet_data/ba_triadic.txt', 'w') as f:
        for triangle in ba_triangles:
            f.write(f"{triangle[0]} {triangle[1]} {triangle[2]}\n")

    return G_er, G_ba, er_edges, er_triangles, ba_edges, ba_triangles


# 执行网络生成
G_er, G_ba, er_edges, er_triangles, ba_edges, ba_triangles = generate_and_save_networks()