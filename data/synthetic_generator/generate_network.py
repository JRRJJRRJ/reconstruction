import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from collections import defaultdict
import os
from community_detection.PCDMS import PCDMS
from numpy.random import choice


def compute_pc(G, node_community):
    pc = {}
    for node in G.nodes():
        k_v = G.degree[node]
        if k_v == 0:
            pc[node] = 0
            continue
        community_edges = defaultdict(int)
        for neighbor in G.neighbors(node):
            c = node_community[neighbor]
            community_edges[c] += 1
        pc[node] = 1 - sum((count / k_v) ** 2 for count in community_edges.values())
    return pc


def compute_ac(G):
    max_deg = max(dict(G.degree()).values())
    return {n: G.degree[n] / max_deg for n in G.nodes()}


def compute_mps_k(G, k_cliques):
    mps = {n: 0 for n in G.nodes()}
    for clique in k_cliques:
        for n in clique:
            mps[n] += 1
    total_cliques = len(k_cliques)
    return {n: mps[n] / total_cliques if total_cliques > 0 else 0 for n in G.nodes()}


def find_k_cliques(G, k):
    cliques = []
    for nodes in combinations(G.nodes(), k):
        if all(G.has_edge(u, v) for u, v in combinations(nodes, 2)):
            cliques.append(tuple(sorted(nodes)))
    return cliques


def sample_k_hyperedges(G, w, k, num_edges, max_attempts_per_edge=50, allow_overlap=True):
    nodes = list(G.nodes())
    total_weight = sum(w.values())
    probs = [w[v] / total_weight for v in nodes]
    hyperedges = set()
    attempts = 0
    max_total_attempts = num_edges * max_attempts_per_edge

    while len(hyperedges) < num_edges and attempts < max_total_attempts:
        sampled = tuple(sorted(choice(nodes, size=k, replace=False, p=probs)))
        if all(G.has_edge(u, v) for u, v in combinations(sampled, 2)):
            if allow_overlap or sampled not in hyperedges:
                hyperedges.add(sampled)
        attempts += 1

    return list(hyperedges)


def main():
    N = 100
    p = 0.1
    G = nx.erdos_renyi_graph(N, p, seed=30)

    # 社区划分
    pcdms_model = PCDMS(k=10, verbose=False)
    node_community = pcdms_model.fit(G)
    communities = pcdms_model.get_communities()

    # 构造 clique 信息
    k3_cliques = find_k_cliques(G, 3)
    k4_cliques = find_k_cliques(G, 4)

    # 基础指标
    PC = compute_pc(G, node_community)
    AC = compute_ac(G)
    MPS3 = compute_mps_k(G, k3_cliques)
    MPS4 = compute_mps_k(G, k4_cliques)

    # 权重计算
    alpha, beta, gamma3, gamma4 = 0.4, 0.3, 0.2, 0.1
    w = {n: alpha * PC[n] + beta * AC[n] + gamma3 * MPS3[n] + gamma4 * MPS4[n] for n in G.nodes()}

    # 生成高阶超边
    k3_edges = sample_k_hyperedges(G, w, k=3, num_edges=300)
    k4_edges = sample_k_hyperedges(G, w, k=4, num_edges=100) if any(MPS4.values()) else []

    # 保存节点指标
    os.makedirs("data", exist_ok=True)
    with open("data/node_metrics.csv", "w", newline="") as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(["node", "PC", "AC", "MPS3", "MPS4", "w"])
        for n in G.nodes():
            writer.writerow([n, PC[n], AC[n], MPS3[n], MPS4[n], w[n]])

    return G, k3_edges, k4_edges


def get_synthetic_hypergraph():
    return main()


if __name__ == "__main__":
    G, k3_edges, k4_edges = main()
    # 保存为边列表文件
    nx.write_edgelist(G, "data/synthetic_generator/graph_edges.txt", data=False)

    # 可视化（可选）
    try:
        pos = nx.spring_layout(G, seed=42)
        fig, ax = plt.subplots(figsize=(10, 8))
        node_colors = [G.degree[n] for n in G.nodes()]
        nx.draw(G, pos, node_size=100, node_color=node_colors, cmap=plt.cm.viridis, ax=ax)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
        sm.set_array(node_colors)
        plt.colorbar(sm, ax=ax, label="Node Degree")
        ax.set_title("Node Degree Visualization")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig("data/node_weights.png", dpi=300)
        plt.show()
    except Exception as e:
        print(f"⚠️ 可视化失败: {e}")
