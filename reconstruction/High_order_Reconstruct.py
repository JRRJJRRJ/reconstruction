import numpy as np
import pandas as pd
import networkx as nx
from itertools import combinations
from collections import defaultdict
from sklearn.linear_model import Lasso
from community_detection.PCDMS import PCDMS  # ✅ 使用包方式导入

# -------------------------------
# -------------------------------
# 工具函数：三阶完全连接体查找
# -------------------------------
def find_k_cliques(G, k):
    cliques = []
    for nodes in combinations(G.nodes(), k):
        if all(G.has_edge(u, v) for u, v in combinations(nodes, 2)):
            cliques.append(tuple(sorted(nodes)))
    return cliques

# -------------------------------
# 节点指标计算（PC、MPS、AC）
# -------------------------------
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

def compute_mps_k(G, k_cliques):
    mps = {n: 0 for n in G.nodes()}
    for clique in k_cliques:
        for n in clique:
            mps[n] += 1
    total_cliques = len(k_cliques)
    return {n: mps[n] / total_cliques if total_cliques > 0 else 0 for n in G.nodes()}

def compute_ac(ts_matrix):
    # ts_matrix: shape (T, N)
    T = ts_matrix.shape[0]
    return {
        n: np.count_nonzero(ts_matrix[:, n] == 1) / T
        for n in range(ts_matrix.shape[1])
    }

# -------------------------------
# 主函数：两阶段重构主程序（Python版 LVexact）
# -------------------------------
def reconstruct_high_order(ts_file, G, alpha=0.4, beta=0.3, gamma=0.3, top_k_percent=5):
    # 读取时间序列
    ts = pd.read_csv(ts_file, header=None).values  # shape: (T, N)
    T, N = ts.shape

    # Step 1: 社区划分
    pcdms_model = PCDMS(k=10, verbose=False)
    node_community = pcdms_model.fit(G)
    communities = pcdms_model.get_communities()

    # Step 2: 计算 PC、MPS、AC
    pc_dict = compute_pc(G, node_community)
    mps_dict = compute_mps_k(G, find_k_cliques(G, 3))
    ac_dict = compute_ac(ts)

    # Step 3: 计算 w(v)
    w = {
        n: alpha * pc_dict[n] + beta * ac_dict[n] + gamma * mps_dict[n]
        for n in G.nodes()
    }

    # Step 4: 筛选关键节点
    threshold = np.percentile(list(w.values()), 100 - top_k_percent)
    global_nodes = [n for n in G.nodes() if w[n] >= threshold]

    # Step 5: 社区内重构（使用一阶+三阶特征 + Lasso）
    results = {}
    for comm in communities:
        sub_ts = ts[:, comm]  # shape: (T, m)
        m = len(comm)

        for idx_i, i in enumerate(comm):
            features = []
            targets = []
            for t in range(1, T - 1):
                row = []
                # pairwise x_i x_j
                for idx_j, j in enumerate(comm):
                    if j != i:
                        row.append(sub_ts[t, idx_i] * sub_ts[t, idx_j])
                # high-order x_i x_j x_k
                for idx_j, j in enumerate(comm):
                    for idx_k, k in enumerate(comm):
                        if j < k and i != j and i != k:
                            row.append(sub_ts[t, idx_i] * sub_ts[t, idx_j] * sub_ts[t, idx_k])
                features.append(row)
                dy = (sub_ts[t + 1, idx_i] - sub_ts[t - 1, idx_i]) / 2  # 中心差分
                targets.append(dy)
            model = Lasso(alpha=0.001)
            model.fit(features, targets)
            results[i] = model.coef_

    # Step 6: 全局重构（仅对关键节点）
    for i in global_nodes:
        features = []
        targets = []
        for t in range(1, T - 1):
            row = []
            for j in range(N):
                if j != i:
                    row.append(ts[t, i] * ts[t, j])
            for j in range(N):
                for k in range(j + 1, N):
                    if i != j and i != k:
                        row.append(ts[t, i] * ts[t, j] * ts[t, k])
            features.append(row)
            dy = (ts[t + 1, i] - ts[t - 1, i]) / 2
            targets.append(dy)
        model = Lasso(alpha=0.001)
        model.fit(features, targets)
        results[i] = model.coef_

    return results
