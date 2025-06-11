import os
import sys
import numpy as np
import random
import csv

# 设置 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.synthetic_generator import get_synthetic_hypergraph

G, k3_edges, k4_edges = get_synthetic_hypergraph()

print("✅ 成功导入 G 图及超边结构")
print("节点数:", len(G.nodes()))
print("三体超边数量:", len(k3_edges))
print("四体超边数量:", len(k4_edges))

# 参数设置
T = 100                     # 时间步数
beta_pair = 0.1            # 成对传播率
beta_k3 = 0.2              # 三体传播率
beta_k4 = 0.3              # 四体传播率
mu = 0.5                   # 恢复概率
initial_infected_ratio = 0.1

# 初始化感染状态
nodes = list(G.nodes())
N = len(nodes)
states = np.zeros((T, N), dtype=int)  # 0: S, 1: I

# 初始感染
initial_infected = random.sample(nodes, int(initial_infected_ratio * N))
for node in initial_infected:
    states[0, node] = 1

# 预构建节点到超边的映射（在时间循环之外）
node_to_k3 = {i: [] for i in range(N)}
for clique in k3_edges:
    for node in clique:
        node_to_k3[node].append(clique)

node_to_k4 = {i: [] for i in range(N)}
for clique in k4_edges:
    for node in clique:
        node_to_k4[node].append(clique)

# 模拟 SIS 动力学过程
for t in range(1, T):
    current = states[t - 1].copy()
    next_state = current.copy()

    for i in range(N):
        if current[i] == 1:
            # 感染者可能恢复
            if random.random() < mu:
                next_state[i] = 0
        else:
            # 易感者被成对邻居感染
            neighbors = list(G.neighbors(i))
            infected_neighbors = sum(current[j] for j in neighbors)
            p_pair = 1 - (1 - beta_pair) ** infected_neighbors

            # 被三体超边感染
            count_k3 = 0
            for clique in node_to_k3[i]:
                others = [n for n in clique if n != i]
                if all(current[n] == 1 for n in others):
                    count_k3 += 1
            p_k3 = 1 - (1 - beta_k3) ** count_k3 if count_k3 > 0 else 0

            # 被四体超边感染
            count_k4 = 0
            for clique in node_to_k4[i]:
                others = [n for n in clique if n != i]
                if all(current[n] == 1 for n in others):
                    count_k4 += 1
            p_k4 = 1 - (1 - beta_k4) ** count_k4 if count_k4 > 0 else 0

            # 总感染概率
            p_total = 1 - (1 - p_pair) * (1 - p_k3) * (1 - p_k4)
            if random.random() < p_total:
                next_state[i] = 1

    states[t] = next_state

# 保存时间序列数据
os.makedirs("data", exist_ok=True)
with open("data/SIS_time_series.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(states)

print("✅ SIS 高阶动力学时间序列已保存至 data/SIS_time_series.csv")