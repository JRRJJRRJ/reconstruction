import numpy as np
import csv
import sys
import os
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.synthetic_generator import get_synthetic_hypergraph

G, k3_edges, k4_edges = get_synthetic_hypergraph()

print("G图为：", G)
print("k3_edges", k3_edges)
print("k4_edges", k4_edges)

# -------------------------------
# 参数设置
# -------------------------------
N = G.number_of_nodes()
T = 200  # 时间步
J_pair = 0.5  # 成对相互作用强度
J_k3 = 1.0  # 三体相互作用强度
J_k4 = 1.0  # 四体相互作用强度
noise = 2.0  # 热噪声强度 (温度)
init_prob = 0.5  # 初始激活概率

np.random.seed(42)
random.seed(42)

# -------------------------------
# 初始化节点状态（-1 或 +1）
# -------------------------------
states = np.zeros((T, N), dtype=int)
states[0] = np.where(np.random.rand(N) < init_prob, 1, -1)

# -------------------------------
# 动力学演化（使用 Glauber 动力学）
# -------------------------------
for t in range(1, T):
    prev = states[t - 1].copy()
    next_state = prev.copy()

    # 随机顺序更新节点
    node_order = list(range(N))
    random.shuffle(node_order)

    for i in node_order:
        # 1. 计算当前状态下的局部场
        local_field = 0

        # 成对相互作用
        for j in G.neighbors(i):
            local_field += J_pair * prev[j]

        # 三体相互作用
        for clique in k3_edges:
            if i in clique:
                others = [node for node in clique if node != i]
                # 计算三个自旋的乘积（包括自身）
                spin_product = prev[i] * prev[others[0]] * prev[others[1]]
                # 相互作用贡献（考虑三体耦合强度）
                local_field += J_k3 * spin_product * prev[i]  # 与自身状态相乘

        # 四体相互作用
        for clique in k4_edges:
            if i in clique:
                others = [node for node in clique if node != i]
                # 计算四个自旋的乘积（包括自身）
                spin_product = prev[i] * np.prod(prev[others])
                # 相互作用贡献
                local_field += J_k4 * spin_product * prev[i]  # 与自身状态相乘

        # 2. 计算状态翻转概率（Glauber 动力学）
        # 翻转能量变化 ΔE = 2 * s_i * H_i
        delta_E = 2 * prev[i] * local_field
        # 翻转概率 P_flip = 1 / (1 + exp(ΔE / T))
        p_flip = 1 / (1 + np.exp(delta_E / noise))

        # 3. 根据概率决定是否翻转状态
        if random.random() < p_flip:
            next_state[i] = -prev[i]

    states[t] = next_state

# -------------------------------
# 保存为 CSV 文件
# -------------------------------
os.makedirs("data", exist_ok=True)
with open("data/ising_time_series.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(states)

print("✅ Ising 高阶动力学时间序列已保存至 data/ising_time_series.csv")