import os
import networkx as nx
from reconstruction.High_order_Reconstruct import reconstruct_high_order
from sklearn.metrics import f1_score

# Step 1: 加载时间序列数据路径
ts_file = "data/synthetic_generator/data/data/ising_time_series.csv"

# Step 2: 加载成对网络结构（必须是二元边）
G = nx.read_edgelist("data/synthetic_generator/data/synthetic_generator/graph_edges.txt", nodetype=int)

# Step 3: 加载真实三阶高阶边
def load_hyperedges(filepath):
    with open(filepath, "r") as f:
        return set(tuple(sorted(map(int, line.strip().split()))) for line in f if line.strip())

true_k3_edges = load_hyperedges("data/synthetic_generator/data/synthetic_generator/hyperedges_k3.txt")

# Step 4: 执行重构
results, predicted_k3_edges = reconstruct_high_order(
    ts_file=ts_file,
    G=G,
    alpha=0.4, beta=0.3, gamma=0.3, top_k_percent=5
)

# Step 5: 统一边集合，计算 F1 分数
all_possible_edges = true_k3_edges.union(predicted_k3_edges)
y_true = [1 if edge in true_k3_edges else 0 for edge in all_possible_edges]
y_pred = [1 if edge in predicted_k3_edges else 0 for edge in all_possible_edges]

# Step 6: 输出结果
f1 = f1_score(y_true, y_pred)
print(f"🔍 F1 score (三阶高阶边): {f1:.4f}")
