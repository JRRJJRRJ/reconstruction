import os
import networkx as nx
from reconstruction.High_order_Reconstruct import reconstruct_high_order

# Step 1: 加载时间序列数据路径（SIS 或 Ising）
ts_file = "data/data/data/ising_time_series.csv"  # 或 "data/sis_time_series.csv"

# Step 2: 加载成对网络结构（可使用你已有的 G）
G = nx.read_edgelist("data/data/data/synthetic_generator/graph_edges.txt", nodetype=int)

# Step 3: 执行高阶重构
results = reconstruct_high_order(ts_file, G, alpha=0.4, beta=0.3, gamma=0.3, top_k_percent=5)

# Step 4: 查看结果（可选）
for node, coef in list(results.items())[:5]:
    print(f"节点 {node} 重构系数向量前 10 维：{coef[:10]}")
