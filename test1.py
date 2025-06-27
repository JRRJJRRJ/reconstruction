from reconstruction.High_order_Reconstruct import  compute_pc
from reconstruction.High_order_Reconstruct import  compute_mps_k
from reconstruction.High_order_Reconstruct import  compute_ac
from reconstruction.High_order_Reconstruct import  compute_node_score
from reconstruction.High_order_Reconstruct import load_data
import json
from reconstruction.High_order_Reconstruct import reconstruct_node_global

ts_path1 = "./RealNet_data/generated_time_series.csv"
graph_path1 = "./RealNet_data/Paired_connection_adjusted_minus1.txt"

with open('communities.json', 'r') as f:
    communinties = json.load(f)

df, G = load_data(ts_path1, graph_path1)

node_scores = {}
for node in G.nodes:
    score = compute_node_score(G, node, communinties, k=3)  # 这里默认 k=3
    node_scores[node] = score

# 提取前5%的高分节点
top_5_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)[:5]
# 加1并转成字符串
top_5_nodes_plus1 = [str(int(node) + 1) for node, score in top_5_nodes]
print("top_5_nodes_plus1:",top_5_nodes_plus1)

nodes_candidate = [str(int(n)) for n in G.nodes]
global_results = {}
print(global_results)

for target in top_5_nodes_plus1:
    result = reconstruct_node_global(target, nodes_candidate, df)
    global_results.update(result)
    print("目标节点为：",target)
    print("results:",result)
print("global_results:",global_results)
