from reconstruction.High_order_Reconstruct import load_data
import numpy as np
ts_path = "./data1/sis_time_series.csv"
graph_path = "./data1/graph_edges.txt"

ts_path1 = "./RealNet_data/generated_time_series.csv"
graph_path1 = "./RealNet_data/Paired_connection_adjusted_minus1.txt"

np.random.seed(42)

df, G = load_data(ts_path1, graph_path1)

missing_in_G = set(df.columns) - set(G.nodes)
if missing_in_G:
    print("missing_in_G:",missing_in_G)
    print(f"警告：df 中存在 {len(missing_in_G)} 个节点未在 G 中出现，将添加进图中")
    G.add_nodes_from(missing_in_G)

print(f"Loaded graph with {len(G.nodes())} nodes and {len(G.edges())} edges")

# 社区划分  (目前只迭代了两轮）
from reconstruction.High_order_Reconstruct import detect_communities
communinties, node_assignments = detect_communities(G, k=5, method='pcdms', random_state=42)

import json

with open('communities.json', 'w') as f:
    json.dump(communinties, f)


print(type(df.columns[0]))  # 看看列名类型：int还是str
print(communinties[:10])       # 看看community前10个节点的类型
print(df.columns[:10])      # 看看df前10列列名
print(df.head())

# 社区内重构
from reconstruction.High_order_Reconstruct import reconstruct_community
InResult={}
for idx, community in enumerate(communinties):
    community = [int(node) for node in community]
    print(f" 正在重构第 {idx + 1} 个社区，包含 {len(community)} 个节点")


    result = reconstruct_community(community, df,1)

    print(f" 第 {idx + 1} 个社区重构完成，共识别出 {len(result)} 条影响关系")

    for key, val in result.items():
        print(f"  {key} => {val:.4f}")

    InResult.update(result)


# 关键节点全局重构
from reconstruction.High_order_Reconstruct import  compute_pc
from reconstruction.High_order_Reconstruct import  compute_mps_k
from reconstruction.High_order_Reconstruct import  compute_ac
from reconstruction.High_order_Reconstruct import  compute_node_score
from reconstruction.High_order_Reconstruct import reconstruct_node_global
# 得分计算
node_scores = {}
for node in G.nodes:
    score = compute_node_score(G, node, communinties, k=3)  # 这里默认 k=3
    node_scores[node] = score

# 提取前5%的高分节点
top_5_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)[:5]
# 加1并转成字符串
top_5_nodes_plus1 = [str(int(node) + 1) for node, score in top_5_nodes]
print("top_5_nodes_plus1:",top_5_nodes_plus1)

# 全局重构
nodes_candidate = [str(int(n)) for n in G.nodes]
global_results = {}

for target in top_5_nodes_plus1:
    result = reconstruct_node_global(target, nodes_candidate, df)
    global_results.update(result)
    print("目标节点为：", target)
    print("results:", result)
print("global_results:", global_results)
print(result)
print("InResult:", InResult)
print(1)

# 合并重构结果
import json
import numpy as np


def merge_results_with_type_conversion(global_results, InResult):
    """
    合并全局重构和局部重构结果，处理节点标识符类型不一致问题

    参数:
        global_results: 全局重构结果字典（字符串节点）
        InResult: 局部重构结果字典（整数节点）

    返回:
        merged_results: 合并后的完整重构结果（字符串节点）
    """
    # 创建合并字典（使用字符串节点）
    merged_results = {}

    # 辅助函数：将节点标识符转换为字符串
    def to_str(node):
        if isinstance(node, int):
            return str(node)
        return node

    # 辅助函数：标准化键格式
    def standardize_key(key):
        target, influencers = key
        target = to_str(target)

        if isinstance(influencers, tuple):
            # 处理二元组影响节点
            sorted_influencers = tuple(sorted(to_str(n) for n in influencers))
            return (target, sorted_influencers)
        else:
            # 处理单节点影响
            return (target, to_str(influencers))

    # 添加局部重构结果（转换为字符串节点）
    for key, value in InResult.items():
        standardized_key = standardize_key(key)
        merged_results[standardized_key] = value

    # 添加全局重构结果（已为字符串节点）
    for key, value in global_results.items():
        standardized_key = standardize_key(key)

        # 如果键已存在，保留绝对值最大的系数
        if standardized_key in merged_results:
            existing_value = merged_results[standardized_key]
            if abs(value) > abs(existing_value):
                merged_results[standardized_key] = value
        else:
            merged_results[standardized_key] = value

    return merged_results


# 使用示例
merged_results = merge_results_with_type_conversion(global_results, InResult)
print("merged_results:", merged_results)



from index.F1_score import F1

precision, recall, f1, metrics = F1("RealNet_data/High_connection.txt", merged_results)
print("precision:", precision)
print("recall:", recall)
print("f1:", f1)
print("metrics:", metrics)
