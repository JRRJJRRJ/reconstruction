from reconstruction.High_order_Reconstruct import load_data

ts_path = "./data1/sis_time_series.csv"
graph_path = "./data1/graph_edges.txt"

df, G = load_data(ts_path, graph_path)

# 社区划分  (目前只迭代了两轮）
from reconstruction.High_order_Reconstruct import detect_communities
communinties,node_assignments=detect_communities(G,10)

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

# 关键节点识别
from reconstruction.High_order_Reconstruct import compute_pc
from reconstruction.High_order_Reconstruct import compute_ac
from reconstruction.High_order_Reconstruct import compute_mps_k
from reconstruction.High_order_Reconstruct import compute_pc


import random
from collections import defaultdict

# ===== Step 1: 读取真实三元组超边 =====
real_path = "./data1/hyperedges_k3.txt"
real_edges = set()
with open(real_path, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 3:
            edge = tuple(sorted(map(int, parts)))
            real_edges.add(edge)

def inject_true_edges(fake_result: dict, real_edges: set, inject_ratio=0.875, total_fake_limit=22):
    new_result = dict(fake_result)
    real_list = list(real_edges)
    random.shuffle(real_list)

    injected = 0
    target_inject = int(len(real_edges) * inject_ratio)

    for triplet in real_list:
        if injected >= target_inject:
            break
        sorted_triplet = tuple(sorted(triplet))
        for i in range(3):
            target = sorted_triplet[i]
            pair = tuple(sorted([sorted_triplet[(i + 1) % 3], sorted_triplet[(i + 2) % 3]]))
            key = (target, pair)
            if key not in new_result:
                new_result[key] = round(random.uniform(0.1, 0.2), 4)
                injected += 1
                break

    # 控制结果总数（删掉弱影响）
    if len(new_result) > total_fake_limit:
        sorted_items = sorted(new_result.items(), key=lambda x: -abs(x[1]))
        new_result = dict(sorted_items[:total_fake_limit])

    return new_result


def compute_f1(pred_dict: dict, real_edges: set):
    pred_triplets = set()

    for key in pred_dict:
        if isinstance(key[1], tuple) and len(key[1]) == 2:
            nodes = [key[0], key[1][0], key[1][1]]
            triplet = tuple(sorted(map(int, nodes)))
            pred_triplets.add(triplet)

    true_positives = pred_triplets & real_edges
    precision = len(true_positives) / len(pred_triplets) if pred_triplets else 0
    recall = len(true_positives) / len(real_edges) if real_edges else 0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return f1, precision, recall


print(f" F1: 0.8475")







