from reconstruction.High_order_Reconstruct import load_data

ts_path = "./data1/sis_time_series.csv"
graph_path = "./data1/graph_edges.txt"

ts_path1 = "./RealNet_data/generated_time_series.csv"
graph_path1 = "./RealNet_data/Paired_connection_adjusted_minus1.txt"

df, G = load_data(ts_path1, graph_path1)

missing_in_G = set(df.columns) - set(G.nodes)
if missing_in_G:
    print("missing_in_G:",missing_in_G)
    print(f"⚠️ 警告：df 中存在 {len(missing_in_G)} 个节点未在 G 中出现，将添加进图中")
    G.add_nodes_from(missing_in_G)

print(f"Loaded graph with {len(G.nodes())} nodes and {len(G.edges())} edges")

# 社区划分  (目前只迭代了两轮）
from reconstruction.High_order_Reconstruct import detect_communities
communinties,node_assignments=detect_communities(G,14)

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



def extract_predicted_hyperedges(merged_results):
    """
    从合并结果中提取预测的三阶超边

    参数:
        merged_results: 合并后的重构结果字典

    返回:
        predicted_hyperedges: 预测的三阶超边集合
    """
    predicted_hyperedges = set()

    for (target, influencers), _ in merged_results.items():
        if isinstance(influencers, tuple) and len(influencers) == 2:
            # 创建排序的三元组 (node1, node2, target)
            hyperedge = tuple(sorted([target, influencers[0], influencers[1]]))
            predicted_hyperedges.add(hyperedge)

    return predicted_hyperedges


# 使用示例
predicted_hyperedges = extract_predicted_hyperedges(merged_results)


def load_true_hyperedges(file_path, min_nodes=3, max_nodes=3):
    """
    加载真实高阶网络文件并解析为三阶超边集合

    参数:
        file_path: 真实高阶网络文件路径
        min_nodes: 考虑的最小节点数
        max_nodes: 考虑的最大节点数

    返回:
        true_hyperedges: 真实的三阶超边集合
    """
    true_hyperedges = set()

    with open(file_path, 'r') as f:
        for line in f:
            # 分割行并移除空白
            parts = line.strip().split()
            nodes = [part.strip() for part in parts if part.strip()]

            # 只处理指定节点数量的行
            if min_nodes <= len(nodes) <= max_nodes:
                # 创建排序的超边元组
                hyperedge = tuple(sorted(nodes))
                true_hyperedges.add(hyperedge)

    return true_hyperedges


# 使用示例
true_hyperedges = load_true_hyperedges("real.txt", min_nodes=3, max_nodes=3)


def evaluate_reconstruction(predicted_hyperedges, true_hyperedges_set):
    """
    评估重构结果与真实高阶网络的匹配程度
    使用优化的哈希表方法进行匹配

    参数:
        predicted_hyperedges: 预测的三阶超边集合
        true_hyperedges_set: 真实的三阶超边集合（集合类型）

    返回:
        precision, recall, f1: 评估指标
        metrics: 详细指标字典
    """
    # 1. 准备真实超边数据
    true_hyperedges = []  # 存储所有真实超边
    target_index = {}  # 目标节点 -> [超边索引]

    # 从集合中提取真实超边
    for hyperedge in true_hyperedges_set:
        # 假设超边格式为 (a, b, c) 的三元组
        if len(hyperedge) == 3:
            target = hyperedge[0]
            inf1, inf2 = hyperedge[1], hyperedge[2]

            true_hyperedges.append({
                'target': target,
                'influencers': (inf1, inf2)
            })

            if target not in target_index:
                target_index[target] = []
            target_index[target].append(len(true_hyperedges) - 1)  # 存储索引

    # 2. 准备预测超边
    predicted_list = []
    for hyperedge in predicted_hyperedges:
        if isinstance(hyperedge, tuple) and len(hyperedge) == 2:
            target, influencers = hyperedge
            if isinstance(influencers, tuple) and len(influencers) == 2:
                inf1, inf2 = influencers
                predicted_list.append({
                    'target': target,
                    'influencers': (inf1, inf2)
                })

    # 3. 匹配过程
    true_positives = 0
    matched_true_indices = set()
    matched_pred_indices = set()
    matches = []  # 存储匹配详情

    for pred_idx, pred in enumerate(predicted_list):
        target = pred['target']
        inf1, inf2 = pred['influencers']

        if target in target_index:
            for true_idx in target_index[target]:
                if true_idx not in matched_true_indices:
                    true_edge = true_hyperedges[true_idx]
                    true_inf1, true_inf2 = true_edge['influencers']

                    # 检查影响节点是否匹配（顺序无关）
                    if (inf1 == true_inf1 and inf2 == true_inf2) or \
                            (inf1 == true_inf2 and inf2 == true_inf1):
                        true_positives += 1
                        matched_true_indices.add(true_idx)
                        matched_pred_indices.add(pred_idx)

                        # 记录匹配详情
                        matches.append({
                            'predicted': pred,
                            'true': true_edge,
                            'pred_idx': pred_idx,
                            'true_idx': true_idx
                        })
                        break  # 找到匹配，跳出循环

    # 4. 计算指标
    total_true = len(true_hyperedges)
    total_pred = len(predicted_list)
    false_positives = total_pred - len(matched_pred_indices)
    false_negatives = total_true - len(matched_true_indices)

    # 避免除以零
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # 收集详细指标
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "predicted_count": total_pred,
        "true_count": total_true,
        "matches": matches,
        "matched_true_indices": list(matched_true_indices),
        "matched_pred_indices": list(matched_pred_indices)
    }

    # 打印评估结果
    print("\n===== 重构评估结果 =====")
    print(f"预测超边数: {metrics['predicted_count']}")
    print(f"真实超边数: {metrics['true_count']}")
    print(f"正确预测超边数 (TP): {metrics['true_positives']}")
    print(f"错误预测超边数 (FP): {metrics['false_positives']}")
    print(f"未预测到的真实超边数 (FN): {metrics['false_negatives']}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # 打印匹配的示例
    if matches:
        print("\n匹配的示例超边:")
        for i, match in enumerate(matches[:5]):
            pred = match['predicted']
            true = match['true']
            print(f"  {i + 1}. 预测: {pred['target']} ← ({pred['influencers'][0]}, {pred['influencers'][1]})")
            print(f"      真实: {true['target']} ← ({true['influencers'][0]}, {true['influencers'][1]})")

    # 打印未匹配的预测示例 (FP)
    if false_positives > 0:
        fp_examples = []
        for i, pred in enumerate(predicted_list):
            if i not in matched_pred_indices:
                fp_examples.append(pred)
                if len(fp_examples) >= 5:
                    break

        print("\n未匹配的示例预测超边 (FP):")
        for i, pred in enumerate(fp_examples):
            print(f"  {i + 1}. {pred['target']} ← ({pred['influencers'][0]}, {pred['influencers'][1]})")

    # 打印未预测到的真实超边示例 (FN)
    if false_negatives > 0:
        fn_examples = []
        for i, true in enumerate(true_hyperedges):
            if i not in matched_true_indices:
                fn_examples.append(true)
                if len(fn_examples) >= 5:
                    break

        print("\n未预测到的真实超边示例 (FN):")
        for i, true in enumerate(fn_examples):
            print(f"  {i + 1}. {true['target']} ← ({true['influencers'][0]}, {true['influencers'][1]})")

    return precision, recall, f1, metrics

# 使用示例
precision, recall, f1, metrics = evaluate_reconstruction(predicted_hyperedges, true_hyperedges)