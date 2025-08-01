import numpy as np
import json
from reconstruction.High_order_Reconstruct import (
    load_data,
    detect_communities,
    reconstruct_community,
    compute_node_score,
    reconstruct_node_global
)
from index.F1_score import F1


def run_reconstruction(ts_path, graph_path, true_edges_path, iterations=5):
    """
    运行网络重构实验并返回平均F1分数
    固定参数:
        random_seed=42
        order=3
        k=5 (社区数量)
        top_percent=0.05 (关键节点比例)
    """
    np.random.seed(42)  # 固定随机种子

    # 存储每次实验的结果
    metrics_history = {'precision': [], 'recall': [], 'f1': []}

    for i in range(iterations):
        # 加载数据 - 使用原始代码的处理方式
        df, G = load_data(ts_path, graph_path)

        # 确保df中所有节点都在图中 - 保持原始处理逻辑
        missing_in_G = set(df.columns) - set(G.nodes)
        if missing_in_G:
            G.add_nodes_from(missing_in_G)

        # 社区划分 - 使用原始参数设置
        communities, node_assignments = detect_communities(G, k=5, method='pcdms', random_state=42)

        # 社区内重构 - 使用order=3
        InResult = {}
        for community in communities:
            # 保持原始类型转换逻辑
            community = [int(node) for node in community]
            result = reconstruct_community(community, df, 1)  # order=3
            InResult.update(result)

        # 关键节点全局重构 - 保持原始计算逻辑
        node_scores = {}
        for node in G.nodes:
            score = compute_node_score(G, node, communities, k=3)
            node_scores[node] = score

        # 提取前5%的高分节点 - 保持原始逻辑
        n_top = max(1, int(len(node_scores) * 0.05))
        top_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)[:n_top]
        top_nodes = [str(int(n) + 1) for n, _ in top_nodes]  # 保持+1转换

        # 全局重构 - 保持原始逻辑
        nodes_candidate = [str(n) for n in G.nodes]
        global_results = {}
        for target in top_nodes:
            result = reconstruct_node_global(target, nodes_candidate, df)
            global_results.update(result)

        # 合并结果 - 使用原始合并函数
        merged_results = merge_results_with_type_conversion(global_results, InResult)

        # 计算F1分数
        precision, recall, f1, _ = F1(true_edges_path, merged_results)

        # 记录本次实验结果
        metrics_history['precision'].append(precision)
        metrics_history['recall'].append(recall)
        metrics_history['f1'].append(f1)

    # 计算平均结果
    avg_results = {
        'precision': np.mean(metrics_history['precision']),
        'recall': np.mean(metrics_history['recall']),
        'f1': np.mean(metrics_history['f1']),
        'all_iterations': metrics_history
    }

    return avg_results


def merge_results_with_type_conversion(global_results, InResult):
    """保持原始合并逻辑不变"""
    merged_results = {}

    # 转换局部重构结果
    for (target, influencer), weight in InResult.items():
        # 保持原始类型处理
        if isinstance(influencer, tuple):
            key = (str(target), tuple(sorted(str(i) for i in influencer)))
        else:
            key = (str(target), str(influencer))
        merged_results[key] = weight

    # 添加全局重构结果
    for (target, influencer), weight in global_results.items():
        # 保持原始类型处理
        if isinstance(influencer, tuple):
            key = (str(target), tuple(sorted(str(i) for i in influencer)))
        else:
            key = (str(target), str(influencer))

        # 保持原始合并策略
        if key in merged_results:
            if abs(weight) > abs(merged_results[key]):
                merged_results[key] = weight
        else:
            merged_results[key] = weight

    return merged_results