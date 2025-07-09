def F1(true_file_path, merged_results):
    """
    评估预测结果与真实高阶相互作用的匹配情况（针对三元超边）。

    参数：
        true_file_path (str): 真实高阶相互作用文件路径（如 'High_connection.txt'）。
                             每行格式为：target node1 node2
        merged_results (dict): 预测结果字典，键为 (target, (node1, node2))，值为预测值。

    返回：
        precision (float): 精确率
        recall (float): 召回率
        f1_score (float): F1 分数
        metrics (dict): 包含 TP、FP、FN、precision、recall、f1 的字典
    """
    # 读取真实高阶相互作用
    true_set = set()
    with open(true_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            target = parts[0]
            node1, node2 = parts[1], parts[2]
            sorted_nodes = tuple(sorted([node1, node2]))
            true_set.add((target, sorted_nodes))

    # 处理预测结果
    pred_set = set()
    for key in merged_results.keys():
        target = key[0]
        node_pair = key[1]
        sorted_nodes = tuple(sorted(node_pair))
        pred_set.add((target, sorted_nodes))

    # 计算指标
    TP = len(true_set & pred_set)
    FP = len(pred_set - true_set)
    FN = len(true_set - pred_set)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # 打印评估结果
    print("===== 简单集合评估结果 =====")
    print(f"TP: {TP}, FP: {FP}, FN: {FN}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")

    # 返回指标
    metrics = {
        "true_positives": TP,
        "false_positives": FP,
        "false_negatives": FN,
        "precision": precision,
        "recall": recall,
        "f1": f1_score,
        "predicted_count": len(pred_set),
        "true_count": len(true_set),
    }

    return precision, recall, f1_score, metrics
