from typing import List, Tuple, Set


def load_triples(file_path: str) -> Set[Tuple[int, int, int]]:
    """
    读取三元组文件，将每行三元组排序后存入集合
    """
    triples = set()
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue  # 跳过格式不对的行
            a, b, c = map(int, parts)
            triples.add(tuple(sorted((a, b, c))))
    return triples


def calculate_f1(real_triples: Set[Tuple[int, int, int]],
                 pred_triples: Set[Tuple[int, int, int]]) -> Tuple[float, float, float]:
    """
    计算预测三元组相对于真实三元组的F1分数
    返回 (precision, recall, f1)
    """
    TP = len(real_triples & pred_triples)  # 正确预测
    FP = len(pred_triples - real_triples)  # 错误预测
    FN = len(real_triples - pred_triples)  # 漏掉的

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1


def get_f1(pred_file: str, real_file: str) -> float:
    """
    读取预测文件和真实文件，返回F1分数
    """
    pred_triples = load_triples(pred_file)
    real_triples = load_triples(real_file)
    _, _, f1 = calculate_f1(real_triples, pred_triples)
    return f1


# 测试用例
if __name__ == "__main__":
    file1 = ""
    file2 = ""

    f1 = get_f1(file1, file2)
    print(f"F1 score: {f1:.4f}")
