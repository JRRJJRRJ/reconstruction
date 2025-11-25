from typing import List, Tuple


def load_triples(file_path: str) -> set:
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


def find_common_triples(file1: str, file2: str) -> List[Tuple[int, int, int]]:
    """
    返回两个文件中都存在的三元组（顺序无关）
    """
    triples1 = load_triples(file1)
    triples2 = load_triples(file2)

    common = triples1 & triples2  # 集合交集
    return list(common)


def calculate_f1(real_triples: set, pred_triples: set):
    """
    计算预测三元组相对于真实三元组的F1分数
    """
    TP = len(real_triples & pred_triples)  # 正确预测
    FP = len(pred_triples - real_triples)  # 错误预测
    FN = len(real_triples - pred_triples)  # 漏掉的

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1

if __name__ == "__main__":
    file1 = "./TEST/RealNet_result/hypertext2009/T=19000/hypertext2009_T19000_W60.txt"
    file2 = "hyT1000.txt"

    common_triples = find_common_triples(file1, file2)

    print(f"两个文件共有 {len(common_triples)} 个相同三元组：")
    for triple in common_triples:
        print(triple)

    real_triples = load_triples(file2)
    pred_triples = load_triples(file1)

    precision, recall, f1 = calculate_f1(real_triples, pred_triples)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 score: {f1:.4f}")