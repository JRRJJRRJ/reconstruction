import numpy as np
import pandas as pd
import networkx as nx
from sklearn.linear_model import Lasso, LassoCV
from sklearn.cluster import KMeans
import itertools
from typing import List, Tuple, Dict
# reconstruction/High_order-Reconstruct.py
from community_detection.PCDMS import PCDMS
import warnings
from sklearn.exceptions import ConvergenceWarning


def load_data(ts_path, graph_path):
    """
    读取时间序列数据和图结构
    """
    # 正确读取时间序列：无列名、无时间列
    df = pd.read_csv(ts_path, header=None)
    df.columns = list(map(str, range(df.shape[1])))  # 或用 range(df.shape[1]) 得到整数列

    # 正确读取图
    try:
        G = nx.read_edgelist(graph_path, nodetype=str)
    except Exception:
        G = nx.read_edgelist(graph_path, delimiter=',', nodetype=str)

    return df, G


def detect_communities(G, k=5, method='pcdms', random_state=None):
    """
    使用PCDMS进行社区检测。

    参数:
        G: networkx.Graph，输入图
        k: int，社区数量
        method: str，当前仅支持 'pcdms'
        random_state: 随机种子

    返回:
        communities: List[List[node]] 社区列表
        node_assignments: Dict[node, community_id] 节点到社区编号的映射
    """
    if method == 'pcdms':
        model = PCDMS(k=k, init_method='clustering', verbose=True, random_state=random_state)
        node_assignments = model.fit(G)
        communities = model.get_communities()
        return communities, node_assignments
    else:
        raise NotImplementedError(f"不支持的社区检测方法: {method}")


def reconstruct_community(community: List[str], df: pd.DataFrame, needshow: int = 0) -> Dict:
    """
    对一个社区中的每个节点进行三阶高阶相互作用重构 (使用Lasso)。
    参数:
        community (list of str): 社区内节点列表
        df (pd.DataFrame): 时间序列数据 (行:时间, 列:节点)
    返回:
        results (dict): 键为 (target, influencer)，其中 influencer 为单节点或节点二元组, 值为回归系数 (float)。
    """
    results = {}
    community = sorted(community)
    # 提取社区时间序列
    df_comm = df[list(map(str, community))]
    X = df_comm.values  # shape (T, m)
    T, m = X.shape
    if T < 3:
        return results  # 时间点不足以做中心差分
    # 计算导数：中心差分 dX/dt ≈ (X[t+1] - X[t-1]) / 2
    dXdt = (X[2:T, :] - X[0:T-2, :]) / 2.0  # shape (T-2, m)
    for i, target in enumerate(community):
        dY = dXdt[:, i]  # 目标节点的导数序列 (长度 T-2)
        # 社区内部其他节点列表
        others = [n for n in community if n != target]
        if not others:
            continue
        # 自变量矩阵：取原数据的时间点1到T-2行作为输入
        TS = X[1:T-1, :]  # shape (T-2, m)
        # 线性项 (单节点影响)
        others_idx = [community.index(n) for n in others]
        X_lin = TS[:, others_idx]  # shape (T-2, len(others))
        # 二元组乘积项 (对应三元组超边)
        pair_indices = list(itertools.combinations(range(len(others_idx)), 2))
        X_pair = []
        for (p, q) in pair_indices:
            X_pair.append(TS[:, others_idx[p]] * TS[:, others_idx[q]])
        if X_pair:
            X_pair = np.stack(X_pair, axis=1)  # shape (T-2, num_pairs)
            X_design = np.hstack((X_lin, X_pair))
        else:
            X_design = X_lin
        if X_design.size == 0:
            continue
        # 使用LassoCV回归求解系数
        try:
            lasso = LassoCV(cv=5, n_alphas=10, max_iter=10000).fit(X_design, dY)
            coefs = lasso.coef_
        except Exception:
            from sklearn.linear_model import Lasso
            lasso = Lasso(alpha=0.01, max_iter=10000).fit(X_design, dY)
            coefs = lasso.coef_
        # 提取非零系数
        # 线性项 -> (target, node)
        for j, node_j in enumerate(others):
            if abs(coefs[j]) > 1e-6:
                results[(target, node_j)] = coefs[j]
        # 二元组乘积项 -> (target, (node_p, node_q))
        num_lin = len(others)
        for k, (p, q) in enumerate(pair_indices):
            coeff = coefs[num_lin + k]
            if abs(coeff) > 1e-6:
                node_p = others[p]
                node_q = others[q]
                hyperedge = tuple(sorted((node_p, node_q)))
                results[(target, hyperedge)] = coeff

    return results

def compute_pc(node: str, communities: List[List[str]]) -> float:
    """
    计算节点参与度系数 PC (简单定义为节点所属社区数量)。
    """
    return float(sum(1 for comm in communities if node in comm))

def compute_ac(G: nx.Graph, node: str) -> float:
    """
    计算节点聚类系数 AC（使用NetworkX聚类系数）。
    """
    return nx.clustering(G, node)

def compute_mps_k(G: nx.Graph, node: str, k: int = 3) -> float:
    """
    计算节点参与的 k 阶团体数 (k=3 时为三角形数)。
    """
    if k == 3:
        return float(nx.triangles(G, node))
    else:
        # 其它阶的团体数计算作为扩展接口，此处返回0
        return 0.0

def compute_node_score(G: nx.Graph, node: str, communities: List[List[str]], k: int = 3) -> float:
    """
    计算节点综合评分 w(v) = PC + AC + MPS_k。
    """
    pc = compute_pc(node, communities)
    ac = compute_ac(G, node)
    mps = compute_mps_k(G, node, k)
    return pc + ac + mps


def reconstruct_node_global(target: str, nodes: List[str], df: pd.DataFrame) -> Dict:
    # 保持原始节点顺序!!!
    nodes_original = nodes[:]  # 保留原始顺序
    nodes_sorted = sorted(nodes)  # 仅用于数据对齐

    # 检查基本条件
    if target not in nodes_sorted:
        return {"target not in nodes_sorted"}
    df_all = df[nodes_sorted]  # 按排序后顺序取数据
    X = df_all.values
    T, N = X.shape
    if T < 3:
        return {" T < 3"}

    # 计算导数 (中心差分)
    dXdt = (X[2:] - X[:-2]) / 2.0  # (T-2, N)
    target_idx = nodes_sorted.index(target)
    dY = dXdt[:, target_idx]

    # 使用原始节点顺序构建特征矩阵!!!
    others = [n for n in nodes_original if n != target]  # 保持原始顺序
    if not others:
        return {"not others"}

    # 准备特征矩阵 (T-2时间点)
    TS = X[1:-1]  # (T-2, N)

    # 获取其他节点在排序后列表中的索引
    other_indices = [nodes_sorted.index(n) for n in others]
    X_lin = TS[:, other_indices]  # 线性项

    # 二阶交互项
    X_pair = []
    pair_nodes = []  # 记录交互项对应的节点对
    for (idx1, n1), (idx2, n2) in itertools.combinations(enumerate(others), 2):
        X_pair.append(TS[:, nodes_sorted.index(n1)] * TS[:, nodes_sorted.index(n2)])
        pair_nodes.append((n1, n2))

    # 合并特征矩阵
    X_design = np.hstack([X_lin] + [np.array(X_pair).T] if X_pair else [])

    # 检查特征维度有效性
    if X_design.size == 0 or X_design.shape[0] < 2:
        return {"X_design.size == 0 or X_design.shape[0] < 2"}

    # 动态调整正则化参数
    n_samples, n_features = X_design.shape
    max_features = max(1, min(n_samples - 1, n_features // 2))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        try:
            # 减少正则化强度: 降低eps, 增加n_alphas
            lasso = LassoCV(
                cv=min(5, n_samples),
                n_alphas=50,
                eps=1e-4,  # 扩展alpha搜索范围
                max_iter=5000,
                selection='random',  # 提高稳定性
                tol=1e-4
            ).fit(X_design, dY)
            coefs = lasso.coef_
        except Exception:
            # 使用更弱的正则化回退
            lasso = Lasso(
                alpha=1e-5,  # 大幅降低正则化强度
                max_iter=10000,
                tol=1e-4,
                warm_start=True
            ).fit(X_design, dY)
            coefs = lasso.coef_

    # 提取有效特征
    results = {}
    threshold = 1e-7  # 降低系数阈值

    # 线性项 (单节点)
    for j, node_j in enumerate(others):
        if abs(coefs[j]) > threshold:
            results[(target, node_j)] = coefs[j]

    # 二阶交互项
    for k, (n1, n2) in enumerate(pair_nodes):
        idx = len(others) + k
        if idx < len(coefs) and abs(coefs[idx]) > threshold:
            # 保持原始节点顺序
            results[(target, (n1, n2))] = coefs[idx]

    return results
def merge_results(community_results: List[Dict], key_results: Dict) -> Dict:
    """
    合并所有社区的重构结果和关键节点的全局重构结果。
    相同 (target, influencer) 在多个社区出现时取平均，关键节点结果覆盖社区结果。
    """
    merged = {}
    temp = {}
    # 收集社区结果
    for res in community_results:
        for (target, influ), w in res.items():
            temp.setdefault((target, influ), []).append(w)
    # 取平均
    for (tgt, inf), weights in temp.items():
        merged[(tgt, inf)] = float(np.mean(weights))
    # 关键节点结果覆盖
    for (target, influ), w in key_results.items():
        merged[(target, influ)] = w
    return merged

def build_Y_matrix(merged_results: Dict, nodes: List[str]):
    """
    构建Y矩阵：维度为 N x (N+C)，其中 N = len(nodes)，C = 节点二元组总数。
    前N列对应成对作用 (pair influence)，后C列对应三元组超边作用。
    返回:
        Y (np.ndarray): 构建的矩阵
        col_names (list): 列名列表
        nodes_sorted (list): 排序后的节点列表 (对应行)
    """
    nodes_sorted = sorted(nodes)
    N = len(nodes_sorted)
    # 所有节点二元组组合 (作为潜在超边)
    triples = list(itertools.combinations(nodes_sorted, 2))
    col_names = []
    # 前 N 列：对作用，按照节点顺序
    for node in nodes_sorted:
        col_names.append(f"pair_{node}")
    # 后 C 列：二元组 (与目标一起形成三元组超边)
    for (i, j) in triples:
        col_names.append(f"triple_{i}_{j}")
    C = len(triples)
    Y = np.zeros((N, N + C))
    node_idx = {node: idx for idx, node in enumerate(nodes_sorted)}
    for (target, influ), w in merged_results.items():
        if isinstance(influ, tuple):
            # 超边情况：(target, (i,j)) 对应列 "triple_i_j"
            # 如果 influencer = (i, j)，超边为 (target, i, j)
            i, j = influ
            # 找到对应列
            if (i, j) in triples:
                col = triples.index((i, j)) + N
                row = node_idx[target]
                Y[row, col] = w
        else:
            # 成对作用：(target, j) 对应 pair_j 列
            row = node_idx[target]
            col = node_idx[influ]
            Y[row, col] = w
    return Y, col_names, nodes_sorted

def extract_hyperedges(Y: np.ndarray, nodes: List[str], col_names: List[str]) -> List[Tuple]:
    """
    使用 KMeans 聚类确定阈值，从 Y 矩阵中提取强度超过阈值的超边（三元组）。
    返回超边列表，每个超边为节点元组 (不含目标，示例输出为二元组标识)。
    """
    N, total_cols = Y.shape
    C = total_cols - N
    triple_cols = []
    weights = []
    # 收集每列的非零值平均幅度作为该列超边强度
    for col in range(N, total_cols):
        col_vals = Y[:, col]
        nz = col_vals[np.nonzero(col_vals)]
        if nz.size > 0:
            w_avg = np.mean(np.abs(nz))
            triple_cols.append(col)
            weights.append(w_avg)
    if not weights:
        return []
    weights = np.array(weights).reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(weights)
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = sum(centers) / 2.0
    # 选取阈值以上的列
    selected = [col for col, w in zip(triple_cols, weights.flatten()) if w >= threshold]
    hyperedges = []
    for col in selected:
        name = col_names[col]
        parts = name.split('_')
        if parts[0] == 'triple' and len(parts) == 3:
            _, i, j = parts
            hyperedges.append((i, j))
    return hyperedges

def evaluate_hyperedges(predicted: List[Tuple], true: List[Tuple]):
    """
    计算预测超边与真实超边的 Precision, Recall, F1，并打印结果。
    """
    pred_set = set(tuple(sorted(x)) for x in predicted)
    true_set = set(tuple(sorted(x)) for x in true)
    tp = len(pred_set & true_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(true_set) if true_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision+recall)>0 else 0.0
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

