import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import networkx as nx


def reconstruct_network_topology(csv_file, alpha=0.01, threshold=0.05, visualize=True):
    """
    使用压缩感知重构网络拓扑

    参数:
    csv_file (str): CSV文件路径
    alpha (float): Lasso回归的正则化强度
    threshold (float): 邻接矩阵的阈值
    visualize (bool): 是否可视化网络

    返回:
    adjacency_matrix (ndarray): 重构的邻接矩阵
    """
    # 1. 读取时间序列数据
    data = pd.read_csv(../data/synth)
    print(f"数据维度: {data.shape[0]}个时间点, {data.shape[1]}个节点")

    # 2. 数据预处理
    scaler = StandardScaler()
    X = scaler.fit_transform(data.values)
    n_time, n_nodes = X.shape

    # 3. 计算时间导数 (一阶差分)
    dX = np.diff(X, axis=0)

    # 4. 压缩感知重构网络
    adjacency_matrix = np.zeros((n_nodes, n_nodes))

    for i in range(n_nodes):
        # 目标: 节点i的变化率
        y = dX[:, i]

        # 特征: 所有节点在上一时刻的状态
        # 注意: 使用时间导数之前的状态
        Phi = X[:-1, :]

        # 使用Lasso回归 (L1正则化实现压缩感知)
        model = Lasso(alpha=alpha, max_iter=10000, tol=1e-4)
        model.fit(Phi, y)

        # 存储权重 (第i个节点的连接权重)
        adjacency_matrix[i, :] = model.coef_

    # 5. 应用阈值处理
    adjacency_matrix[np.abs(adjacency_matrix) < threshold] = 0

    # 6. 可视化结果
    if visualize:
        plt.figure(figsize=(12, 5))

        # 邻接矩阵热力图
        plt.subplot(121)
        plt.imshow(adjacency_matrix, cmap='coolwarm', interpolation='none')
        plt.colorbar(label='连接强度')
        plt.title("重构的邻接矩阵")
        plt.xlabel("源节点")
        plt.ylabel("目标节点")

        # 网络图
        plt.subplot(122)
        G = nx.DiGraph(adjacency_matrix)
        pos = nx.spring_layout(G)

        # 绘制节点和边
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue')
        nx.draw_networkx_edges(G, pos, edge_color='gray', width=1.5,
                               arrows=True, arrowsize=15)
        nx.draw_networkx_labels(G, pos, font_size=10)

        plt.title("重构的网络拓扑")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    return adjacency_matrix


# 使用示例
if __name__ == "__main__":
    # 替换为你的CSV文件路径
    input_csv = "node_time_series.csv"

    # 重构网络拓扑
    adj_matrix = reconstruct_network_topology(
        csv_file=input_csv,
        alpha=0.05,  # 正则化强度 (越大越稀疏)
        threshold=0.03,  # 连接阈值
        visualize=True
    )

    # 打印重构的邻接矩阵
    print("\n重构的邻接矩阵:")
    print(adj_matrix)