import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

# 设置风格和字体 - 与之前代码统一
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子
np.random.seed(42)

# 定义网络参数
networks = [
    {"name": "Hypertext2009", "n": 85, "type": "sparse"},
    {"name": "Thiers12", "n": 156, "type": "modular"},
    {"name": "InVS15", "n": 211, "type": "mixed"},
    {"name": "LyonSchool", "n": 222, "type": "modular"}
]

for net in networks:
    G = nx.Graph()
    n_nodes = net['n']

    # 生成不同拓扑
    if net['type'] == 'sparse':
        G = nx.erdos_renyi_graph(n_nodes, p=0.06, seed=42)

    elif net['type'] == 'modular':
        sizes = [n_nodes // 3, n_nodes // 3, n_nodes - 2 * (n_nodes // 3)]
        p_matrix = [[0.15, 0.01, 0.01],
                    [0.01, 0.15, 0.01],
                    [0.01, 0.01, 0.15]]
        G = nx.stochastic_block_model(sizes, p_matrix, seed=42)

    elif net['type'] == 'mixed':
        G = nx.erdos_renyi_graph(n_nodes, p=0.04, seed=42)
        for _ in range(20):
            nodes = np.random.choice(n_nodes, 3, replace=False)
            G.add_edges_from([(nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[2], nodes[0])])

    # 获取布局
    pos = nx.spring_layout(G, seed=42)

    # 绘制
    plt.figure(figsize=(8, 8))  # 单独一张图
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='dimgray')  # 边颜色加深，alpha略高

    # 找孤立节点
    isolates = list(nx.isolates(G))

    # 普通节点（非孤立节点）颜色加深
    non_isolates = list(set(G.nodes()) - set(isolates))
    nx.draw_networkx_nodes(G, pos,
                           nodelist=non_isolates,
                           node_color='royalblue',  # 深一点
                           node_size=15)

    # 绘制更多三角形模体（深黄色）
    triangles = [clique for clique in nx.enumerate_all_cliques(G) if len(clique) == 3]
    for t in triangles[:120]:  # 最多显示 120 个
        nodes_pos = [pos[n] for n in t]
        poly = Polygon(nodes_pos, closed=True, fill=True, alpha=0.7, color='goldenrod')
        plt.gca().add_patch(poly)

    # 标题使用 Times New Roman 字体，字体大小与之前一致
    plt.title(net['name'], fontsize=26, fontweight='bold')  # 使用18号字体，加粗
    plt.axis('off')

    # 保存为矢量图 (SVG)
    plt.savefig(f"{net['name']}_structure1.svg", format="svg", bbox_inches='tight')

    # 同时保存为 PDF
    plt.savefig(f"{net['name']}_structure1.pdf", format="pdf", bbox_inches='tight')
    plt.close()

print("所有网络图已保存为 SVG 矢量图。")