import matplotlib.pyplot as plt
import networkx as nx

# 读取 Paired_connection.txt 构建简单图
G_pair = nx.Graph()
with open('Paired_connection.txt', 'r') as f:
    for line in f:
        u, v = line.strip().split()
        G_pair.add_edge(u, v)

# 绘制简单图（成对拓扑）
plt.figure(figsize=(20, 12))
pos = nx.spring_layout(G_pair, k=0.5)  # 节点间距控制参数k

nx.draw_networkx_nodes(G_pair, pos, node_size=200, node_color='skyblue')
nx.draw_networkx_edges(G_pair, pos, edge_color='black', width=1)
nx.draw_networkx_labels(G_pair, pos, font_size=8)

plt.title("Paired Connection Graph")
plt.axis('off')
plt.tight_layout()
plt.savefig("paired_connection.png")
plt.show()


# 读取 High_connection.txt 构建高阶结构
G_high = nx.Graph()
hyperedges = []  # 存储每个高阶超边的节点组（多于两个）

with open('High_connection.txt', 'r') as f:
    for line in f:
        nodes = line.strip().split()
        hyperedges.append(nodes)
        # 为了节点可视化，添加所有点到图中
        for u in nodes:
            G_high.add_node(u)
        # 添加所有成对边（方便后续黑色绘制）
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                G_high.add_edge(nodes[i], nodes[j])

# 绘制高阶图
plt.figure(figsize=(100, 80))
pos = nx.spring_layout(G_high, k=0.8)  # 拉大间距以清晰显示高阶结构

# 基础节点和黑边
nx.draw_networkx_nodes(G_high, pos, node_size=150, node_color='lightgreen')
nx.draw_networkx_edges(G_high, pos, edge_color='black', width=0.5)
nx.draw_networkx_labels(G_high, pos, font_size=7)

# 红色高阶连接（仅高阶，不重复绘制黑边）
for nodes in hyperedges:
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            # 红线覆盖黑线
            nx.draw_networkx_edges(
                G_high,
                pos,
                edgelist=[(nodes[i], nodes[j])],
                edge_color='red',
                width=1.2,
                style='solid'
            )

plt.title("High-order Connection Graph", fontsize=12)
plt.axis('off')
plt.tight_layout()
plt.savefig("high_order_connection.png")
plt.show()
