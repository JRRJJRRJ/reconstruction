import numpy as np
import networkx as nx
from scipy.optimize import minimize
import time
from collections import defaultdict

import numpy as np
import networkx as nx
from scipy.optimize import minimize
import time
from collections import defaultdict

np.random.seed(42)
import numpy as np
import networkx as nx
from scipy.optimize import minimize
import time
from collections import defaultdict


class PCDMS:
    def __init__(self, k, learning_rate=0.01, max_iters=100,
                 tol=0.005, init_method='random', verbose=False,
                 random_state=None):  # 添加随机状态参数
        self.k = k
        self.eta = learning_rate
        self.max_iters = max_iters
        self.tol = tol
        self.init_method = init_method
        self.verbose = verbose
        self.random_state = random_state  # 保存随机状态
        self.M = None
        self.communities = None
        self.node_assignments = None
        self.runtime = 0
        self.neighbor_indices = None

        # 创建可复现的随机数生成器
        if self.random_state is None:
            self.rng = np.random.default_rng()
        elif isinstance(self.random_state, int):
            self.rng = np.random.default_rng(self.random_state)
        else:
            self.rng = self.random_state

    def fit(self, G):
        start_time = time.time()
        self.G = G
        self.n_nodes = G.number_of_nodes()
        self.nodes = list(G.nodes())

        # 构建邻居字典和索引映射
        self.neighbors = {}
        self.node_to_index = {node: idx for idx, node in enumerate(self.nodes)}
        for node in self.nodes:
            self.neighbors[node] = list(G.neighbors(node))

        # 构建邻居索引列表
        self.neighbor_indices = {}
        for i, node in enumerate(self.nodes):
            self.neighbor_indices[i] = [self.node_to_index[n] for n in self.neighbors[node]]

        # 初始化M矩阵
        self._initialize_M()

        # 迭代优化
        prev_M = np.copy(self.M)
        converged = False

        for iter in range(self.max_iters):
            # 遍历所有节点
            for i in range(self.n_nodes):
                # 使用带梯度的优化
                res = minimize(
                    fun=lambda mu: self._node_objective(mu, i),
                    jac=lambda mu: self._node_gradient(mu, i),
                    x0=self.M[i, :],
                    method='L-BFGS-B',
                    bounds=[(0, None)] * self.k,
                    options={'maxiter': 10}
                )
                self.M[i, :] = res.x

            # 计算最大元素变化
            max_delta = np.max(np.abs(self.M - prev_M))
            if self.verbose:
                print(f"Iteration {iter + 1}, Max Delta: {max_delta:.6f}")

            if max_delta < self.tol:
                converged = True
                if self.verbose:
                    print(f"Converged after {iter + 1} iterations")
                break

            prev_M = np.copy(self.M)

        # 分配社区
        self._assign_communities()

        self.runtime = time.time() - start_time
        if self.verbose and not converged:
            print(f"Reached max iterations ({self.max_iters}) without full convergence")

        return self.node_assignments

    def _initialize_M(self):
        """改进的初始化方法（添加随机种子控制）"""
        self.M = np.zeros((self.n_nodes, self.k))

        if self.init_method == 'random':
            # 使用可复现的随机数生成器
            for i in range(self.n_nodes):
                comm_idx = self.rng.integers(0, self.k)
                self.M[i, comm_idx] = self.rng.random()

        elif self.init_method == 'clustering':
            # 基于聚类系数的初始化
            clustering = nx.clustering(self.G)
            for i, node in enumerate(self.nodes):
                cluster_coeff = clustering.get(node, 0.1)
                comm_idx = self.rng.integers(0, self.k)  # 使用rng
                self.M[i, comm_idx] = cluster_coeff

        elif self.init_method == 'lmn':
            # 简化的局部最小邻域初始化
            degrees = dict(self.G.degree())
            for i, node in enumerate(self.nodes):
                # 选择度最小的邻居
                neighbors = self.neighbors[node]
                if neighbors:
                    min_deg_neighbor = min(neighbors, key=lambda n: degrees[n])
                    min_idx = self.node_to_index[min_deg_neighbor]
                    comm_idx = np.argmax(self.M[min_idx, :])
                    self.M[i, comm_idx] = 1.0
                else:
                    comm_idx = self.rng.integers(0, self.k)  # 使用rng
                    self.M[i, comm_idx] = 1.0

        # 归一化
        row_max = self.M.max(axis=1)
        row_max[row_max == 0] = 1
        self.M = self.M / row_max[:, np.newaxis]

    # 其余方法保持不变...

    def _motif_probability(self, u_idx, v1_idx, v2_idx, c):
        """修正的概率计算"""
        Mu = self.M[u_idx, c]
        Mv1 = self.M[v1_idx, c]
        Mv2 = self.M[v2_idx, c]

        # 计算P_c(u, v1)
        dot_products_v1 = [np.dot(Mu, self.M[vi_idx, c])
                           for vi_idx in self.neighbor_indices[u_idx]]
        exp_terms_v1 = np.exp(-np.array(dot_products_v1))
        denom_v1 = np.sum(exp_terms_v1)
        num_v1 = np.exp(-np.dot(Mu, Mv1))
        p_v1 = num_v1 / (denom_v1 + 1e-10)

        # 计算P_c(u, v2)
        dot_products_v2 = [np.dot(Mu, self.M[vi_idx, c])
                           for vi_idx in self.neighbor_indices[u_idx]]
        exp_terms_v2 = np.exp(-np.array(dot_products_v2))
        denom_v2 = np.sum(exp_terms_v2)
        num_v2 = np.exp(-np.dot(Mu, Mv2))
        p_v2 = num_v2 / (denom_v2 + 1e-10)

        return p_v1 * p_v2

    def _node_objective(self, mu, u_idx):
        """修正的目标函数"""
        original_mu = np.copy(self.M[u_idx, :])
        self.M[u_idx, :] = mu

        loss = 0
        u = self.nodes[u_idx]
        neighbors = self.neighbors[u]
        n_neighbors = len(neighbors)

        # 遍历所有邻居对
        for i in range(n_neighbors):
            v1 = neighbors[i]
            v1_idx = self.node_to_index[v1]

            for j in range(i + 1, n_neighbors):
                v2 = neighbors[j]
                v2_idx = self.node_to_index[v2]

                # 检查三角模体是否存在
                motif_exists = 1 if self.G.has_edge(v1, v2) else 0

                for c in range(self.k):
                    p = self._motif_probability(u_idx, v1_idx, v2_idx, c)

                    # 根据模体存在性计算似然
                    if motif_exists:
                        if p > 0:
                            loss -= np.log(p)
                    else:
                        if p < 1:
                            loss -= np.log(1 - p)

        self.M[u_idx, :] = original_mu
        return loss

    def _node_gradient(self, mu, u_idx):
        """实现论文中的梯度公式"""
        original_mu = np.copy(self.M[u_idx, :])
        self.M[u_idx, :] = mu

        grad = np.zeros(self.k)
        u = self.nodes[u_idx]
        neighbors = self.neighbors[u]

        for c in range(self.k):
            # 第一项和第二项
            term = np.zeros(self.k)

            # 第三项：分子和分母
            numerator = np.zeros(self.k)
            denominator = 0

            for vi in neighbors:
                vi_idx = self.node_to_index[vi]
                Mvi = self.M[vi_idx, c]
                dot_prod = np.dot(mu[c], Mvi)
                exp_term = np.exp(-dot_prod)

                # 只考虑存在的边（论文公式）
                if self.G.has_edge(u, vi):
                    term += Mvi

                numerator -= Mvi * exp_term
                denominator += exp_term

            # 组合梯度分量
            grad[c] = term[c] + (numerator[c] / (denominator + 1e-10))

        self.M[u_idx, :] = original_mu
        return grad

    def _assign_communities(self):
        """社区分配（保持不变）"""
        threshold = np.median(self.M)
        self.communities = [[] for _ in range(self.k)]
        self.node_assignments = {}

        # 第一遍分配
        for i, node in enumerate(self.nodes):
            for c in range(self.k):
                if self.M[i, c] > threshold:
                    self.communities[c].append(node)

        # 确保每个节点都有社区
        for i, node in enumerate(self.nodes):
            assigned = False
            max_comm = np.argmax(self.M[i, :])

            for c, comm in enumerate(self.communities):
                if node in comm:
                    assigned = True
                    if node not in self.node_assignments or self.M[i, c] > self.M[i, self.node_assignments[node]]:
                        self.node_assignments[node] = c

            if not assigned:
                self.communities[max_comm].append(node)
                self.node_assignments[node] = max_comm

        return self.node_assignments

    # 其余辅助方法保持不变...
    def get_community_membership(self):
        """获取节点社区分配结果"""
        return self.node_assignments

    def get_communities(self):
        """获取社区列表"""
        return self.communities

    def get_modularity(self):
        """计算模块度分数"""
        if not self.communities:
            raise ValueError("Run fit() first")
        return nx.algorithms.community.modularity(self.G, self.communities)

    def print_summary(self):
        """打印算法摘要"""
        print("\n===== PCDMS 算法结果 =====")
        print(f"运行时间: {self.runtime:.4f}秒")
        print(f"社区数量: {self.k}")
        print(f"模块度: {self.get_modularity():.4f}")
        print("\n社区分布:")
        for i, comm in enumerate(self.communities):
            print(f"社区 {i}: {len(comm)}个节点")
        print(f"\n节点社区分配示例:")
        for node, comm in list(self.node_assignments.items())[:5]:
            print(f"节点 {node} -> 社区 {comm}")