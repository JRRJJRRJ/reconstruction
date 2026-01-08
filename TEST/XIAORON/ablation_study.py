import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import networkx as nx
from reconstruction.High_order_Reconstruct import (
    load_data,
    detect_communities,
    reconstruct_community,
    compute_ac,
    compute_mps_k,
    reconstruct_node_global
)
from index.F1_score import F1
from GetF1 import merge_results_with_type_conversion

# 设置风格和字体（尽量贴近你给的论文图）
# - 使用 serif + Times New Roman
# - mathtext 设为 regular/stix，避免出现与正文不一致的“数学字体”
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "mathtext.default": "regular",
    # 全局字号基准（其余局部再按需覆盖）
    "font.size": 18,
    "axes.linewidth": 1.5,
    "axes.unicode_minus": False,
})

def run_full_hrck(ts_path, graph_path, true_edges_path, k=5, top_percent=0.05, iterations=1):
    """
    配置1: 完整HRCK框架（红色）
    包含：社区划分 + 完整多维权值函数(PC+AC+MPS) + 局部重构 + 全局重构
    """
    np.random.seed(42)
    metrics_history = {'precision': [], 'recall': [], 'f1': []}
    
    for i in range(iterations):
        df, G = load_data(ts_path, graph_path)
        missing_in_G = set(df.columns) - set(G.nodes)
        if missing_in_G:
            G.add_nodes_from(missing_in_G)
        
        # 社区划分
        communities, _ = detect_communities(G, k=k, method='pcdms', random_state=42)
        
        # 社区内局部重构
        InResult = {}
        for community in communities:
            community = [int(node) for node in community]
            result = reconstruct_community(community, df, 1)
            InResult.update(result)
        
        # 关键节点识别 - 完整多维权值函数(PC+AC+MPS)
        from reconstruction.High_order_Reconstruct import compute_pc, compute_node_score
        node_scores = {}
        for node in G.nodes:
            score = compute_node_score(G, node, communities, k=3)  # PC+AC+MPS
            node_scores[node] = score
        
        # 提取关键节点
        n_top = max(1, int(len(node_scores) * top_percent))
        top_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)[:n_top]
        top_nodes = [str(int(n) + 1) for n, _ in top_nodes]
        
        # 关键节点全局重构
        nodes_candidate = [str(n) for n in G.nodes]
        global_results = {}
        for target in top_nodes:
            result = reconstruct_node_global(target, nodes_candidate, df)
            global_results.update(result)
        
        # 合并结果
        merged_results = merge_results_with_type_conversion(global_results, InResult)
        
        # 计算F1分数
        precision, recall, f1, _ = F1(true_edges_path, merged_results)
        metrics_history['precision'].append(precision)
        metrics_history['recall'].append(recall)
        metrics_history['f1'].append(f1)
    
    return {
        'precision': np.mean(metrics_history['precision']),
        'recall': np.mean(metrics_history['recall']),
        'f1': np.mean(metrics_history['f1']),
        'all_iterations': metrics_history
    }

def run_no_community(ts_path, graph_path, true_edges_path, top_percent=0.05, iterations=1):
    """
    配置2: 无社区划分组件（蓝色）
    去掉社区划分，多维权值函数去掉PC，仅根据关键节点进行全局重构
    """
    np.random.seed(42)
    metrics_history = {'precision': [], 'recall': [], 'f1': []}
    
    for i in range(iterations):
        df, G = load_data(ts_path, graph_path)
        missing_in_G = set(df.columns) - set(G.nodes)
        if missing_in_G:
            G.add_nodes_from(missing_in_G)
        
        # 不进行社区划分，关键节点识别仅使用AC+MPS（去掉PC）
        node_scores = {}
        for node in G.nodes:
            # 仅使用AC和MPS，不包含PC
            ac = compute_ac(G, node)
            mps = compute_mps_k(G, node, k=3)
            node_scores[node] = ac + mps  # 去掉PC
        
        # 提取关键节点
        n_top = max(1, int(len(node_scores) * top_percent))
        top_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)[:n_top]
        top_nodes = [str(int(n) + 1) for n, _ in top_nodes]
        
        # 仅全局重构（无局部重构）
        nodes_candidate = [str(n) for n in G.nodes]
        global_results = {}
        for target in top_nodes:
            result = reconstruct_node_global(target, nodes_candidate, df)
            global_results.update(result)
        
        # 仅使用全局结果
        merged_results = {}
        for (target, influencer), weight in global_results.items():
            if isinstance(influencer, tuple):
                key = (str(target), tuple(sorted(str(i) for i in influencer)))
            else:
                key = (str(target), str(influencer))
            merged_results[key] = weight
        
        # 计算F1分数
        precision, recall, f1, _ = F1(true_edges_path, merged_results)
        metrics_history['precision'].append(precision)
        metrics_history['recall'].append(recall)
        metrics_history['f1'].append(f1)
    
    return {
        'precision': np.mean(metrics_history['precision']),
        'recall': np.mean(metrics_history['recall']),
        'f1': np.mean(metrics_history['f1']),
        'all_iterations': metrics_history
    }

def run_no_key_nodes(ts_path, graph_path, true_edges_path, k=5, iterations=1):
    """
    配置3: 无关键节点（绿色）
    仅进行社区内局部重构，不进行关键节点全局重构
    """
    np.random.seed(42)
    metrics_history = {'precision': [], 'recall': [], 'f1': []}
    
    for i in range(iterations):
        df, G = load_data(ts_path, graph_path)
        missing_in_G = set(df.columns) - set(G.nodes)
        if missing_in_G:
            G.add_nodes_from(missing_in_G)
        
        # 社区划分
        communities, _ = detect_communities(G, k=k, method='pcdms', random_state=42)
        
        # 仅社区内局部重构
        InResult = {}
        for community in communities:
            community = [int(node) for node in community]
            result = reconstruct_community(community, df, 1)
            InResult.update(result)
        
        # 不进行关键节点全局重构，仅使用局部结果
        merged_results = {}
        for (target, influencer), weight in InResult.items():
            if isinstance(influencer, tuple):
                key = (str(target), tuple(sorted(str(i) for i in influencer)))
            else:
                key = (str(target), str(influencer))
            merged_results[key] = weight
        
        # 计算F1分数
        precision, recall, f1, _ = F1(true_edges_path, merged_results)
        metrics_history['precision'].append(precision)
        metrics_history['recall'].append(recall)
        metrics_history['f1'].append(f1)
    
    return {
        'precision': np.mean(metrics_history['precision']),
        'recall': np.mean(metrics_history['recall']),
        'f1': np.mean(metrics_history['f1']),
        'all_iterations': metrics_history
    }

def plot_ablation_results(results_dict, save_path='ablation_study.png'):
    """
    绘制消融实验结果对比图
    
    参数:
        results_dict: 字典，格式为 {
            'network_name': {
                'full_hrck': f1_score,
                'no_community': f1_score,
                'no_key_nodes': f1_score
            }
        }
    """
    # 现在将多个网络“拼接成一幅图”：x轴=网络名称；柱体=三种消融配置
    networks = list(results_dict.keys())
    n_networks = len(networks)

    # 增加图的高度，避免画面“扁平”
    fig, ax = plt.subplots(1, 1, figsize=(6 * max(2, n_networks), 6.8), dpi=400)

    # 三种方法（内部key不变）
    methods = ["full_hrck", "no_community", "no_key_nodes"]
    display_labels = {
        "full_hrck": "Full_HRCK",
        "no_community": "No_community",
        "no_key_nodes": "No_key_nodes",
    }

    # 柱体样式：白底 + 彩色边框 + 纹理
    method_styles = {
        # hatch 字符越多线越密：这里加密一些
        "full_hrck": {"edgecolor": "#f1c40f", "hatch": "///", "label": display_labels["full_hrck"]},
        "no_community": {"edgecolor": "#3498db", "hatch": "|||", "label": display_labels["no_community"]},
        "no_key_nodes": {"edgecolor": "#2ecc71", "hatch": "---", "label": display_labels["no_key_nodes"]},
    }

    x = np.arange(n_networks)
    # 每个网络一组，组内3根柱：留一点点缝隙
    bar_width = 0.26
    offset = 0.28
    offsets = [-offset, 0.0, offset]

    for m_idx, method in enumerate(methods):
        vals = [results_dict[net].get(method, 0) for net in networks]
        style = method_styles[method]
        ax.bar(
            x + offsets[m_idx],
            vals,
            width=bar_width,
            facecolor="white",
            edgecolor=style["edgecolor"],
            linewidth=2.0,
            hatch=style["hatch"],
            label=style["label"],
            zorder=3,
        )

    # y轴设置
    ax.set_ylabel('F1 Score', fontsize=22)
    ax.set_ylim(0, 1.0)

    # x轴：下标改为网络名称；不需要标题
    ax.set_xticks(x)
    ax.set_xticklabels(networks, fontsize=20, fontweight='bold')
    ax.tick_params(axis='x', which='major', length=0, pad=10)

    # 让左右坐标轴与最边上的柱子留出空白
    ax.set_xlim(-0.6, (n_networks - 1) + 0.6)

    # y轴刻度风格（贴近论文图：向内、上/右也有刻度）
    ax.tick_params(axis='y', which='major', length=10, width=1.5, direction='in', right=True, labelsize=18)
    ax.tick_params(axis='x', which='major', direction='in', top=True)
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.tick_params(axis='y', which='minor', length=5, width=1.2, direction='in', right=True)

    # 图例：上方“中间偏左”的位置（半透明底色，避免挡住柱子）
    legend = ax.legend(
        loc='upper left',
        fontsize=16,
        frameon=True,
        fancybox=False,
        framealpha=0.55,
        edgecolor="#777777",
        ncol=1,
        borderpad=0.6,
        handlelength=1.6,
        handletextpad=0.6,
        labelspacing=0.45,
        borderaxespad=0.2,
        bbox_to_anchor=(0.30, 0.995),
    )
    legend.get_frame().set_linewidth(1.0)
    legend.get_frame().set_facecolor("#f2f2f2")

    # 网格
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    pdf_path = save_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, dpi=400, bbox_inches='tight')
    print(f"图片已保存为 {save_path} 和 {pdf_path}")
    plt.show()

# ========== 主函数：手动输入结果或运行实验 ==========
if __name__ == "__main__":
    # 方式1: 手动输入结果数据（推荐，用于快速绘图）
    manual_results = {
        'Thiers12': {
            'full_hrck': 0.94,      # 完整HRCK框架
            'no_community': 0.30,   # 无社区划分
            'no_key_nodes': 0.76    # 无关键节点
        },
        'LyonSchool': {
            'full_hrck': 0.96,
            'no_community': 0.32,
            'no_key_nodes': 0.81
        }
    }

    # 绘制结果
    plot_ablation_results(manual_results, save_path='ablation_study_results.png')
    
    # 方式2: 运行实际实验（取消注释以使用）

    # # 实验配置
    # networks_config = {
    #     'Thiers12': {
    #         'ts_path': './RealNet_data/generated_time_series.csv',
    #         'graph_path': './RealNet_data/Paired_connection_adjusted_minus1.txt',
    #         'true_edges_path': './RealNet_data/High_connection.txt'
    #     },
    #     'Hypertext2009': {
    #         'ts_path': './RealNet_data/generated_time_series.csv',
    #         'graph_path': './RealNet_data/Paired_connection_adjusted_minus1.txt',
    #         'true_edges_path': './RealNet_data/High_connection.txt'
    #     }
    # }
    #
    # results = {}
    # for network_name, config in networks_config.items():
    #     print(f"\n正在进行 {network_name} 的消融实验...")
    #
    #     # 配置1: 完整HRCK
    #     print("  运行配置1: 完整HRCK...")
    #     result1 = run_full_hrck(**config, k=5, top_percent=0.05, iterations=1)
    #
    #     # 配置2: 无社区划分
    #     print("  运行配置2: 无社区划分...")
    #     result2 = run_no_community(**config, top_percent=0.05, iterations=1)
    #
    #     # 配置3: 无关键节点
    #     print("  运行配置3: 无关键节点...")
    #     result3 = run_no_key_nodes(**config, k=5, iterations=1)
    #
    #     results[network_name] = {
    #         'full_hrck': result1['f1'],
    #         'no_community': result2['f1'],
    #         'no_key_nodes': result3['f1']
    #     }
    #
    #     print(f"  完成! F1分数: HRCK={result1['f1']:.3f}, 无社区={result2['f1']:.3f}, 无关键节点={result3['f1']:.3f}")

    # # 绘制结果
    # plot_ablation_results(results, save_path='ablation_study_results.png')
