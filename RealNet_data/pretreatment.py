# 读取 nverts 和 simplices 文件，输出到 Simplices_txt 文件中

with open('email-Enron-nverts.txt', 'r') as f_nverts, \
     open('email-Enron-simplices.txt', 'r') as f_simplices, \
     open('Simplices_txt', 'w') as f_output:

    simplices_lines = [line.strip() for line in f_simplices]
    ptr = 0

    for line in f_nverts:
        n = int(line.strip())  # 当前单纯形的节点数
        simplex = simplices_lines[ptr:ptr + n]  # 取出该单纯形的节点
        f_output.write(' '.join(simplex) + '\n')  # 写入一行
        ptr += n  # 移动指针
# 将 Simplices_txt 中的内容按连接类型划分为两个文件

with open('Simplices_txt', 'r') as f_input, \
     open('Paired_connection.txt', 'w') as f_pair, \
     open('High_connection.txt', 'w') as f_high:

    seen_pair_edges = set()  # 记录已写入的成对边
    seen_high_edges = set()  # 记录已写入的高阶边

    for line in f_input:
        nodes = line.strip().split()
        if len(nodes) == 2:
            edge_tuple = tuple(sorted(nodes))  # 排序去重
            if edge_tuple not in seen_pair_edges:
                f_pair.write(' '.join(nodes) + '\n')
                seen_pair_edges.add(edge_tuple)
        elif len(nodes) > 2:
            edge_tuple = tuple(sorted(nodes))  # 排序去重
            if edge_tuple not in seen_high_edges:
                f_high.write(' '.join(nodes) + '\n')
                seen_high_edges.add(edge_tuple)


missing_nodes = ['10', '124', '16', '31', '35']
missing_nodes = sorted(int(m) for m in missing_nodes)  # 转为整数排序


# 顺位补全操作
def adjust_node(node: str) -> str:
    val = int(node)
    offset = sum(1 for m in missing_nodes if val > m)
    return str(val - offset)

# 处理成对边
with open('Paired_connection.txt', 'r') as f_in, \
     open('Paired_connection_adjusted.txt', 'w') as f_out:

    for line in f_in:
        u, v = line.strip().split()
        u_new = adjust_node(u)
        v_new = adjust_node(v)
        f_out.write(f"{u_new} {v_new}\n")

# 处理高阶边
with open('High_connection.txt', 'r') as f_in, \
     open('High_connection_adjusted.txt', 'w') as f_out:

    for line in f_in:
        nodes = line.strip().split()
        new_nodes = [adjust_node(node) for node in nodes]
        f_out.write(' '.join(new_nodes) + '\n')

input_file = 'Paired_connection_adjusted.txt'
output_file = 'Paired_connection_adjusted_minus1.txt'

with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
    for line in fin:
        nodes = line.strip().split()
        if all(node.isdigit() for node in nodes):  # 只处理纯数字节点
            new_nodes = [str(int(node) - 1) for node in nodes]
            fout.write(' '.join(new_nodes) + '\n')
        else:
            print(f"⚠️ 非纯数字行跳过: {line.strip()}")
