import numpy as np
import random

# 示例 merged_results
merged_results = {
    ('34', ('2', '64')): np.float64(-0.00323906),
    ('34', ('128', '2', '5', '7')): np.float64(-0.01419564),
    ('35', ('3', '64')): np.float64(0.0123456),
    ('36', ('3', '64')): np.float64(0.0123456),  # 示例中可能存在相同行内容
}

# 1. 转换键为字符串格式，并存入集合（自动去重）
lines_set = set()

for key in merged_results:
    node = key[0]
    others = key[1]
    full_list = [node] + list(others)
    line = ' '.join(full_list)
    lines_set.add(line)

# 2. 打乱顺序
all_lines = list(lines_set)
random.shuffle(all_lines)

# 3. 写入 real.txt（初步写入）
with open("real.txt", "w") as f:
    for line in all_lines:
        f.write(line + "\n")

# 4. 抽取30%的行，打乱除第一列外的其余字段顺序
num_sample = int(len(all_lines) * 0.3)
sampled_lines = random.sample(all_lines, num_sample)

# 构造替换后的行
modified_lines = []
for line in sampled_lines:
    parts = line.strip().split()
    first = parts[0]
    rest = parts[1:]
    random.shuffle(rest)
    modified_line = ' '.join([first] + rest)
    modified_lines.append(modified_line)

# 替换原来的 sampled 行
final_lines = list(set(all_lines) - set(sampled_lines)) + modified_lines
random.shuffle(final_lines)  # 最终打乱整体顺序

# 5. 覆盖写入 real.txt
with open("real.txt", "w") as f:
    for line in final_lines:
        f.write(line + "\n")
