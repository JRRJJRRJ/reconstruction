import os
import re

# 文件所在目录（绝对路径）
folder = r"J:\pycharm\MyProject\HighOrder\TEST\pic\flip_F1\flip_data\flip_results"

# 获取目录下所有txt文件
files = [f for f in os.listdir(folder) if f.endswith(".txt")]

# 正则匹配文件名末尾 -数字.txt
pattern = re.compile(r"(.*-)(\d+)(\.txt)$")

# 先解析出数字，按数字降序排列，保证先改大的
file_tuples = []
for f in files:
    match = pattern.match(f)
    if match:
        number = int(match.group(2))
        file_tuples.append((number, f))

# 按数字降序排序
file_tuples.sort(reverse=True)

for number, f in file_tuples:
    match = pattern.match(f)
    if match:
        prefix = match.group(1)
        suffix = match.group(3)

        new_number = number + 5
        new_name = f"{prefix}{new_number}{suffix}"

        old_path = os.path.join(folder, f)
        new_path = os.path.join(folder, new_name)

        if os.path.exists(new_path):
            print(f"❌ 目标文件已存在，跳过: {new_name}")
            continue

        os.rename(old_path, new_path)
        print(f"✅ {f} -> {new_name}")
