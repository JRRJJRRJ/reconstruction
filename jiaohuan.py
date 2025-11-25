import numpy as np
import random

data_str = """
14 19 36
57 24 55
17 23 26
3 15 65
30 67 80
10 30 14
27 30 22
2 24 57
11 46 59
20 54 18
26 17 29
63 79 81
11 41 67
3 25 68
41 61 80
17 29 42
69 71 74
2 13 47
3 49 68
14 24 84
29 42 21
75 27 8
40 30 24
8 38 84
29 78 45
67 80 30
30 37 84
31 58 38

"""

# 转为 numpy 数组
rows = [list(map(int, line.split())) for line in data_str.splitlines() if line.strip()]
data = np.array(rows, dtype=int)

# === 第一步：打乱行 ===
np.random.shuffle(data)

# === 第二步：随机选择部分行，再交换它们的列 ===
num_rows, num_cols = data.shape
selected_rows = random.sample(range(num_rows), k=random.randint(1, num_rows))

for r in selected_rows:
    perm = np.random.permutation(num_cols)  # 随机列顺序
    data[r] = data[r, perm]

# === 输出结果 ===
print(f"选中的行: {selected_rows}")
for row in data:
    print(" ".join(map(str, row)))
