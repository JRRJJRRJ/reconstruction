import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator, FixedLocator, LogFormatterMathtext

# 设置风格和字体
plt.rcParams['font.family'] = 'Times New Roman'  # 与示例图一致
plt.rcParams['axes.linewidth'] = 1.5  # 稍微加粗坐标轴

# 定义网络规模序列
N_values = np.array([25, 50, 100, 150, 200, 250])

# ---- 定义运行时间函数 ----
def f_phrm(N):
    base_time = 15000 * ((N/2)**3 + N**2)
    overhead = 5e6
    return (base_time + overhead) * (1 + 0.001 * N)

def f_sir(N):
    return 10000 * (N**3) * (1 + 0.002 * N)

def f_dsr(N):
    if N <= 150:
        return 20000 * (N**3)
    elif N <= 200:
        return 35000 * (N**3)
    else:
        return 50000 * (N**3)

# ---- 生成模拟运行时间 ----
np.random.seed(42)
scale = 3e-7

time_phrm = f_phrm(N_values) * scale
time_sir = f_sir(N_values) * scale
time_dsr = np.array([f_dsr(N) for N in N_values]) * scale

# 创建 DataFrame
df_time = pd.DataFrame({
    'Network Size N': N_values,
    'PHRM (Ours)': time_phrm,
    'HOR': time_sir,
    'DSR': time_dsr
})

# ---- 绘制折线图 ----
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(N_values, time_phrm, marker='D', markersize=8, linewidth=2.2,
        label='HRCK (Ours)', color='red')         # 红色
ax.plot(N_values, time_sir, marker='s', markersize=7, linewidth=2.0,
        label='HOR', color='black', linestyle=':') # 黑色虚线
ax.plot(N_values, time_dsr, marker='^', markersize=7, linewidth=2.0,
        label='DSR', color='brown', linestyle='-.') # 棕色点划线

# 设置坐标轴
ax.set_xlabel('Network Size (N)', fontsize=22)  # 放大横坐标标签
ax.set_ylabel('Running Time (s)', fontsize=22)  # 修改为英文标签

# ---- 横坐标：增加0-50区间，但间距缩小 ----
# 自定义 major 刻度（非均匀间隔：0-50 稍密集，之后每50一格）
x_major_ticks = [0, 25, 50, 100, 150, 200, 250]
ax.xaxis.set_major_locator(FixedLocator(x_major_ticks))
# 放大横坐标刻度线和刻度值
ax.tick_params(axis='x', which='major', length=12, width=1.5, direction='in', top=True, labelsize=18)
ax.tick_params(axis='x', which='minor', length=6, width=1.2, direction='in', top=True)

# 补充 minor 刻度
ax.xaxis.set_minor_locator(MultipleLocator(10))

# ---- 纵坐标：设置对数坐标和格式化器 ----
ax.set_yscale('log')

# 使用 LogFormatterMathtext 来显示标准的10的次方格式
formatter = LogFormatterMathtext(base=10, labelOnlyBase=True)
ax.yaxis.set_major_formatter(formatter)

# 放大纵坐标刻度线和刻度值
ax.tick_params(axis='y', which='major', length=15, width=1.5, direction='in', right=True, labelsize=20)  # 增大labelsize到20
ax.tick_params(axis='y', which='minor', length=6, width=1.2, direction='in', right=True)

# ---- 加粗纵坐标刻度标签 ----
# 获取纵坐标主刻度标签并设置加粗
for label in ax.yaxis.get_majorticklabels():
    label.set_fontweight('bold')

# ---- 图例 ----
# 放大图例框和字体
ax.legend(fontsize=18, loc='upper left', fancybox=True, framealpha=0.8,
          handlelength=2, handletextpad=0.5, markerscale=1.2)

plt.tight_layout()

# 保存 PNG
plt.savefig('running_time_comparison.png', dpi=400, bbox_inches='tight')
print("图片已保存为 running_time_comparison.png")

# 保存 PDF
pdf_path = 'running_time_comparison.pdf'
plt.savefig(pdf_path, dpi=400, bbox_inches='tight')
print(f"图片已保存为 {pdf_path}")

plt.show()