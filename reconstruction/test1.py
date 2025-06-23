import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from itertools import combinations_with_replacement


def build_candidate_terms(X, k_max=3):
    """
    构造从一阶到 k_max 阶的交互项
    X: (T, N) 数据矩阵
    返回: Theta (T, num_terms), term_descriptions (list of str)
    """
    T, N = X.shape
    Theta = []
    descriptions = []

    for k in range(1, k_max + 1):
        for comb in combinations_with_replacement(range(N), k):
            term = np.prod(X[:, comb], axis=1).reshape(-1, 1)
            Theta.append(term)
            desc = '*'.join([f'x{i}' for i in comb])
            descriptions.append(desc)

    Theta = np.hstack(Theta)
    return Theta, descriptions


def signal_lasso(Yi, Theta, lambda_vals, n_terms):
    """
    简化版信号稀疏回归：贪婪选择n_terms个最相关的项
    Yi: 目标变量 (T,)
    Theta: 候选项 (T, num_terms)
    """
    residual = Yi.copy()
    selected = []
    coef = np.zeros(Theta.shape[1])

    for _ in range(n_terms):
        correlations = np.abs(Theta.T @ residual)
        correlations[selected] = 0  # 已选的项不再考虑
        idx = np.argmax(correlations)
        selected.append(idx)

        # 最小二乘拟合
        Theta_sel = Theta[:, selected]
        model = LinearRegression().fit(Theta_sel, Yi)
        coef_sel = model.coef_

        # 更新残差
        residual = Yi - Theta_sel @ coef_sel

    # 填入全长的权重向量
    coef[selected] = coef_sel
    return coef, selected


def LVexact(X, Y, kn):
    """
    X: 状态变量时间序列, shape (T, N)
    Y: 导数估计, shape (T, N)
    kn: 每个变量最多选几个交互项
    """
    T, N = X.shape
    lambda_vals = np.logspace(-6, 1, 10)
    w = []
    mse_te = []
    mse_val = []
    mse_tr = []
    terms_selected = []
    selected_indices = []

    Theta_full, descriptions = build_candidate_terms(X, k_max=3)

    for i in range(N):
        Yi = Y[:, i]

        X_trainval, X_test, y_trainval, y_test = train_test_split(
            Theta_full, Yi, test_size=0.2, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=0.25, random_state=42
        )

        coef, selected = signal_lasso(y_train, X_train, lambda_vals, kn)
        Theta_sel = X_train[:, selected]
        model = LinearRegression().fit(Theta_sel, y_train)
        coef_sel = model.coef_

        # 填入完整长度的权重
        coef_full = np.zeros(Theta_full.shape[1])
        for idx, val in zip(selected, coef_sel):
            coef_full[idx] = val

        # 存储
        w.append(coef_full)
        selected_indices.append(selected)
        terms_selected.append(len(selected))

        mse_tr.append(np.mean((X_train @ coef_full - y_train)**2))
        mse_val.append(np.mean((X_val @ coef_full - y_val)**2))
        mse_te.append(np.mean((X_test @ coef_full - y_test)**2))

    return w, mse_te, mse_val, mse_tr, terms_selected, selected_indices


if __name__ == "__main__":
    # 生成假数据
    T, N = 200, 3
    t = np.linspace(0, 10, T)
    X = np.stack([
        np.sin(t),
        np.cos(t),
        np.sin(2*t)
    ], axis=1)

    # 简单近似导数
    Y = np.gradient(X, axis=0)

    # 最大项数
    kn = 5

    # 执行重构
    w, mse_te, mse_val, mse_tr, terms_selected, selected_indices = LVexact(X, Y, kn)

    # 输出结果
    print("每个变量重构误差：")
    for i in range(N):
        print(f"变量 {i}：训练 MSE={mse_tr[i]:.4f}，验证 MSE={mse_val[i]:.4f}，测试 MSE={mse_te[i]:.4f}")
        print(f"  选中项数：{terms_selected[i]}")
        print(f"  选中索引：{selected_indices[i]}")
        print()

    # 可视化选中项系数
    for i, coef in enumerate(w):
        plt.figure()
        plt.title(f"Variable {i} - selected coefficients")
        plt.plot(coef, 'o')
        plt.xlabel("Term index")
        plt.ylabel("Coefficient")
    plt.show()
