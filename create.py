import os

# 父文件夹路径（所有 T 文件夹都在这里）
base_folder = r'./TEST/SyntheticNet_result/SIS'  # 改成你的路径
os.makedirs(base_folder, exist_ok=True)

# T 和 W 的取值
T_values = [1000, 2000, 3000,4000,5000,6000,7000,8000,9000,10000,11000,12000,13000,14000,15000,16000,17000,18000,19000,20000]    # 文件夹对应的 T 值
W_values = [5,10, 20, 30, 40, 50,60,70,80,90,100]  # 每个文件夹下的文件 W 值

for T in T_values:
    folder_name = f'T={T}'
    folder_path = os.path.join(base_folder, folder_name)
    os.makedirs(folder_path, exist_ok=True)  # 文件夹不存在就创建

    for W in W_values:
        filename = f'SIS_T{T}_W{W}.txt'
        file_path = os.path.join(folder_path, filename)
        # 创建空文件
        with open(file_path, 'w') as f:
            pass

print("所有文件夹和文件已创建完成！")
