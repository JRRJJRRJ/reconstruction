import os

base_dir = "TEST/RealNet_result/InVS15"

for T in range(1000, 20001, 1000):  # T=1000,2000,...,20000
    folder = os.path.join(base_dir, f"T={T}")
    if not os.path.exists(folder):
        continue

    for filename in os.listdir(folder):
        if filename.startswith("Thiers12_T"):
            new_name = filename.replace("Thiers12_T", "InVS15_T")
            old_path = os.path.join(folder, filename)
            new_path = os.path.join(folder, new_name)
            os.rename(old_path, new_path)
            print(f"重命名: {old_path} → {new_path}")
