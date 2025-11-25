import os
import json
import time
import numpy as np
from GetF1 import run_reconstruction


# 3. 主函数（调用封装函数）
def main():
    """主函数：运行不同网络和参数的实验"""
    # 实验配置
    experiments = [
        # SIS合成网络 - 不同时间步长
        {
            "name": "SIS",
            "base_dir": "./Synthetic_Networks/SIS",
            "time_steps": [100, 2000, 4000, 6000, 8000, 10000],
            "iterations": 5,
            "ts_pattern": "T_{}/generated_time_series.csv",
            "graph_file": "Paired_connection_adjusted_minus1.txt",
            "true_edges_file": "High_connection.txt"
        },

        # Ising合成网络 - 不同时间步长
        {
            "name": "Ising",
            "base_dir": "./Synthetic_Networks/Ising",
            "time_steps": [100, 2000, 4000, 6000, 8000, 10000],
            "iterations": 5,
            "ts_pattern": "T_{}/generated_time_series.csv",
            "graph_file": "Paired_connection_adjusted_minus1.txt",
            "true_edges_file": "High_connection.txt"
        },

        # 真实网络
        {
            "name": "RealNet1",
            "ts_path": "./Real_Networks/Network1/RealNet_data/generated_time_series.csv",
            "graph_path": "./Real_Networks/Network1/RealNet_data/Paired_connection_adjusted_minus1.txt",
            "true_edges_path": "./Real_Networks/Network1/RealNet_data/High_connection.txt",
            "iterations": 5
        },
        # 其他真实网络配置...
    ]

    # 存储所有结果
    all_results = {}
    start_time = time.time()

    print("=" * 80)
    print("Starting Network Reconstruction Experiments")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # 运行实验
    for exp in experiments:
        if "time_steps" in exp:  # 处理合成网络
            print(f"\n Starting experiments for {exp['name']} network")
            net_results = {}

            for T in exp["time_steps"]:
                # 构建文件路径
                ts_path = os.path.join(exp["base_dir"], exp["ts_pattern"].format(T))
                graph_path = os.path.join(exp["base_dir"], exp["graph_file"])
                true_edges_path = os.path.join(exp["base_dir"], exp["true_edges_file"])

                # 确保文件存在
                if not all(os.path.exists(p) for p in [ts_path, graph_path, true_edges_path]):
                    print(f" Missing files for T={T}, skipping...")
                    continue

                print(f"   Running T={T} (iterations={exp['iterations']})...")

                # 调用重构函数 - 这里调用run_reconstruction
                result = run_reconstruction(
                    ts_path=ts_path,
                    graph_path=graph_path,
                    true_edges_path=true_edges_path,
                    iterations=exp["iterations"]
                )

                net_results[T] = result
                print(f"   Completed T={T}! Avg F1: {result['f1']:.4f}")

            all_results[exp["name"]] = net_results
            print(f" Finished all time steps for {exp['name']} network!")

        else:  # 处理真实网络
            print(f"\n Starting experiments for real network: {exp['name']}")

            # 确保文件存在
            if not all(os.path.exists(exp[p]) for p in ["ts_path", "graph_path", "true_edges_path"]):
                print(f" Missing files for {exp['name']}, skipping...")
                continue

            # 调用重构函数 - 这里调用run_reconstruction
            result = run_reconstruction(
                ts_path=exp["ts_path"],
                graph_path=exp["graph_path"],
                true_edges_path=exp["true_edges_path"],
                iterations=exp["iterations"]
            )

            all_results[exp["name"]] = result
            print(f" Finished {exp['name']}! Avg F1: {result['f1']:.4f}")

    # 计算总耗时
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    # 保存结果
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_file = f"reconstruction_results_{timestamp}.json"

    with open(result_file, 'w') as f:
        json.dump(all_results, f, indent=4)

    print("\n" + "=" * 80)
    print(" All experiments completed!")
    print(f"️ Total time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f" Results saved to: {result_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()