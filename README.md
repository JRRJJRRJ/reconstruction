[main1.py](main1.py)main1函数是文件入口，调用即可进行网络重构

[data1](data1)data1文件夹下是已经整理的合成网络的所有数据，其生成过程间data文件夹
[graph_edges.txt](data1/graph_edges.txt)该文件为合成网络中的真实成对结构
[hyperedges_k3.txt](data1/hyperedges_k3.txt)为合成网络中的真实超边结构
[sis_time_series.csv](data1/sis_time_series.csv)为生成的时间序列数据

[community_detection](community_detection)该文件夹包含[PCDMS.py](community_detection/PCDMS.py)文件，主要被调用用于执行社区划分

[RealNet_data](RealNet_data)文件夹下包含了真实网络的一些数据：
[email-Enron-nverts.txt](RealNet_data/email-Enron-nverts.txt)、[email-Enron-simplices.txt](RealNet_data/email-Enron-simplices.txt)、[email-Enron-times.txt](RealNet_data/email-Enron-times.txt)这三个文件为真实网络的预处理数据
[generated_time_series.csv](RealNet_data/generated_time_series.csv)为真实网络时间序列数据
[High_connection.txt](RealNet_data/High_connection.txt)为真实网络的高阶超边结构
[Paired_connection.txt](RealNet_data/Paired_connection.txt)为真实网络的成对结构
[Infect.py](RealNet_data/Infect.py)、[pretreatment.py](RealNet_data/pretreatment.py)、[view.py](RealNet_data/view.py)三个文件功能分别为时间序列数据生成、真实网络文件预处理、真实网络可视化

[reconstruction](reconstruction)文件夹下包含了重构函数
[High_order_Reconstruct.py](reconstruction/High_order_Reconstruct.py)为重构函数，被调用执行局部重构和全局重构并返回重构结果

