
[community_detection](community_detection)该文件夹包含[PCDMS.py](community_detection/PCDMS.py)文件，主要被调用用于执行社区划分
[SynNet_data](SynNet_data)文件夹下包含了合成网络的数据：
[RealNet_data](RealNet_data)文件夹下包含了真实网络的数据：
[email-Enron-nverts.txt](RealNet_data/email-Enron-nverts.txt)、[email-Enron-simplices.txt](RealNet_data/email-Enron-simplices.txt)、[email-Enron-times.txt](RealNet_data/email-Enron-times.txt)这三个文件为真实网络的预处理数据
[generated_time_series.csv](RealNet_data/generated_time_series.csv)为真实网络时间序列数据
[High_connection.txt](RealNet_data/High_connection.txt)为真实网络的高阶超边结构
[Paired_connection.txt](RealNet_data/Paired_connection.txt)为真实网络的成对结构
[Infect.py](RealNet_data/Infect.py)、[pretreatment.py](RealNet_data/pretreatment.py)、[view.py](RealNet_data/view.py)三个文件功能分别为时间序列数据生成、真实网络文件预处理、真实网络可视化

[reconstruction](reconstruction)文件夹下包含了重构函数
[High_order_Reconstruct.py](reconstruction/High_order_Reconstruct.py)为重构函数，被调用执行局部重构和全局重构并返回重构结果

[TEST](TEST)文件夹为实验部分，涉及到了论文中的所有图像的绘制，具体体现在(TEST/pic)文件夹中，而TEST/RealNet_result为实验中使用的真实网络的结果，TEST/SyntheticNet_result为实验中使用的合成网络的结果


