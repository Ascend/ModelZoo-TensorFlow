# SESEMI-master

#### 简介：
SESEMI的工作属于半监督学习(SSL)的框架，在图像分类的背景下，它可以利用大量的未标记数据，在有限的标记数据中显著改进监督分类器的性能。具体来说，我们利用自监督损失项作为正则化(应用于标记数据)和SSL方法(应用于未标记数据)类似于一致性正则化。尽管基于一致性正则化的方法获得了最先进的SSL结果，但这些方法需要仔细调优许多超参数，在实践中通常不容易实现。为了追求简单和实用，我们的模型具有自监督正则化，不需要额外的超参数来调整最佳性能。

启动训练
boot_modelarts.py


#### 模型训练

1.  配置训练参数
首先在脚本run_1p.sh中，配置训练数据集--data、选择网络--network、输入训练数据量--labels参数。

2.  启动训练
运行boot_modelarts.py


#### 环境设置
Ascend: 1*Ascend 910(32GB) | ARM: 24 核 96GB
python3.7
Tensorflow 1.15.0
opencv-python
scipy1.2.0
numpy
ModelArts配置：
OBS Path:/aeaemi/sesemi-master/log/
Data Path in OBS:/aeaemi/sesemi-master/dataset/

#### 训练精度
精度验证指标为分类的错分率。
|样本数量|  论文精度 |GPU精度 | NPU精度 |
|---------|-------- -- |----------|-----------|
| 1000    |29.44±0.24|   0.2876 | 0.2983     |
| 2000    |21.53±0.18|   0.2186 | 0.2179     |


#### 数据集说明：

我们提供了CIFAR-10数据集，其他数据集需要修改其中的相对路径才可跑通，我们提供的数据如下：
URL:
https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=sD3UuWOA7+bXMQtWCv8y4+U8h1F32v1qT6N1/xkznQhDes8t6dinTzBhKrnPqNYqVDVNfbANARmzk2zF006mmER/6izbNTJLeIz0UlG9b6b0cOmeYkm/ftRvmAlE802cyHOesS57GvfBqT4goeQM4+wgNg/9x0Ccfp309tEq4PmoqxTBbeoM29Ocd3o2pMTnv4oWRYR8KKb/7D0TVhbOzBtZshz89Fmajg0ehho7uwHBW0HYyhdmry2FhGR/JsuutA2yLQ68NeBcyXLYCf0ARJidSH3T5gOHO/470zpRasnsNXE9m2T4RrLpObDOu9v5dz9cxSA/GmBtVqe6C0issrP/cLSyCgrtfCPKQgPpJo39mkBSDtQsetgD45uUMPodLz9k2G+qFnlpoZrU5YhB/Q1hfCSqlZhp3n5Fbu8tyGx6AJWZXOuiZuEeZxhMZAybD9oJcRmcoSqtZ1+QYyvRQmGw1a3+O+Fidt7CpxdjOJ9YnxgDA4s3PedC8EkXBMUFJLGsUDSd65HIH/d1K7Zb1E7Ti6v8cM8CDAqmI97VP11LupaVMasyJ16+TKpgx07cSxioUwH5HP72NwG2rVjeCJPYMYEKZ1oeNPpSMpSXoxft/E8XMPYPrW2DOJU47rklpr9kzYL+PsASc2M3jAsJgg==

提取码:
123456

*有效期至: 2022/11/08 09:28:47 GMT+08:00
#### 训练日志
\训练日志