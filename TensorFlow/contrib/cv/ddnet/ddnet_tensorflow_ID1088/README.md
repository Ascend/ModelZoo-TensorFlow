# 基本信息
## 发布者（Publisher）：Huawei
## 应用领域（Application Domain）：cv
## 修改时间（Modified） ：2021.12.28
## 框架（Framework）：TensorFlow_1.15.0
## 模型格式（Model Format）：ckpt
## 精度（Precision）：double
## 处理器（Processor）：昇腾910
## 描述（Description）：基于TensorFlow框架的DD-NET网络训练代码

# 概述
DD-NET
* 参考论文
https://arxiv.org/pdf/1907.09658.pdf
* 参考实现
https://github.com/fandulu/DD-Net

# 默认配置
* 数据集
    * data/JHMDB
* 训练超参
    * Train epoch：1200

# 支持特性
| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 否    |
| 混合精度  | 否    |
| 数据并行  | 否    |

# 混合精度训练
不支持

# 开启混合精度
不支持

# 训练环境准备
1. requirements.txt
2. 镜像：ascend-share/5.0.3.alpha005_tensorflow-ascend910-cp37-euleros2.8-aarch64-training:1.15.0-2.0.12_1116


# 快速上手
# 数据集准备
在训练脚本中指定数据集路径，即可正常使用。

# 模型训练
python ddnet.py

# 迁移学习指导
无

# 脚本和示例代码
ddnet.py

# 精度和性能对比
|     | Accuracy | 单步性能 |
|-----|----------|----------|
| GPU | 0.8352   | 200us    |
| NPU | 0.8409   | 285us    |
