# ProtoAttend: Attention-Based Prototypical Learning


## 基本信息

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Computer Vision**

**版本（Version）：1.0**

**修改时间（Modified） ：2022.5**

**大小（Size）：84KB**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Research**

**描述（Description）：基于注意力机制的原型学习**

## 概述

ProtoAttend是一种可解释的机器学习方法，该方法基于原型的少数相关示例做出决策。ProtoAttend 可以集成到各种神经网络架构中，包括预训练模型。它利用一种注意力机制，将编码表示与样本相关联，以确定原型。
在不牺牲原始模型准确性的情况下，生成的模型在三个高影响问题上优于现有技术：（1）它实现了高质量的可解释性，输出与决策最相关的样本（即基于样本的可解释性方法）； (2) 它通过量化原型标签之间的不匹配来实现最先进的置信度估计； (3) 它获得了分布不匹配检测的最新技术。所有这些都可以通过最少的额外测试时间和实际可行的训练时间计算成本来实现。

采用Fashion-MNIST 数据集，使用 ResNet 作为图像编码器模型。

要将实验修改为其他数据集和模型：
实现数据批处理和预处理功能（修改 input_data.py 和数据迭代器，如 iter_train 等）。
集成适合数据类型的编码器模型函数（修改model.py中的cnn_encoder）。
重新优化新数据集的学习超参数。

- 参考论文

  > https://arxiv.org/pdf/1902.06292.pdf

- 参考开源实现

  > https://github.com/google-research/google-research/tree/master/protoattend

## 默认配置

- 训练超参
  - Batch size: 128
  - Train step: 100000
  - init_learning_rate: 0.001
  
## 快速上手

### 环境配置

requirements.txt 内记录了所需要的第三方依赖与其对应版本，可以通过命令配置所需环境。

### 模型训练

1. 数据集


    Fashion-MNIST


2. 执行训练脚本

   - 可通过train_full_1p.sh验证精度

   ```
   sh train_full_1p.sh --data_path='./dataset' --output_path='/home/ma-user/modelarts/outputs/train_url_0/'
   ```
   
   - 或通过train_performance_1p.sh验证性能
   
   ```
      sh train_performance_1p.sh --data_path='./dataset' --output_path='/home/ma-user/modelarts/outputs/train_url_0/'
   ```


## 高级参考

### 脚本和示例代码

```
.
├── dataset 			// 数据集
├── test
│   ├── train_full_1p.sh        //单卡全量训练启动脚本
│   ├── train_performance_1p.sh //单卡训练验证性能启动脚本
├── input_data.py 	        //处理数据
├── load_data.py 	        //读取数据
├── main_protoattend.py 	//训练模型
├── model.py 	                //模型定义
├── options.py 	                //参数配置
├── utils.py 	                //工具类
├── requirements.txt 		//训练python依赖列表
└── README.md 			// 代码说明文档
```

### 脚本参数

```
--data_path              数据集路径
--output_path            模型文件保存路径
```

## 训练结果

- 精度结果对比

| 精度指标项 | 论文发布 | GPU实测 | NPU实测 |
| ---------- | -------- | :------ | ------- |
| Acc(%) | 94.47     | 93.81    | 93.81    |
