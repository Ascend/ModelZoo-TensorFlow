# 概述
    mean-teacher是一种用于图像分类的半监督学习方法，能够在拥有少量有标签数据的情况下训练出分类准确率很高的网络模型。

- 论文链接: [Weight-averaged consistency targets improve semi-supervised deep learning results](https://arxiv.org/abs/1703.01780)

- 官方代码仓: [链接](https://github.com/CuriousAI/mean-teacher/)

- 精度性能比较:

|  | 论文 | GPU | Ascend |
| ------ | ------ | ------ | ------ |
| error | 12.31% | 13.50% | 14.20% |
| 性能(s/steps) |  |  |  |
# 环境
    - python 3.7.5
    - Tensorflow 1.15
    - Ascend910

# 训练
## 数据集
    使用./prepare_data.sh脚本预处理数据集
##训练超参见train.py参数列表
## 单卡训练命令
```commandline
sh ./test/train_full_1p.sh
```

# 功能测试
少量step(单epoch)运行
```commandline
sh ./test/train_performance_1p.sh
```

# 模型固化

# 部分脚本和示例代码
```text
├── README.md                                //说明文档
├── requirements.txt			//依赖
├──test		                         //训练脚本目录								 
│    ├──train_performance_1p.sh			 
│    ├──train_full_1p.sh
├──train_cifar10.py                 	     //训练脚本
```

