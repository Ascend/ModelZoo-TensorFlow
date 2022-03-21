# ADA-Net

## 概述

ADA-Nets是一种半监督学习方法。由于标记样本的数量有限，在半监督学习中存在一个必要的抽样偏差，这往往导致在标记数据和未标记数据之间存在相当大的经验分布不匹配。ADA-Net采用了一种对抗性训练策略来最小化标记数据和非标记数据之间的分布距离。

- 参考论文 [Semi-Supervised Learning by Augmented Distribution Alignment](https://arxiv.org/abs/1905.08171)  Qin Wang, Wen Li, Luc Van Gool (ICCV 2019 Oral)
- 参考项目https://github.com/qinenergy/adanet

## 默认配置

- 数据图片resize为32*32

- 训练超参：
  - batch_size：128
  - ul_batch_size：128
  - eval_batch_size：128
  - num_epochs：120
  - epoch_decay_start：80
  - learning_rate：0.01
  - mom1：0.9
  - mom2:   0.5
  
- 训练数据集：

  SVHN数据集

## 支持特性

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 否       |
| 混合精度   | 是       |
| 数据并行   | 否       |

### 混合精度训练

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

### 开启混合精度

脚本默认不开启混合精度。开启混合精度参考如下：

```python
config = tf.ConfigProto()
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True
custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
```



## 文件目录

cnn.py：训练网络架构。

dataset_utils.py：处理数据的通用函数文件。

flip_gradient.py：网络定义文件。

layers.py：网络定义文件。

svhn.py：处理cifar数据库，生成tfrecord数据。为训练提供dataset。

test_svhn.py：测试svhn数据集上的效果。

train_svhn.py：svhn数据集训练文件。

## 训练环境准备

硬件环境：

Ascend: 1*Ascend 910(32GB) | ARM: 24 核 96GB

1. 硬件环境准备请参见各硬件产品文档"[驱动和固件安装升级指南](https://support.huawei.com/enterprise/zh/category/ai-computing-platform-pid-1557196528909)"。需要在硬件设备上安装与CANN版本配套的固件与驱动。

2. 宿主机上需要安装Docker并登录[Ascend Hub中心](https://ascendhub.huawei.com/#/detail?name=ascend-tensorflow-arm)获取镜像。

运行环境：

- Python 3.0及以上
- Numpy
- OpenCV
- scipy
- TensorFlow 1.15

## 快速上手

### 数据集准备
从The Street View House Numbers(SVHN) Dataset官网上下载SVHN数据集，包括train 32×32.mat和test 32×32.mat。将两个mat文件放入./dataset/svhn文件夹（可以通过参数data_dir自定义）

运行以下命令通过svhn数据集生成对应tfrecord文件。
```bash
python svhn.py --data_dir=./dataset/svhn/ --dataset_seed=1
```


### 模型训练

参考下方的训练过程来训练模型

预训练模型下载链接：链接：https://pan.baidu.com/s/19gS1tjY1RIG2bQPMFg3t_Q 
提取码：s9we


## 高级参考

###  脚本和示例代码

```
|-- LICENSE
|-- README.md
|-- cnn.py
|-- dataset_utils.py
|-- flip_gradient.py
|-- layers.py
|-- modelzoo_level.txt
|-- requirements.txt
|-- svhn.py
|-- test_svhn.py
`-- train_svhn.py
```

### 脚本参数

```
    --logdir              		the path of store models 
    --seed       				initial random seed
    --validation                
    --batch_size                the number of examples in a batch
    --ul_batch_size             the number of unlabeled examples in a batch
    --eval_batch_size           the number of eval examples in a batch
    --num_epochs                the number of epochs for training
    --epoch_decay_start       	epoch of starting learning rate decay
    --num_iter_per_epoch        the number of updates per epoch      
    --learning_rate           	initial leanring rate
    --mom1            			initial momentum rate
    --mom2            			momentum rate after epoch_decay_start
    --data_dir             		the path of data
    --num_labeled_examples		The number of labeled examples
    --num_valid_examples 		The number of validation examples
    --dataset_seed 				dataset seed
```

### 训练过程

```bash
# 根据数据集路径修改data_dir的值
python train_svhn.py  --data_dir=./dataset/svhn/ --log_dir=./log/svhn/ --num_epochs=2000 --epoch_decay_start=1500  --dataset_seed=1
```

### 验证过程

```bash
# 根据数据集路径修改data_dir的值和模型存储路径log_dir
python test_svhn.py --data_dir=./dataset/svhn/ --log_dir=<path_to_model_dirr> --dataset_seed=1
```

## 训练精度对比

|      | GPU   | NPU   | 论文  |
| ---- | ----- | ----- | ----- |
| ACC  | 95.22 | 92.94 | 95.38 |


## 训练性能对比

| GPU NVIDIA V100   | NPU           |
| ------------- | ------------- |
| 13.93 s/epoch | 26.54 s/epoch |

