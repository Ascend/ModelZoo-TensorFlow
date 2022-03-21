# SENet-Tensorflow


 **概述 ** 
 
迁移se_resnet到ascend910平台

代码来源：

在[lovejing0306/TF2CIFAR10](https://arxiv.org/abs/1709.01507)网络基础上进行se_block的插入和tf1.x适配

相关论文：

[Squeeze Excitation Networks](https://arxiv.org/abs/1709.01507)

支持特性：

混合精度训练

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

开启混合精度

脚本已默认开启混合精度，设置precision_mode参数的脚本参考如下。

custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")

 **准备工作** 
 
训练环境准备
1. Tensorflow 1.15
2. Ascend910
3. tf_slim=1.1.0（pip install）
4. tflearn=0.5.0（pip install）
5. cuda10.0


**代码及路径解释** 

```
SEResNet_hw_dw8023_for_Tensorflow
└─ 
  ├─README.md
  ├─run_gpu.py gpu训练启动
  ├─eval.py npu测试模型精度
  ├─modelart_senet.py modelarts训练模式启动文件
  ├─help_modelarts.py modelarts训练辅助工具
  ├─run_npu.py npu训练启动
  ├─seresnetv2.py 实现SE_ResNet110网络架构
  ├─model 用于存放预训练模型 #obs://effcientdet/output/
  	├─se_resnet_110.ckpt
  	└─...
  ├─cifar10.py 读取并制作cifar10数据集
  ├─eval.sh npu测试启动脚本
  ├─run_gpu.sh gpu训练启动脚本
  ├─run_npu.sh npu训练启动脚本
```


 **训练** 
 

1、数据集准备：
请用户自行准备好数据集，下载cifar10数据集-python版本压缩包，并上传到训练环境上，解压缩后格式如下

```
SEResNet_hw_dw8023_for_Tensorflow
└─ 
  ├─README.md
  ├─...其他代码文件
  ├─cifar-10-batches-py
  	├─test_batch
    ├─data_batch_1
    ├─test_batch_2
    ├─test_batch_3
    ├─test_batch_4
    ├─test_batch_5
    ├─batches.meta
  	└─readme.html

```
**2、模型训练**

**配置训练参数**

weight_decay = 0.0001             权重衰减

momentum = 0.9                    动量

init_learning_rate = 0.01         初始学习率

batch_size = 128                  每个NPU的batch size

iteration = 391                   每个epoch的迭代数

test_iteration = 10               测试迭代数

total_epochs = 160                训练epoch次数 

data_dir = 'cifar-10-batches-py'  数据集位置

num_class = 10                    数据集实际类别数

**npu训练启动脚本（单卡）**
        bash run_npu.sh 
**gpu训练启动脚本**
        bash run_gpu.sh 


预训练模型暂未使用

**精度与性能**

训练耗费约8小时，将精度结果与gpu复现结果进行比较（基于cifar10数据集的top1-error）

|                          |  论文   |  gpu  |  npu   |
|--------------------------|--------|--------|--------|
| Classification error (%) |  5.21  |  5.21  |  5.45  |

精度总结：npu的top-1 error比gpu的top-1 error稍高0.2%，即精度有1%以内的微小落后。

性能上以V100与Ascend910作比较
|                          |  论文  |     V100     |     Ascend910   |
|--------------------------|--------|--------------|-----------------|
|          性能数据         |  未给出|  141ms/step  |     91ms/step   |

性能总结：Ascend910较V100的性能高出不少。
