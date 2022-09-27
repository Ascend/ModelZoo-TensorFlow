-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [快速上手](#快速上手.md)
-   [训练结果](#训练结果.md)
-   [高级参考](#高级参考.md)

<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）： Huawei**

**应用领域（Application Domain）： Computer Vision** 

**版本（Version）：1.0**

**修改时间（Modified） ：2022.07.24**

**大小（Size）：126kb**

**框架（Framework）： TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）： Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）： Research**

**描述（Description）：基于TensorFlow框架的Polygen网络训练代码** 

<h2 id="概述.md">概述</h2>

PolyGen是三维网格的生成模型，可顺序输出网格顶点和面。PolyGen由两部分组成：一个是顶点模型，它无条件地对网格顶点进行建模，另一个是面模型，它对以输入顶点为条件的网格面进行建模。顶点模型使用一个masked Transformer解码器来表示顶点序列上的分布。对于面模型，PolyGen将Transformer与pointer network相结合，以表示可变长度顶点序列上的分布。

- 参考论文：

  [[2002.10880\] PolyGen: An Autoregressive Generative Model of 3D Meshes (arxiv.org)](https://arxiv.org/abs/2002.10880)

- 参考实现：
  [https://github.com/deepmind/deepmind-research/tree/master/polygen](https://github.com/deepmind/deepmind-research/tree/master/polygen)

- 适配昇腾 AI 处理器的实现：
  https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/Polygen_ID2061_for_TensorFlow


- 通过Git获取对应commit\_id的代码方法如下：
  
    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    ```

## 默认配置


- 训练超参
    * Batch size：1
    * Training step：5000
    * Learning rate： 5e-4
    
- 编码器与解码器结构超参

    - dropout rate：0
    - number of layers： 3
    - hidden layer ： 128
    - fc layer ： 512

## 支持特性
| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 否    |
| 混合精度  | 是    |
| 数据并行  | 否    |

## 混合精度训练
昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度
``` python
config_proto = tf.ConfigProto()
custom_op = config_proto.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = 'NpuOptimizer'
custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
config = npu_config_proto(config_proto=config_proto)
with tf.Session(config=config) as sess:
```

<h2 id="训练环境准备.md">训练环境准备</h2>
1. 准备裸机环境

    Atlas服务器包含昇腾AI处理器，可用于模型训练，训练前请参考《[CANN软件安装指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/51RC2alpha007/softwareinstall/instg/atlasdeploy_03_0002.html)》进行环境搭建。除OS依赖、固件、驱动、CANN等软件包之外，用户还需参考文档在裸机环境中安装TensorFlow框架相关模块。

4. Requirements
```
python==3.7.5
dm-sonnet==1.36
numpy==1.18.0
tensor2tensor==1.14
tensorboard==1.15.0
tensorflow==1.15.0
```




<h2 id="快速上手.md">快速上手</h2>

## 数据集准备
在训练脚本中已指定数据集路径，可正常使用。

## 模型训练
- 选择合适的下载方式下载源码与数据集，并上传到裸机环境。

- 启动训练之前，首先要配置程序运行相关环境变量。

  环境变量配置信息参见：

     [Ascend 910训练平台环境变量设置 - Wiki - Gitee.com](https://gitee.com/ascend/modelzoo/wikis/其他案例/Ascend 910训练平台环境变量设置)


- 单卡训练 

  1. 配置训练参数。

     首先在脚本test/train_full_1p.sh中，配置training steps，precision_mode等参数，请用户根据实际路径配置data_path，或者在启动训练的命令行中以参数形式下发。

     ```
      training_steps=5000
      data_path="../meshes"
     ```

  2. 启动训练。

     启动单卡训练 （脚本为 AvatarGAN_ID1305_for_TensorFlow/train_full_1p.sh） 

     ```
     bash train_full_1p.sh --data_path=../meshes --training-steps=5000 --precision_mode=mix
     ```


<h2 id="训练结果.md">训练结果</h2>

## 结果比对

精度结果对比

| Platform            | Loss(vertices) | Loss(faces) |
| ------------------- | -------------- | ----------- |
| GPU                 | 0.01837750  | 0.01971974            |
| NPU（不加混合精度） | 0.01822554     | 0.01276514  |
| NPU（加混合精度）   | 0.07918512     | 0.04801641  |

性能结果比对  

Platform | second per step | TimeToTrain(5000 steps) 
--- | --- | --- 
GPU | 1.4925 seconds | 124min 22s
NPU（不加混合精度） | 2.2745 seconds | 189min 32s 
NPU（加混合精度） | 0.1096 seconds | 9min 48s 



<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码

```bash
Polygen
└─
  ├─meshes                数据集
  |  ├─cone.obj
  |  ├─cube.obj
  |  ├─cylinder.obj
  |  ├─icosphere.obj
  ├─test                  测试脚本
  |  ├─model_test.sh
  |  ├─train_full_1p.sh
  ├─data_utils.py
  ├─model_test.py         模型测试代码
  ├─modelzoo_level.txt   
  ├─module.py             自定义网络模型
  ├─README.md      
  ├─train.py              执行训练主函数
  ├─requirements.txt      依赖需求
```

## 脚本参数

| 参数 | 默认值 | 说明|
|---| ---|---|
|--training_steps|5000|number of training steps|
|--precision_mode|mix|开启混合精度|
|--data_path|"../meshes"|设置数据集路径|

## 训练过程

1.  通过“模型训练”中的训练指令启动单卡训练。

2.  参考脚本的模型存储路径为./output。





