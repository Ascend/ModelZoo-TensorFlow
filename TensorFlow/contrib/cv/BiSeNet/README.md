- [基本信息](#基本信息.md)
- [概述](#概述.md)
- [训练环境准备](#训练环境准备.md)
- [快速上手](#快速上手.md)
- [迁移学习指导](#迁移学习指导.md)
- [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Semantic Segmentation**

**版本（Version）：1.1**

**修改时间（Modified） ：2022.12.13**

**大小（Size）：256KB**

**框架（Framework）：TensorFlow_1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的BiSeNet训练代码**

<h2 id="概述.md">概述</h2>

## 简述<a name="section194554031510"></a>

 BiSeNet是一种新的双向分割网络的Tensorflow 实现。用于实时性语义分割

- 参考论文：

  [BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation](https://arxiv.org/abs/1808.00897)

- 参考实现：

  https://github.com/pdoublerainbow/bisenet-tensorflow

- 适配昇腾 AI 处理器的实现：
  
  https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/BiSeNet

- 通过Git获取对应commit\_id的代码方法如下：
  
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    

## 默认配置<a name="section91661242121611"></a>

-   训练超参（单卡）：

    - BATCH_SIZE = 8
    - LEARNING_RATE = 1.e-6
    - MOMENTUM = 0.05
    - RANDOM_SEED = 123
    - WEIGHT_DECAY = 0.0005
    - MAX_EPOCH = 2000
    

## 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 混合精度   | 是       |


## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

脚本默认开启混合精度，代码如下：

```
    config = tf.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"

    custom_op.parameter_map["use_off_line"].b = True
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")

    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭remap
    config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF

```

<h2 id="训练环境准备.md">训练环境准备</h2>

-  硬件环境和运行环境准备请参见《[CANN软件安装指南](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=installation-update)》
-  运行以下命令安装依赖。
```
pip3 install requirements.txt
```
说明：依赖配置文件requirements.txt文件位于模型的根目录

<h2 id="快速上手.md">快速上手</h2>

## 数据集准备<a name="section361114841316"></a>

1、模型训练使用CamVid数据集，数据集请用户自行获取（下载链接http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/）

2、将数据集分为421张train、112张val、168张test

3、数据集处理后，放入模型目录下，在训练脚本中指定数据集路径，可正常使用

数据集目录示例
```
├── CamVid
│   ├── train
│   │   ├── xxx.png
│   │   ├── xxx.png
│   ├── train_labels
│   │   ├── xxx.png
│   │   ├── xxx.png
│   ├── val
│   │   ├── xxx.png
│   │   ├── xxx.png
│   ├── val_labels
│   │   ├── xxx.png
│   │   ├── xxx.png
│   ├── test
│   │   ├── xxx.png
│   │   ├── xxx.png
│   ├── test_labels
│   │   ├── xxx.png
│   │   ├── xxx.png
│   ├── class_dict.csv
```

4、BiSeNet训练的模型及数据集可以参考"简述 -> 参考实现"


## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练。

    - 启动训练之前，首先要配置程序运行相关环境变量。

      环境变量配置信息参见：

      [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)

    - 单卡训练

          1. 配置训练参数
        
             首先在脚本train_full_1p.sh中，配置训练数据集路径，请用户根据实际路径配置data_path，output_path，示例如下所示：
        
             ```
             # 路径参数初始化
                --data_path=${data_path}
                --output_path=${output_path} 
             ```
        
          2. 启动训练（脚本为./test/train_full_1p.sh） 
        
             ```
             bash train_full_1p.sh --data_path
             ```
           


<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码

```
├── builders                      // 前端预训练权重获取
├── configuration.py              // 模型超参数配置
├── Dataset                       // 模型数据集处理
├── frontends                     // 模型前端部分代码
├── LICENSE
├── Logs                          // 权重文件默认生成目录
├── models                        // 模型整体代码
├── README.md
├── test
│   ├── train_full_1p.sh          // 训练性能入口
│   └── train_performance_1p.sh   // 训练精度入口，包含准确率评估
├── test_npu.py                   // 测试启动文件
├── train_npu.py                  // 训练启动文件
└── utils                         // 调用模块
```

## 脚本参数

```
--data_path
--output_path
--train_epochs
--batch_size
```

## 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡训练。模型存储路径为${cur_path}/output/$ASCEND_DEVICE_ID，包括训练的log以及checkpoints文件。以单卡训练为例，loss信息在文件${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。

## NPU/GPU 网络测试精度
|             | NPU          | GPU          | 
| ----------- | ------------ | ------------ | 
| mean IOU    | 0.48         | 0.48         |
```
测试时使用测试集，运行test_npu.py脚本
```

## NPU/GPU 网络训练性能 

|             | NPU          | GPU          | 
| ----------- | ------------ | ------------ | 
| step time   | 0.75s        | 1.50s        |
```
其中GPU为v100
```
## 综合评价
NPU上训练后的精度与GPU基本一致，但是达不到论文上的结果。
NPU在训练性能上高于GPU。