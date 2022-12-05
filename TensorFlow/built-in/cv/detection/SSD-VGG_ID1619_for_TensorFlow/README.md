- [基本信息](#基本信息.md)
- [概述](#概述.md)
- [训练环境准备](#训练环境准备.md)
- [快速上手](#快速上手.md)
- [迁移学习指导](#迁移学习指导.md)
- [高级参考](#高级参考.md)
## 基本信息

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Object Detection** 

**版本（Version）：1.1**

**修改时间（Modified） ：2021.09.14**

**大小（Size）：1193MB**

**框架（Framework）：TensorFlow_1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的SSD-VGG图像检测网络训练代码** 

## 概述

SSD-VGG是采用单个深度神经网络模型实现目标检测和识别的方法。模型主要特点：多尺度特征映射。将卷积特征层添加到截取的基础网络的末端。大尺度的特征图有较多的信息，可以用来检测小物体。而小尺度的特征图用来检测较大的物体。允许在多个尺度上对检测结果进行预测。采用卷积层作为预测器。代替了全连接层，直接采用卷积对不同的特征图进行提取检测结果。

- 参考论文：

  [https://arxiv.org/abs/1810.04805](https://gitee.com/link?target=https%3A%2F%2Farxiv.org%2Fabs%2F1810.04805)

- 参考实现：

  NA

- 适配昇腾 AI 处理器的实现：
  
  https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/built-in/cv/detection/SSD-VGG_ID1619_for_TensorFlow

- 通过Git获取对应commit\_id的代码方法如下：
  
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    

#### 默认配置<a name="section91661242121611"></a>

-   网络结构
    - 24-layer, 1024-hidden, 16-heads, 340M parameters
-   训练超参（单卡）：
    - Batch size: 8
    - Momentum: 0.9
    - LR scheduler: cosine
    - Learning rate(LR): 0.00075;0.0001;0.0001
    - Optimizer: MomentumOptimizer
    - Weight decay: 0.0005
    - Train epoch: 200


#### 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 是       |
| 混合精度   | 是       |
| 数据并行   | 是       |


#### 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

#### 开启混合精度<a name="section20779114113713"></a>

拉起脚本中，传入--precision_mode='allow_mix_precision'

```
 ./train_full_1p.sh --help
parameter explain:
    --precision_mode         precision mode(allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision)
    --over_dump                  if or not over detection, default is False
    --data_dump_flag         data dump flag, default is False
    --data_dump_step             data dump step, default is 10
    --profiling                  if or not profiling for performance debug, default is False
    --data_path                  source data of training
    -h/--help                    show help message
```

相关代码示例:

```
parser.add_argument('--precision_mode', type=str, default='allow_fp32_to_fp16',
                        help='precision mode, default is allow_fp32_to_fp16')
```

## 训练环境准备

-  硬件环境和运行环境准备请参见《[CANN软件安装指南](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=installation-update)》
-  运行以下命令安装依赖。
```
pip3 install requirements.txt
```
说明：依赖配置文件requirements.txt文件位于模型的根目录

## 快速上手

#### 数据集准备<a name="section361114841316"></a>
- 用户自行准备好数据集，模型训练使用Pascal VOC数据集，数据集请用户自行获取

   ```
  bash download-data.sh
  ```
- 数据集训练前需要做预处理操作
  ```
  ./process_dataset.py
  ```
- 数据集处理后，放入模型目录下，在训练脚本中指定数据集路径，可正常使用。



#### 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练。

    - 启动训练之前，首先要配置程序运行相关环境变量。

      环境变量配置信息参见：

      [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)

    - 单卡训练

      ```
      cd test;
      bash train_full_1p.sh --data_path=./data/
      ```
    
       
    


## 高级参考

#### 脚本和示例代码

```
    ├── test
    │   ├── train_full_1p.sh                            // 执行全量训练脚本
    │   └── train_performance_1p.sh
    ├── LICENSE
    ├── README.md
    ├── average_precision.py
    ├── data_queue.py
    ├── detect.py
    ├── download-data.sh
    ├── export_model.py
    ├── infer.py
    ├── modelzoo_level.txt
    ├── pascal_summary.py
    ├── process_dataset.py
    ├── run_1p.sh
    ├── source_pascal_voc.py
    ├── ssdutils.py
    ├── ssdvgg.py
    ├── train.py
    ├── training_data.py
    ├── transforms.py
    └── utils.py
```



#### 脚本参数

```
--data_path                                    data path，default is the path of train.py
--name                                         project name，default='ckpt'    
--epochs                                       train epochs，default=200
--batch-size                                   batch size，default=8
--checkpoint-interval                          checkpoint interval，default=200
--lr-values                                    learning rate values，default='0.00075;0.0001;0.00001'
--lr-boundaries                                learning rate change boundaries (in batches)，default='320000;400000'
--momentum                                     momentum for the optimizer，default=0.9  
--weight-decay                                 L2 normalization factor，default=0.0005
--continue-training                            continue training from the latest checkpoint，default='False'
--num-workers                                  number of parallel generators，default=mp.cpu_count()
--precision_mode                               precision mode, default is allow_fp32_to_fp16
--over_dump                                    over flow dump, True or False, default is False
--data_dump                                    data dump, True or False, default is False
--dump_path                                    dump path，default='/home/HwHiAiUser/'
```




#### 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡，8卡网络训练。模型存储路径为${cur_path}/output/$ASCEND_DEVICE_ID，包括训练的log以及checkpoints文件。以8卡训练为例，loss信息在文件${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。