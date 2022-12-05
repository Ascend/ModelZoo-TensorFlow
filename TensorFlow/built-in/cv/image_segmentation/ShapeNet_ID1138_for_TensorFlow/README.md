- [基本信息](#基本信息.md)
- [概述](#概述.md)
- [训练环境准备](#训练环境准备.md)
- [快速上手](#快速上手.md)
- [迁移学习指导](#迁移学习指导.md)
- [高级参考](#高级参考.md)

## 基本信息

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Instance Segmentation**

**版本（Version）：1.1**

**修改时间（Modified） ：2022.04.02**

**大小（Size）**_**：132K**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的ShapeNet网络训练代码**

## 概述

与通过定制的卷积算子捕获3D点云中的局部模式的文献相反，本文研究了如何有效地将此类点云投影到2D图像空间中的问题，从而使传统2D卷积神经网络（CNN） ）（例如U-Net）可用于细分。为此，我们受到图绘制的激励，并将其重新构造为整数编程问题，以了解每个单个点云的拓扑保留图到网格映射。为了在实践中加快计算速度，我们进一步提出了一种新颖的分层近似算法。

- 参考论文：

    [https://arxiv.org/abs/2003.05593v2](https://arxiv.org/abs/2003.05593v2)

- 参考实现：

    [https://github.com/Zhang-VISLab/Learning-to-Segment-3D-Point-Clouds-in-2D-Image-Space](https://github.com/Zhang-VISLab/Learning-to-Segment-3D-Point-Clouds-in-2D-Image-Space)

- 适配昇腾 AI 处理器的实现：
  
    [https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/built-in/cv/image_segmentation/ShapeNet_ID1138_for_TensorFlow](https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/built-in/cv/image_segmentation/ShapeNet_ID1138_for_TensorFlow)

- 通过Git获取对应commit\_id的代码方法如下：

    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    ```

#### 默认配置<a name="section91661242121611"></a>
-   网络结构
    -   优化器：Adam
    -   初始学习率：1e-4
    -   单卡batchsize：4
    -   Epoch：1
    -   Steps：15000

-   训练超参（单卡）：
    -   Batch size: 4
    -   Learning rate: 1e-4
    -   Train epoch: 1
    -   Train steps：15000


#### 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 否       |
| 混合精度   | 是       |
| 数据并行   | 否       |


#### 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

#### 开启混合精度<a name="section20779114113713"></a>
相关代码示例。

```
config_proto = tf.ConfigProto(allow_soft_placement=True)
  custom_op = config_proto.graph_options.rewrite_options.custom_optimizers.add()
  custom_op.name = 'NpuOptimizer'
  custom_op.parameter_map["use_off_line"].b = True
  custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
  config_proto.graph_options.rewrite_options.remapping = RewriterConfig.OFF
  session_config = npu_config_proto(config_proto=config_proto)
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

- 模型使用 shapenet_part_seg_hdf5_data 数据集，请用户自行下载，具体获取方法参见 

  ```
  ./ShapeNet_ID1138_for_TensorFlow/S0_download_data.sh
  ```

  

- 获取数据集后，进行数据预处理，并将预处理后的数据放入模型目录下，在训练脚本中指定数据集路径，可正常使用。数据预处理和最终数据集文件结构示例如下：
```
数据预处理，详见：
./ShapeNet_ID1138_for_TensorFlow/S1_network_dataset_combination.py
./ShapeNet_ID1138_for_TensorFlow/S1_network_dataset_preparation.py

最终数据集文件结构示例：
├── ShapeNet_dataset
│   ├── ShapeNet_prepro.hdf5
│   ├── ShapeNet_training.hdf5
则 data_path=./ShapeNet_dataset 即可
```



#### 模型训练

- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练。
  
    1. 启动训练之前，首先要配置程序运行相关环境变量。

       环境变量配置信息参见：

          [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)
    

    2. 单卡训练
       
        2.1 设置单卡训练参数（脚本位于./ShapeNet_ID1138_for_TensorFlow/test/train_full_1p.sh），示例如下。

        
        ```
        #训练epoch
        train_epochs=1
        #训练batch_size
        batch_size=4
        #训练steps
        train_steps=15000
        #数据集文件
        data_path=./ShapeNet_dataset
        ```
        
        2.2 单卡训练指令（脚本位于./ShapeNet_ID1138_for_TensorFlow/test/train_full_1p.sh） 
        
        运行export ASCEND_DEVICE_ID=0 (0~7)以指定单卡训练时使用的卡

        ```
        bash train_full_1p.sh --data_path=xx
        
        数据集应有如下结构（数据切分可能不同），配置data_path时需指定为data这一层，例：--data_path=/home/ShapeNet_dataset
        
        ├── ShapeNet_dataset
        │   ├── ShapeNet_prepro.hdf5
        │   ├── ShapeNet_training.hdf5
        ```

## 迁移学习指导

- 数据集准备。

    1.  获取数据。
        请参见“快速上手”中的数据集准备
    
- 模型训练

    请参考“快速上手”章节

## 高级参考

#### 脚本和示例代码

    ├── README.md                                //说明文档
    ├── requirements.txt                         //依赖
    ├── test
    |    |—— train_full_1p.sh                    //单卡训练脚本
    |    |—— train_performance_1p.sh             //单卡训练脚本
    ├── S2_network_training.py			 // 训练入口脚本


#### 脚本参数

```
batch_size                                       训练batch_size
learning_rate                                    初始学习率
train_epochs                                     总训练epoch数
train_steps                                      总训练steps数
```

#### 训练过程

通过“模型训练”中的训练指令启动单卡训练。
将训练脚本（train_full_1p.sh）中的data_path设置为训练数据集的路径。具体的流程参见“模型训练”的示例。