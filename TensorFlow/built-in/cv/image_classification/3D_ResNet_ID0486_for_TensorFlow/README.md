- [基本信息](#基本信息.md)
- [概述](#概述.md)
- [训练环境准备](#训练环境准备.md)
- [快速上手](#快速上手.md)
- [迁移学习指导](#迁移学习指导.md)
- [高级参考](#高级参考.md)

## 基本信息

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Image Classification**

**版本（Version）：1.1**

**修改时间（Modified） ：2022.04.02**

**大小（Size）**_**：81.8K**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的3D_ResNet网络训练代码**

## 概述

-    3D_ResNet是resnet网络的一种3D拓展，对图像分类的效果有一定提升。

- 参考论文：

    [https://arxiv.org/pdf/1512.03385.pdf](https://arxiv.org/pdf/1512.03385.pdf)

- 参考实现：

    [https://github.com/JihongJu/keras-resnet3d](https://github.com/JihongJu/keras-resnet3d)

- 适配昇腾 AI 处理器的实现：
  
    [https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/built-in/cv/image_classification/3D_ResNet_ID0486_for_TensorFlow](https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/built-in/cv/image_classification/3D_ResNet_ID0486_for_TensorFlow)

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
    -   优化器：SGD
    -   单卡batchsize：10
    -   总Epoch数：20

-   训练超参（单卡）：
    -   Batch size: 10
    -   Train epoch: 20


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

- 模型训练使用自制数据集，数据集制作方法如下：


```python
import numpy as np
X_train = np.random.rand(1000, 64, 64, 32, 1)
labels = np.random.randint(0, 2, size=[1000])
y_train = np.eye(2)[labels]
# data_path 为数据集生成路径，需自定义设置
np.save(file="data_path/X_train.npy", arr=X_train)
np.save(file="data_path/labels.npy", arr=labels)
np.save(file="data_path/y_train.npy", arr=y_train)
```

- 生成数据集后，在训练脚本中指定数据集路径，即可正常使用。

#### 模型训练<a name="section715881518135"></a>
- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练。
  
    1. 启动训练之前，首先要配置程序运行相关环境变量。

       环境变量配置信息参见：

          [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)
    

    2. 单卡训练
       
        2.1 设置单卡训练参数（脚本位于./3D_ResNet_ID0486_for_TensorFlow/test/train_full_1p.sh），示例如下。
            
        
        ```
        # 训练epoch
        train_epochs=20
        # 训练batch_size
        batch_size=10
        ```
        

        2.2 单卡训练指令（脚本位于./3D_ResNet_ID0486_for_TensorFlow/test/train_full_1p.sh） 

        ```
        于终端中运行export ASCEND_DEVICE_ID=0 (0~7)以指定单卡训练时使用的卡
        bash train_full_1p.sh --data_path=xx
        数据集应有如下结构（数据切分可能不同），配置data_path时需指定为data这一层，例：--data_path=/home/data
        ├─data
           ├─X_train.npy
           ├─labels.npy
           ├─y_train.npy
        ```

## 迁移学习指导

- 数据集准备。

    1.  获取数据。
        请参见“快速上手”中的数据集准备
    
- 模型训练

    请参考“快速上手”章节

## 高级参考

#### 脚本和示例代码<a name="section08421615141513"></a>


    ├── README.md                                //说明文档
    ├── requirements.txt                         //依赖
    ├── test
    |    |—— train_full_1p.sh                    //单卡训练脚本
    |    |—— train_performance_1p.sh             //单卡训练脚本
    ├── train.py                                 //训练入口脚本


#### 脚本参数<a name="section6669162441511"></a>

```
X_train_file                                    X_train_file
labels_file                                     labels_file
y_train_file                                    y_train_file
batch_size                                      训练batch_size
learning_rate                                   初始学习率
train_epochs                                    总训练epoch数
```

#### 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡训练。
将训练脚本（train_full_1p.sh）中的data_path设置为训练数据集的路径。具体的流程参见“模型训练”的示例。