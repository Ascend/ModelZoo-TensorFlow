- [基本信息](#基本信息.md)
- [概述](#概述.md)
- [训练环境准备](#训练环境准备.md)
- [快速上手](#快速上手.md)
- [迁移学习指导](#迁移学习指导.md)
- [高级参考](#高级参考.md)

## 基本信息

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Recommendation**

**版本（Version）：1.1**

**修改时间（Modified） ：2022.04.08**

**大小（Size）：512K**

**框架（Framework）：TensorFlow_2.6.2**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Research**

**描述（Description）：基于TensorFlow2.X框架的推荐网络训练代码**


## 概述

   STAMP模型的全称是：Short-Term Attention/Memory Priority Model for Session-based Recommendation。该模型是一种新的短期注意/记忆优先级模型，该模型能够从会话上下文的长期记忆中捕获用户的通用兴趣，同时从最后点击的短期记忆中考虑用户当前的兴趣。


  - 参考实现：
    https://github.com/ZiyaoGeng/Recommender-System-with-TF2.0/tree/master/STAMP(https://github.com/ZiyaoGeng/Recommender-System-with-TF2.0/tree/master/STAMP)


  - 适配昇腾 AI 处理器的实现：

    https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow2/built-in/recommendation/STAMP_ID2628_for_TensorFlow2.X

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
    -   16-layer,  4M parameters


-   训练超参（单卡）：
    -   Batch size: 128
    -   maxlen：40
    -   embed_dim：100
    -   learning_rate: 0.005
    -   Train epoch: 30


#### 支持特性<a name="section1899153513554"></a>

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 否    |
| 混合精度  | 是    |
| 数据并行  | 否    |

#### 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

#### 开启混合精度<a name="section20779114113713"></a>
拉起脚本中，传入--precision_mode='allow_mix_precision'

```
 ./train_performance_1p_16bs.sh --help

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
flags.DEFINE_string(name='precision_mode', default= 'allow_fp32_to_fp16',
                    help='allow_fp32_to_fp16/force_fp16/ ' 
                    'must_keep_origin_dtype/allow_mix_precision.')

npu_device.global_options().precision_mode=FLAGS.precision_mode
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

采用Diginetica数据集进行测试，将其处理为用户序列。数据集的处理见utils文件，主要分为：

-    读取数据（可以取部分数据进行测试）；
-    过滤掉session长度为1的样本；
-    过滤掉包含某物品（出现次数小于5）的样本；
-    对特征itemId进行LabelEncoder，将其转化为0, 1,...范围；
-    按照evetdate、sessionId排序；
-    按照eventdate划分训练集、验证集、测试集；
-    生成序列【无负样本】，生成新的数据，格式为hist, label，因此需要使用tf.keras.preprocessing.sequence.pad_sequences方法进行填充/切割，此外，由于序列中只有一个特征item_id，经过填充/切割后，维度会缺失，所以需要进行增添维度；
-    生成一个物品池item pooling：物品池按序号排序；
-    得到feature_columns：无密集数据，稀疏数据为item_id；
-    生成用户行为列表，方便后续序列Embedding的提取，在此处，即item_id；
-    最后返回feature_columns, behavior_list, (train_X, train_y), (val_X, val_y), (test_X, test_y)；


#### 模型训练<a name="section715881518135"></a>
- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练。
  
    1. 启动训练之前，首先要配置程序运行相关环境变量。

       环境变量配置信息参见：

          [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)

2. 单卡训练
   
    2.1 设置单卡训练参数（脚本位于STAMP_ID2628_for_TensorFlow2.X/test/train_full_1p.sh）
    
    ```
       bash train_full_1p.sh  --precision_mode='allow_mix_precision'
    ```

    2.2 单卡训练指令（STAMP_ID2628_for_TensorFlow2.X/test） 
    
    ```
    于终端中运行export ASCEND_DEVICE_ID=0 (0~7)以指定单卡训练时使用的卡
    bash train_full_1p.sh --data_path=xx
    数据集应为h5类型，配置data_path时需指定为data这一层，例：--data_path=/home/data
    ├─data
       ├─product-categories.csv
       ├─products.csv
       ├─train-clicks.csv
       ├─train-item-views.csv
       ├─train-purchases.csv
       ├─train-queries.csv
     
    ```



## 迁移学习指导


1.  数据集准备。
    请参见“快速上手”中的数据集准备

2.  修改训练脚本。

    _（修改模型配置文件、模型脚本，根据客户实际业务数据做对应模型的修改，以适配）_

    1.  修改配置文件。

    2.  加载预训练模型。_（预加载模型继续训练或者使用用户的数据集继续训练）_

3.  模型训练。

    _可以参考“模型训练”中训练步骤。（根据实际情况，开源数据集与自定义数据集的训练方法是否一致？）_

4.  模型评估。（根据实际情况）_可以参考“模型训练”中训练步骤。_

## 高级参考

#### 脚本参数<a name="section6669162441511"></a>

```
    --data_path                                                   default='./',help="""directory to data"""
    --batch_size                                               default=128, type=int,help="""batch size for 1p"""
    --epochs                                                    default=30, type=int,help="""epochs"""
    --steps_per_epoch                                         default=50, type=int,help="""Eval batch size"""
    --learning_rate                                      default=0.005, type=float,help="""The value of learning_rate"""
    --precision_mode                                  default="allow_mix_precision", type=str,help='the path to save over dump data'
    --over_dump                                dest='over_dump', type=ast.literal_eval,help='if or not over detection, default is False'
    --data_dump_flag                           dest='data_dump_flag', type=ast.literal_eval,help='data dump flag, default is False'
    --data_dump_step                           default="10",help='data dump step, default is 10'
    --profiling                              dest='profiling', type=ast.literal_eval help='if or not profiling for performance debug, default is False'
    --profiling_dump_path                    default="/home/data", type=str, help='the path to save profiling data'
    --over_dump_path                         default="/home/data", type=str, help='the path to save over dump data'
    --data_dump_path                         default="/home/data", type=str, help='the path to save dump data'
    --use_mixlist                            dest='use_mixlist', type=ast.literal_eval,help='use_mixlist flag, default is False'
    --fusion_off_flag                        dest='fusion_off_flag', type=ast.literal_eval,help='fusion_off flag, default is False'
    --mixlist_file                           default="ops_info.json", type=str,help='mixlist file name, default is ops_info.json'
    --fusion_off_file                        default="fusion_switch.cfg", type=str,help='fusion_off file name, default is fusion_switch.cfg'
```

#### 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡，8卡网络训练。模型存储路径为${cur_path}/output/$ASCEND_DEVICE_ID，包括训练的log以及checkpoints文件。以8卡训练为例，loss信息在文件${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。
