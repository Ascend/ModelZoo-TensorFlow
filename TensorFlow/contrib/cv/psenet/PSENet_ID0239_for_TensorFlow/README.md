#PSENet_ID0359_for_Tensorflow

## 目录
-   [基本信息](#基本信息)
-   [概述](#概述)
-   [训练环境准备](#训练环境准备)
-   [快速上手](#快速上手)
-   [高级参考](#高级参考)
## 基本信息

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Natural Language Processing**

**版本（Version）：1.1**

**修改时间（Modified） ：2021.7.19**

**大小（Size）：2.0M**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于渐进式尺寸可扩展网络的形状鲁棒文本检测训练代码**

## 概述

PSENet_ID0359_for_Tensorflow是一种基于语义分割的文字框检测模型，本文提出了一种新的实例分割网络，即渐进式扩展网络(PSENet)。PSENet有两方面的优势。 首先，psenet作为一种基于分割的方法，能够对任意形状的文本进行定位.其次，我们提出了一种渐进的尺度扩展算法，该算法可以成功地识别相邻文本实例(如上图所示）。具体地，我们将每个文本实例分配给多个预测的分割区域。为了方便起见，我们将这些分割区域表示为本文中的“核”kernel，并且对于一个文本实例，有几个对应的内核。每个内核与原始的整个文本实例共享相似的形状，并且它们都位于相同的中心点但在比例上不同。为了得到最终的检测结果，我们采用了渐进的尺度扩展算法。它基于宽度优先搜索(BFS)，由三个步骤组成：
从具有最小尺度的核开始(在此步骤中可以区分实例)；
通过逐步在较大的核中加入更多的像素来扩展它们的区域；
完成直到发现最大的核。


- 参考论文：

    https://arxiv.org/pdf/1806.02559.pdf

- 参考实现：

    https://github.com/liuheng92/tensorflow_PSENet

- 适配昇腾 AI 处理器的实现：
  
    https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/psenet/PSENet_ID0239_for_TensorFlow

- 通过Git获取对应commit\_id的代码方法如下：
  
    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    ```

#### 默认配置<a name="section91661242121611"></a>

- 训练数据集预处理（以ImageNet2012训练集为例，仅作为用户参考示例）：

  请参考“概述”->“参考实现”

- 训练超参

  - Batch size: 14
  - Learning rate（LR）：0.0001
  - Train_step: 100000


#### 支持特性<a name="section1899153513554"></a>

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 否    |
| 混合精度  | 是    |
| 并行数据  | 否    |

#### 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

#### 开启混合精度<a name="section20779114113713"></a>

脚本已默认开启混合精度，设置precision_mode参数的脚本参考如下。

```
  run_config = NPURunConfig(        
  		model_dir=flags_obj.model_dir,        
  		session_config=session_config,        
  		keep_checkpoint_max=5,        
  		save_checkpoints_steps=5000,        
  		enable_data_pre_proc=True,        
  		iterations_per_loop=iterations_per_loop,        			
  		log_step_count_steps=iterations_per_loop,        
  		precision_mode='allow_mix_precision',        
  		hcom_parallel=True      
        )

  precision_mode="allow_mix_precision"
  opt = NPUDistributedOptimizer(opt)
  loss_scale_manager = FixedLossScaleManager(loss_scale=FLAGS.loss_scale)
  rank_size = int(os.getenv('RANK_SIZE'))
```

	if rank_size > 1:
		opt = NPULossScaleOptimizer(opt, loss_scale_manager, is_distributed=True)
	else:
		opt = NPULossScaleOptimizer(opt, loss_scale_manager, is_distributed=False)


## 训练环境准备

1.  硬件环境准备请参见各硬件产品文档"[驱动和固件安装升级指南]( https://support.huawei.com/enterprise/zh/category/ai-computing-platform-pid-1557196528909)"。需要在硬件设备上安装与CANN版本配套的固件与驱动。
2.  宿主机上需要安装Docker并登录[Ascend Hub中心](https://ascendhub.huawei.com/#/detail?name=ascend-tensorflow-arm)获取镜像。

    当前模型支持的镜像列表如[表1](#zh-cn_topic_0000001074498056_table1519011227314)所示。

    **表 1** 镜像列表

    <a name="zh-cn_topic_0000001074498056_table1519011227314"></a>
    <table><thead align="left"><tr id="zh-cn_topic_0000001074498056_row0190152218319"><th class="cellrowborder" valign="top" width="47.32%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001074498056_p1419132211315"><a name="zh-cn_topic_0000001074498056_p1419132211315"></a><a name="zh-cn_topic_0000001074498056_p1419132211315"></a><em id="i1522884921219"><a name="i1522884921219"></a><a name="i1522884921219"></a>镜像名称</em></p>
    </th>
    <th class="cellrowborder" valign="top" width="25.52%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001074498056_p75071327115313"><a name="zh-cn_topic_0000001074498056_p75071327115313"></a><a name="zh-cn_topic_0000001074498056_p75071327115313"></a><em id="i1522994919122"><a name="i1522994919122"></a><a name="i1522994919122"></a>镜像版本</em></p>
    </th>
    <th class="cellrowborder" valign="top" width="27.16%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001074498056_p1024411406234"><a name="zh-cn_topic_0000001074498056_p1024411406234"></a><a name="zh-cn_topic_0000001074498056_p1024411406234"></a><em id="i723012493123"><a name="i723012493123"></a><a name="i723012493123"></a>配套CANN版本</em></p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="zh-cn_topic_0000001074498056_row71915221134"><td class="cellrowborder" valign="top" width="47.32%" headers="mcps1.2.4.1.1 "><a name="zh-cn_topic_0000001074498056_ul81691515131910"></a><a name="zh-cn_topic_0000001074498056_ul81691515131910"></a><ul id="zh-cn_topic_0000001074498056_ul81691515131910"><li><em id="i82326495129"><a name="i82326495129"></a><a name="i82326495129"></a>ARM架构：<a href="https://ascend.huawei.com/ascendhub/#/detail?name=ascend-tensorflow-arm" target="_blank" rel="noopener noreferrer">ascend-tensorflow-arm</a></em></li><li><em id="i18233184918125"><a name="i18233184918125"></a><a name="i18233184918125"></a>x86架构：<a href="https://ascend.huawei.com/ascendhub/#/detail?name=ascend-tensorflow-x86" target="_blank" rel="noopener noreferrer">ascend-tensorflow-x86</a></em></li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="25.52%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001074498056_p1450714271532"><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><em id="i72359495125"><a name="i72359495125"></a><a name="i72359495125"></a>21.0.2</em></p>
    </td>
    <td class="cellrowborder" valign="top" width="27.16%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001074498056_p18244640152312"><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><em id="i162363492129"><a name="i162363492129"></a><a name="i162363492129"></a><a href="https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software" target="_blank" rel="noopener noreferrer">5.0.2</a></em></p>
    </td>
    </tr>
    </tbody>
    </table>


## 快速上手

#### 数据集准备

请参考“概述”->“参考实现”

#### 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。

- 启动训练之前，首先要配置程序运行相关环境变量。

  环境变量配置信息参见：

     [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)

- 单卡训练 

  1. 配置训练参数。

     首先在脚本train_performance_1p.sh和train_full_1p.sh中，配置训练数据集路径，请用户根据实际路径配置，数据集参数如下所示：

     ```
      data_path = /npu/traindata/icdar2013_2015
     ```

  2. 启动训练。

     启动单卡性能训练 （脚本为PSENet_ID0359_for_Tensorflow/test/train_performance_1p.sh） 

     ```
     bash train_performance_1p.sh --data_path=/npu/traindata/icdar2013_2015
     ```
     启动单卡精度训练 （脚本为PSENet_ID0359_for_Tensorflow/test/train_full_1p.sh） 

     ```
     bash train_full_1p.sh --data_path=/npu/traindata/icdar2013_2015
     ```

## 高级参考

#### 脚本和示例代码<a name="section08421615141513"></a>

```shell
├── checkpoint        ----存放训练ckpt的路径
├── eval.py           ----推理入口py     
├── eval.sh           ----推理shell，计算icdar2015测试集的精度、召回率、F1 Score
├── evaluation        ----精度计算相关的py，新增
├── LICENSE
├── nets              ----网络模型定义，包含backbone
│   ├── __init__.py
│   ├── model.py
│   ├── __pycache__
│   └── resnet
├── npu_train.py      ----NPU训练
├── ocr               ----数据集存放目录
│   ├── ch4_test_images  --test图片
│   └── icdar2015        --train图片
├── pretrain_model    ----backbone
├── pse               ----后处理PSE代码
│   ├── include
│   ├── __init__.py
│   ├── Makefile
│   ├── pse.cpp
│   ├── pse.so
│   └── __pycache__
├── readme.md
├── train_npu.sh     ----NPU训练入口shell
├── train.py         ----GPU训练
└── utils            ----数据集读取和预处理
    ├── data_provider
    ├── __init__.py
    ├── __pycache__
    └── utils_tool.py
```

#### 脚本参数<a name="section6669162441511"></a>

```
--Rank_Size              使用NPU卡数量，默认：1
--train_steps        训练迭代次数，默认：400
--precision_mode         NPU运行时，默认开启混合精度
--data_path              数据集路径，/npu/traindata/icdar2013_2015
--batch_size             每个NPU的batch size，默认：14
--learning_rate          初始学习率，默认：0.0001
```


#### 训练过程<a name="section1589455252218"></a>

1.  通过“模型训练”中的训练指令启动性能或者精度训练。性能和精度通过运行不同脚本，支持性能、精度网络训练。

2.  参考脚本的模型存储路径为test/output/*，训练脚本train_*.log中可查看性能、精度的相关运行状态。

3.  测试结束后会打印网络运行时间和精度数据，在test/output/*/train_*.log中可查看相关数据。
