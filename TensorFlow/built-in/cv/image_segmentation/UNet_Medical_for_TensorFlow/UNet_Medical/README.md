-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [快速上手](#快速上手.md)
-   [迁移学习指导](#迁移学习指导.md)
-   [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Instance Segmentation**

**版本（Version）：1.1**

**修改时间（Modified） ：2021.08.04**

**大小（Size）：2.1MB**

**框架（Framework）：TensorFlow_1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：FP32**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的网络训练代码**

<h2 id="概述.md">概述</h2>

UNet 模型是用于 2D 图像分割的卷积神经网络。该模型在 Volta、Turing 和 NVIDIA Ampere GPU 架构上使用 Tensor Cores 以混合精度进行训练。因此，研究人员可以获得比没有 Tensor Cores 的训练快 2.2 倍的结果，同时体验混合精度训练的好处。

-   参考论文：

    https://arxiv.org/abs/1505.04597
-   参考实现：

    https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Segmentation/UNet_Medical
-   适配昇腾 AI 处理器的实现：
    
    
     https://gitee.com/ascend/modelzoo/tree/master/built-in/TensorFlow/Research/cv/image_segmentation/UNet_Medical_for_TensorFlow/UNet_Medical
        

-   通过Git获取对应commit\_id的代码方法如下：
    
    
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    

## 默认配置<a name="section91661242121611"></a>

-   网络结构
    -   初始学习率为0.0001
    -   单卡batchsize：1
    -   模式train_and_evaluate

-   训练数据集：
    -   使用自带脚本准备数据集 python download_dataset.py --data_dir /data

## 支持特性<a name="section1899153513554"></a>

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 是    |
| 混合精度  | 是    |
| 数据并行  | 是    |


## 混合精度训练<a name="section168064817164"></a>

 混合精度训练昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

    run_config = NPURunConfig(
        save_summary_steps=params.max_steps,
        model_dir=params.model_dir,
        session_config=config,
        save_checkpoints_steps=params.max_steps,
        keep_checkpoint_max=5,
        enable_data_pre_proc=True,
        log_step_count_steps=10,
        iterations_per_loop=params.iterations_per_loop,
        precision_mode='allow_mix_precision' if params.use_amp else None,
        hcom_parallel=True
    )

<h2 id="训练环境准备.md">训练环境准备</h2>

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
    <td class="cellrowborder" valign="top" width="25.52%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001074498056_p1450714271532"><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><em id="i72359495125"><a name="i72359495125"></a><a name="i72359495125"></a>20.2.0</em></p>
    </td>
    <td class="cellrowborder" valign="top" width="27.16%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001074498056_p18244640152312"><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><em id="i162363492129"><a name="i162363492129"></a><a name="i162363492129"></a><a href="https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software" target="_blank" rel="noopener noreferrer">20.2</a></em></p>
    </td>
    </tr>
    </tbody>
    </table>


<h2 id="快速上手.md">快速上手</h2>

## 数据集准备<a name="section361114841316"></a>

1. 训练模型使用ssTEM数据集，用户自行准备好数据集
2. 数据集的下载及处理，请用户参考”概述--> 参考实现“ 开源代码处理
3. 数据集处理后放在模型目录下，在训练脚本中指定数据集路径，可正常使用

## 模型训练<a name="section715881518135"></a>

-  单击“立即下载”，并选择合适的下载方式下载源码包。
-  开始训练    
   
    1. 启动训练之前，首先要配置程序运行相关环境变量。

       环境变量配置信息参见：

          [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)
    
    2. 单卡训练 

        2.1 配置loop.sh脚本中`data_dir`,请用户根据实际路径配置，数据集参数如下所示：

            --data_dir=/data

        2.2 单p指令如下:

            bash loop.sh

 

<h2 id="迁移学习指导.md">迁移学习指导</h2>

- 数据集准备。

    1. 请参见“快速上手”中的数据集准备。

-   模型训练。

    参考“模型训练”中训练步骤。


<h2 id="高级参考.md">高级参考</h2>

脚本和示例代码

```
.
├── Dockerfile                           // 构建容器文件
├── LICENSE
├── NOTICE
├── README.md
├── test
│   ├── env.sh
│   ├── train_full_1p.sh
│   ├── train_performance_1p.sh
├── dllogger
│   ├── logger.py
└── examples

```


## 脚本参数<a name="section6669162441511"></a>

```
 --exec_mode: Select the execution mode to run the model (default: train). Modes available:
      evaluate - loads checkpoint (if available) and performs evaluation on validation subset (requires --crossvalidation_idx other than None).
      train_and_evaluate - trains model from scratch and performs validation at the end (requires --crossvalidation_idx other than None).
      predict - loads checkpoint (if available) and runs inference on the test set. Stores the results in --model_dir directory.
      train_and_predict - trains model from scratch and performs inference.
--model_dir: Set the output directory for information related to the model (default: /results).
--log_dir: Set the output directory for logs (default: None).
--data_dir: Set the input directory containing the dataset (default: None).
--batch_size: Size of each minibatch per GPU (default: 1).
--crossvalidation_idx: Selected fold for cross-validation (default: None).
--max_steps: Maximum number of steps (batches) for training (default: 1000).
--seed: Set random seed for reproducibility (default: 0).
--weight_decay: Weight decay coefficient (default: 0.0005).
--log_every: Log performance every n steps (default: 100).
--learning_rate: Model’s learning rate (default: 0.0001).
--augment: Enable data augmentation (default: False).
--benchmark: Enable performance benchmarking (default: False). If the flag is set, the script runs in a benchmark mode - each iteration is timed and the performance result (in images per second) is printed at the end. Works for both train and predict execution modes.
--warmup_steps: Used during benchmarking - the number of steps to skip (default: 200). First iterations are usually much slower since the graph is being constructed. Skipping the initial iterations is required for a fair performance assessment.
--use_xla: Enable accelerated linear algebra optimization (default: False).
--use_amp: Enable automatic mixed precision (default: False).

```

## 训练过程<a name="section1589455252218"></a>

1. 通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡网络训练。 
2. 将训练脚本（loop.sh）中的data_dir设置为训练数据集的路径。具体的流程参见“模型训练”的示例。 
3. 模型存储路径为“${cur_path}/output/$ASCEND_DEVICE_ID”，包括训练的log以及checkpoints文件。
4. 以单卡训练为例，loss信息在文件${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中，示例如下。 

