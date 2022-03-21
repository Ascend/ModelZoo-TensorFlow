1. [基本信息](#基本信息.md)
2. [概述](#概述.md)
3. [训练环境准备](#训练环境准备.md)
4. [快速上手](#快速上手.md)
5. [迁移学习指导](#迁移学习指导.md)
6. [高级参考](#高级参考.md)

## 基本信息

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Natural Language Processing**

**版本（Version）：1.1**

**修改时间（Modified） ：2021.04.08**

**大小（Size）**_**：【深加工】**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：84%~88%**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：finetune**

**描述（Description）：基于TensorFlow框架的bertbase和bertlarge网络在MRPC数据集上的finetune**

## 概述

- BERT是一种预训练语言表示的方法，这意味着我们在大型文本语料库（例如Wikipedia）上训练通用的“语言理解”模型，然后将该模型用于我们关心的下游NLP任务（例如问题回答）。BERT优于以前的方法，因为它是第一个用于预训练NLP的*无监督*，*深度双向*系统。

- 参考论文

  ```
  https://arxiv.org/pdf/1810.04805.pdf
  ```

- 参考实现

  ```
  https://github.com/google-research/bert
  ```
  
- 适配昇腾 AI 处理器的实现

  ```
  https://gitee.com/ascend/modelzoo/tree/master/built-in/TensorFlow/Research/nlp/Bert_Google_finetune_for_TensorFlow
  ```

​    

### 默认配置

- 网络结构

  初始学习率为5e-5

  单卡batchsize：32

- 训练超参（单卡）

  ```
  - Batch size: 32
  - Learning rate(LR): 2e-5
  - Train epoch: 3.0
  ```

### 支持特性

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 否       |
| 混合精度   | 是       |
| 数据并行   | 是       |

### 混合精度训练

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

### 开启混合精度 

相关代码示例。

```
run_config = NPURunConfig(
        model_dir=self.config.model_dir,
        session_config=session_config,
        keep_checkpoint_max=5,
        save_checkpoints_steps=5000,
        enable_data_pre_proc=True,
        iterations_per_loop=iterations_per_loop,
        precision_mode='allow_mix_precision',
        hcom_parallel=True
      ）
```

## 训练环境准备

1. 硬件环境准备请参见各硬件产品文档"[驱动和固件安装升级指南](https://support.huawei.com/enterprise/zh/category/ai-computing-platform-pid-1557196528909)"。需要在硬件设备上安装与CANN版本配套的固件与驱动。

2. 宿主机上需要安装Docker并登录[Ascend Hub中心](https://ascendhub.huawei.com/#/detail?name=ascend-tensorflow-arm)获取镜像。

   当前模型支持的镜像列表如[表1](#zh-cn_topic_0000001074498056_table1519011227314)所示。

   **表 1** 镜像列表

   | *镜像名称*                                                   | *镜像版本* | *配套CANN版本*                                               |
| ------------------------------------------------------------ | ---------- | ------------------------------------------------------------ |
   | *ARM架构：[ascend-tensorflow-arm](https://ascend.huawei.com/ascendhub/#/detail?name=ascend-tensorflow-arm)**x86架构：[ascend-tensorflow-x86](https://ascend.huawei.com/ascendhub/#/detail?name=ascend-tensorflow-x86)* | *20.2.0*   | *[20.2](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software)* |

## 快速上手

## 数据集准备 

1. 下载GLUE data数据集
2. 下载预训练模型[BERT-Base, Uncased],[BERT-Large, Uncased]
3. 数据集的处理可以参考"简述->参考实现"处理

## 模型训练 

- 下载训练脚本

- 执行finetune前，请根据实际情况修改env.sh中的环境变量；本次finetune使用的数据集是GLUE DATA中的MRPC。

  参数如下：

  ```
  --task_name           finetune dataset
  --data_path           source data of training
  --model_path          the path of pretrain ckpt
  --train_batch_size    training batch
  --learning_rate       learning_rate
  --num_train_epochs    epochs
  --output_dir          output dir
  ```

  执行finetune

  ```
  cd test
  source env.sh
  bash train_full_1p.sh --data_path=glue_data/MRPC --model_path=BertBase_ckpt
  ```

## 执行结果

Bertbase:

```
---------Final Result--------
Final Precision Accuracy : 0.8504902
Final Performance ms/step : 83.29
Final Training Duration sec : 476
```

Bertlarge:

```
---------Final Result--------
Final Precision Accuracy : 0.6838235
Final Performance ms/step : 139.26
Final Training Duration sec : 706
```

