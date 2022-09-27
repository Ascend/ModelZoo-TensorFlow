## 基本信息

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）： Image Classification**

**版本（Version）：1.1**

**修改时间（Modified） ：2021.12.27**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的MT-net网络训练代码**

## 概述

MT-net是一种基于参数优化的小样本学习算法，基本思路还是延续了MAML两级训练的元学习思想，在先前的元学习算法基础上增加了一个变换矩阵，得到变换网络Transformation Networks (T-net)，在变换网络的基础上增加了一个二元掩码矩阵得到掩码变换网络Mask Transformation Networks (MT-net)。

- 参考论文：

  [Lee Y ,  Choi S . Gradient-Based Meta-Learning with Learned Layerwise Metric and Subspace[J].  2018.](https://arxiv.org/abs/1801.05558)

- 参考实现：

- 适配昇腾 AI 处理器的实现：

  https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/MT-NET_ID1283_for_TensorFlow

- 通过Git获取对应commit_id的代码方法如下：

  ```
  git clone {repository_url}    # 克隆仓库的代码
  cd {repository_name}    # 切换到模型的代码仓目录
  git checkout  {branch}    # 切换到对应分支
  git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
  cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```

## 混合精度+Loss Scale

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

源码默认开启混合精度+Loss Scale。

## 训练环境准备

1. 硬件环境准备请参见各硬件产品文档"[驱动和固件安装升级指南](https://gitee.com/link?target=https%3A%2F%2Fsupport.huawei.com%2Fenterprise%2Fzh%2Fcategory%2Fai-computing-platform-pid-1557196528909)"。需要在硬件设备上安装与CANN版本配套的固件与驱动。

2. 宿主机上需要安装Docker并登录[Ascend Hub中心](https://gitee.com/link?target=https%3A%2F%2Fascendhub.huawei.com%2F%23%2Fdetail%3Fname%3Dascend-tensorflow-arm)获取镜像。

   当前模型支持的镜像列表如[表1](https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/MT-NET_ID1283_for_TensorFlow#zh-cn_topic_0000001074498056_table1519011227314)所示。

   **表 1** 镜像列表

   | *镜像名称*                                                   | *镜像版本* | *配套CANN版本*                                               |
   | ------------------------------------------------------------ | ---------- | ------------------------------------------------------------ |
   | *ARM架构：[ascend-tensorflow-arm](https://gitee.com/link?target=https%3A%2F%2Fascend.huawei.com%2Fascendhub%2F%23%2Fdetail%3Fname%3Dascend-tensorflow-arm)**x86架构：[ascend-tensorflow-x86](https://gitee.com/link?target=https%3A%2F%2Fascend.huawei.com%2Fascendhub%2F%23%2Fdetail%3Fname%3Dascend-tensorflow-x86)* | *20.2.0*   | *[20.2](https://gitee.com/link?target=https%3A%2F%2Fsupport.huawei.com%2Fenterprise%2Fzh%2Fascend-computing%2Fcann-pid-251168373%2Fsoftware)* |

## 快速上手

- 数据集准备

  选取任务few-shot sine wave regression，使用数据集sinusoid（通过data_generate.py随机生成，不需要额外下载数据集sinusoid）

##  模型训练

- 单击“立即下载”，并选择合适的下载方式下载源码包。

- 启动训练之前，首先要配置程序运行相关环境变量。

  环境变量配置信息参见：

  [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend 910训练平台环境变量设置?sort_id=3148819)

- 执行训练文件（训练脚本为experiments/sine_modelarts_npu.sh）

  ```python
  python boot_modelarts.py
  ```

  训练脚本experiments/sine_modelarts_npu.sh中，超参如下：

  ```
  datasource=sinusoid
  metatrain_iterations=60000
  meta_batch_size=4
  update_lr=0.01
  norm=None
  resume=True
  update_batch_size=10
  use_T=True
  use_M=True
  share_M=True
  # 其余超参参考main.py默认参数
  ```

- 训练成功将会生成文件：

```bash
cls_5.mbs_4.ubs_10.numstep1.updatelr0.01.temp1.0MTnetnonorm
```

## 精度指标

精度指标以postloss为准，GPU精度为0.6980781，NPU精度为0.58435994。

## 性能指标

性能指标以单步迭代的耗时为准，GPU性能为0.019s左右，NPU性能为0.013s左右。











