# UNet3D_for_Tensorflow

## 目录
-   [基本信息](#基本信息)
-   [概述](#概述)
-   [训练环境准备](#训练环境准备)
-   [快速上手](#快速上手)
-   [高级参考](#高级参考)

## 基本信息

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）： Instance Segmentation**

**版本（Version）：1.1**

**修改时间（Modified）：2021.7.19**

**大小（Size）：664K**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Research**

**描述（Description）：利用unet网络进行医学图像分割训练代码**




## 概述

该网络基于之前的 u-net 架构，它使用一个收缩编码器分析整个图像，使用一个连续扩展解码器产生全分辨率分割。虽然 u-net 是一个完全 2D 架构，本文提出的网络采用 3D 数据作为输入并使用相应的 3D 操作处理它们，特别是3D 卷积、3D 最大池化和 3D 上卷积层。此外，我们避免网络架构中的瓶颈并使用批量归一化从而达到更快的收敛。

- 参考论文：

    https://arxiv.org/pdf/1606.06650

- 参考实现：

    https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Segmentation/UNet_3D_Medical

- 适配昇腾 AI 处理器的实现：
  
    https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/built-in/cv/image_segmentation/UNet3D_ID0057_for_TensorFlow
  
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

  - Batch size: 2
  - Learning rate（LR）：0.0001
  - model_dir:result
  - max_steps=80 

#### 支持特性<a name="section1899153513554"></a>

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 是    |
| 混合精度  | 是    |
| 并行数据  | 是    |

#### 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

#### 开启混合精度<a name="section20779114113713"></a>

脚本已默认开启混合精度，设置precision_mode参数的脚本参考如下。

```
run_config = NPURunConfig(
        hcom_parallel=True,
        precision_mode="allow_fp32_to_fp16",
        enable_data_pre_proc=True,
        save_checkpoints_steps=checkpoint_steps,
        session_config=estimator_config,
        iterations_per_loop=iteration_loop_value,
        keep_checkpoint_max=5
    )
```

  

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

     首先在脚本run_performance_1p.sh、train_accuracy_1p.sh、run_performance_8p.sh和run_accuracy_8p.sh中，配置训练数据集路径，请用户根据实际路径配置，数据集参数如下所示：

     ```
     --data_dir=/data/BraTS2019Train
     ```

  2. 启动训练。

     启动单卡性能训练 （脚本为UNet3D_for_Tensorflow/script/run_performance_1p.sh） 

     ```
     bash run_performance_1p.sh  --data_dir=/data/BraTS2019Train
     ```
     启动单卡精度训练 （脚本为UNet3D_for_Tensorflow/script/train_accuracy_1p.sh） 

     ```
     bash train_accuracy_1p.sh  --data_dir=/data/BraTS2019Train
     ```
     启动8卡性能训练 （脚本为UNet3D_for_Tensorflow/script/run_performance_8p.sh） 

     ```
     bash run_performance_8p.sh  --data_dir=/data/BraTS2019Train
     ```
     启动8卡精度训练 （脚本为UNet3D_for_Tensorflow/script/run_accuracy_1p.sh） 

     ```
     bash run_accuracy_8p.sh  --data_dir=/data/BraTS2019Train
     ```

## 高级参考

#### 脚本和示例代码<a name="section08421615141513"></a>

```

├── main.py       
├── export.py            
├── download_dataset.py      
├── LICENSE
├── tf_exports             
│   ├── tf_export.py
├── script               ----精度和性能脚本
│   ├── run_accuracy_1p.sh
│   └── run_performance_1p.sh
│   ├── run_performance_8p.sh
│   └── train_accuracy_1p.sh
│   └── train_performance_1p.sh
│   ├── train_performance_8p.sh
├── readme.md 
|── main_npu.py          
├── requirements

```

#### 脚本参数<a name="section6669162441511"></a>

```
--Rank_Size              使用NPU卡数量，默认：1
--max_steps        训练迭代次数，默认：1000
--data_dir              数据集路径，/data/BraTS2019Train
--batch_size             每个NPU的batch size
--learning_rate          初始学习率，默认：0.0001
```


#### 训练过程<a name="section1589455252218"></a>

1.  通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡、8卡网络训练。

2.  参考脚本的模型存储路径为results/1p或者results/8p，训练脚本log中可查看相关信息。

3.  测试结束后会打印网络运行时间和精度数据，在results/1p或者results/8p中查看相关数据。

