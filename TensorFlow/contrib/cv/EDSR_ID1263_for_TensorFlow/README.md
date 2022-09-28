-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [快速上手](#快速上手.md)
-   [迁移学习指导](#迁移学习指导.md)
-   [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：** 超分辨率网络

**版本（Version）：1.1**

**修改时间（Modified） ：2021.11.16**

**大小（Size）：14.2M**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Research**

**描述（Description）：基于TensorFlow框架的EDSR超分辨率网络训练代码** 

<h2 id="概述.md">概述</h2>

EDSR网络是一个单图像超分辨率增强深度残差网络。这是EDSR模型，每个比例有不同的模型。架构如下所示。、
![Alt text](images/EDSR.png?raw=true "EDSR architecture") 

- 参考论文：

    [Bee Lim ,Sanghyun Son, Heewon Kim, Seungjun Nah, Kyoung Mu Lee. “Enhanced Deep Residual Networks for Single Image Super-Resolution.” arXiv:1707.02921v1](https://arxiv.org/abs/1707.02921) 

- 参考实现：

    

- 适配昇腾 AI 处理器的实现：
    
  https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/EDSR_ID1263_for_TensorFlow
        


- 通过Git获取对应commit\_id的代码方法如下：
    
    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    ```

## 默认配置<a name="section91661242121611"></a>

- 训练数据集预处理（以Div2K训练集为例，仅作为用户参考示例）：

  - 图像的输入尺寸为48*48
  - 随机翻转图像
  - 比例整除-创建低分辨率
  - 根据ImageNet数据集通用的平均值和标准偏差对输入图像进行归一化

- 测试数据集预处理（以Div2K验证集为例，仅作为用户参考示例）

  - 图像的输入尺寸为48*48
  - 根据ImageNet数据集通用的平均值和标准偏差对输入图像进行归一化

- 训练超参

  - Batch size: 16
  - Momentum: 0.9
  - Learning rate(lr): 0.0001
  - Optimizer: Adam
  - num_blocks:32
  - num_filters:256
  - from_scrach:True
  - scale:2
  - decay_rate: 0.9
  - decay_step:500
  - save_step:5
  - train_steps: 100


## 支持特性<a name="section1899153513554"></a>

| 特性列表  | 是否支持 |
|-------|------|
| 混合精度  | 是    |

## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

脚本已默认开启混合精度，设置参数参考如下。

  ```
    elif chip == 'npu':
        import npu_bridge
        from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

        config = tf.ConfigProto()
        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        custom_op.parameter_map["use_off_line"].b = True
        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
  ```


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
    <td class="cellrowborder" valign="top" width="25.52%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001074498056_p1450714271532"><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><em id="i72359495125"><a name="i72359495125"></a><a name="i72359495125"></a>21.0.2</em></p>
    </td>
    <td class="cellrowborder" valign="top" width="27.16%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001074498056_p18244640152312"><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><em id="i162363492129"><a name="i162363492129"></a><a name="i162363492129"></a><a href="https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software" target="_blank" rel="noopener noreferrer">21.0.2</a></em></p>
    </td>
    </tr>
    </tbody>
    </table>


<h2 id="快速上手.md">快速上手</h2>

- 数据集准备
1. [模型训练使用Div2K数据集](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=WDId8mhPMoBh6LweIUTNus3dwVTw4IfwBduzfdyKIf6Ikg/Jsq56SJ8Vgh3+YEsG0qqAlmjnbecCHNw67usMN/lbLAYzGfSmYIEAVqrErCfe8IU9OvSkHSvtCnl+zBpnh883nZs7o5v+PtknEKX9B10XGbvhTeh4Kc1IVfGCnGnnCSDfPqPlMsWxSq/SaxQ/xITmnomu5T69jZtZsTeYhDjBO3oX1oi5aeElVU38rXnkilsmiNzQlq9I39ry8Hvmv97dSZvODPlZdGHKcfGdAgLir/TYzbIWNBraS0kAj5cas65LnR7BKrKOnuXeX93tMnOfBWbjypEcRSjQIb7vhAmKd9jzEP9nqTGV0BBeZ9PvDIjEHHB8bOkCNlBJNucKXdYiJsh5iRYsO5Ycd+SiF1MbG88G2nnZwVPMl7z/VYm+AIAV4LpMVcwotw+x9M5ewiFRCM0kKeSR7zILNTY+Pm9FmtFNO2+S2Cc8aZnS2I9w+SBV/KgTK+hiIz39mSkQcARD6GzwyVOKLrJU37bZ8A==)，提取码为123456，数据集请用户自行获取。

2. 数据集处理后，放入模型目录下，在训练脚本中指定数据集路径，可正常使用。
   

## 模型训练<a name="section715881518135"></a>

- 单击[“立即下载”](https://github.com/LimBee/NTIRE2017)，并选择合适的下载方式下载源码包。

- 启动训练之前，首先要配置程序运行相关环境变量。

  环境变量配置信息参见：

     [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)

- 单卡训练 

  1. 配置训练参数。

     训练数据集放在obs桶中，配置训练数据集路径，请用户根据实际路径配置，数据集参数如下所示：

     ```
      --data_dir=edsr/cache/DIV2K
     ```

  2. 启动训练。

     启动单卡训练 （脚本为EDSR_ID1263_for_TensorFlow/scripts/run_1p.sh） 

     ```
     bash run_1p.sh
     ```

<h2 id="迁移学习指导.md">迁移学习指导</h2>

- 数据集准备。

  数据集要求如下：

  1. 获取数据。

     如果要使用自己的数据集，需要将数据集放到自己的obs桶中。参考代码中的数据集存放路径如下：

     - 训练集： /edsr/cache/DIV2K
     - 测试集： /edsr/cache/DIV2K

     训练数据集和测试数据集以文件名中的train和valid加以区分。

     数据集也可以放在其它目录，则修改对应的脚本入参data_dir即可。

  2. 准确标注类别标签的数据集。

  3. 数据集每个类别所占比例大致相同。


<h2 id="高级参考.md">高级参考</h2>


## 训练过程<a name="section1589455252218"></a>

1.  通过“模型训练”中的训练指令启动单卡训练。

2.  参考脚本的模型存储路径为obs://edsr0/log/edsr-9/log/，训练脚本log中包括如下信息。

```
 96%|█████████▌| 766/800 [01:25<00:03,  9.69it/s]
 96%|█████████▌| 768/800 [01:25<00:03,  9.83it/s]
 96%|█████████▋| 770/800 [01:25<00:03,  9.93it/s]
 96%|█████████▋| 772/800 [01:26<00:03,  7.82it/s]
 97%|█████████▋| 774/800 [01:26<00:03,  8.40it/s]
 97%|█████████▋| 776/800 [01:26<00:02,  8.87it/s]
 97%|█████████▋| 778/800 [01:26<00:02,  9.24it/s]
 98%|█████████▊| 780/800 [01:27<00:02,  9.51it/s]
 98%|█████████▊| 782/800 [01:27<00:01,  9.72it/s]
 98%|█████████▊| 784/800 [01:27<00:01,  9.86it/s]
 98%|█████████▊| 786/800 [01:27<00:01,  9.96it/s]
 98%|█████████▊| 788/800 [01:27<00:01,  9.87it/s]
 99%|█████████▉| 790/800 [01:28<00:01,  9.96it/s]
 99%|█████████▉| 792/800 [01:28<00:00,  8.26it/s]
 99%|█████████▉| 794/800 [01:28<00:00,  8.76it/s]
100%|█████████▉| 796/800 [01:28<00:00,  9.16it/s]
100%|█████████▉| 798/800 [01:29<00:00,  9.46it/s]
100%|██████████| 800/800 [01:29<00:00,  9.70it/s]
100%|██████████| 800/800 [01:29<00:00,  8.96it/s]
Step nr: [800/?] - Loss: 3.87270 - Lr: 0.0000950
```

## 推理过程/NPU网络训练精度<a name="section1465595372416"></a>

1.  通过“模型训练”中的测试指令启动测试。

3.  推理脚本的参数eval_dir可以配置为checkpoint所在的文件夹路径，则该路径下所有.ckpt文件都会根据进行推理。

4.  测试结束后会打印验证集的最终训练结果，如下所示。

```
Epoch nr: [100/100]  - Loss: 2.95130 - val PSNR: 33.720 - val SSIM: 0.927

Training finished.
===>>>Copy Event or Checkpoint from modelarts dir:/cache/result to obs:s3://edsr0/log/edsr-9/output/null/result
I ran successfully.
[Modelarts Service Log]2021-09-08 19:09:28,962 - INFO - Begin destroy training processes
[Modelarts Service Log]2021-09-08 19:09:28,963 - INFO - proc-rank-0-device-0 (pid: 107) has exited
[Modelarts Service Log]2021-09-08 19:09:28,964 - INFO - End destroy training processes

```
## GPU 网络训练精度
```
100%|██████████| 800/800 [11:10<00:00,  1.19it/s]
Step nr: [800/?] - Loss: 3.28985 - Lr: 0.0000774
100%|██████████| 800/800 [11:09<00:00,  1.20it/s]
Step nr: [800/?] - Loss: 3.25910 - Lr: 0.0000774
100%|██████████| 100/100 [11:03<00:00,  6.63s/it]
Epoch nr: [100/100]  - Loss: 3.25910 - val PSNR: 33.258 - val SSIM: 0.922

Training finished.
I ran successfully.

Process finished with exit code 0
```
## NPU/GPU 网络训练性能 
| NPU  | GPU |
|-------|------|
| 1.67min/epoch| 2.93min/epoch|
```
其中GPU为v100
```
## 综合评价
NPU上训练后的精度与GPU基本一致，NPU训练结果略高于GPU，但是达不到论文上的结果。
NPU在训练性能上远超GPU。

