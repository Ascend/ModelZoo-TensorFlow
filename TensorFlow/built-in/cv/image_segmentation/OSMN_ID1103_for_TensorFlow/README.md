# OSMN: One-Shot Modulation Network for Semi-supervised Video Segmentation

## 目录
* [基本信息](#基本信息)
* [概述](#概述)
* [训练环境准备](#训练环境准备)
* [快速上手](#快速上手)
* [高级参考](#高级参考)

## 基本信息

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Instance Segmentation**

**版本（Version）：1.1**

**修改时间（Modified） ：2021.12.22**

**大小（Size）：829KB**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的OSMN网络训练代码**

## 概述

   OSMN是利用modulators模块快速地调整分割网络使其可以适应特定的物体，而不需要执行数百次的梯度下降；同时不需要调整所有的参数。在视频目标分割上有两个关键的点：视觉外观和空间中持续的移动。为了同时使用视觉和空间信息，将视觉modulator和空间modulator进行合并，在第一帧的标注信息和目标空间位置的基础上分别学习如何调整主体分割网络。

-   参考论文：
      https://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_Efficient_Video_Object_CVPR_2018_paper.pdf

-   参考实现：
      https://github.com/linjieyangsc/video_seg
  
-   适配昇腾 AI 处理器的实现：    
      https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/built-in/cv/image_segmentation/OSMN_ID1103_for_TensorFlow


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

## 安装依赖
参照：requirements.txt

## 快速上手

### 数据集准备<a name="section361114841316"></a>
1. 下载MS-COCO 2017数据集
2. 在TF model zoo中下载VGG16 预训练模型vgg_16.ckpt，放到 `models/` 目录下

### 模型训练<a name="section715881518135"></a>
```
 以数据目录为./data、预训练模型目录为 ./models为例:
  cd test
  source ./env.sh
  bash train_full_1p.sh  --data_path=../data（全量）
  bash train_performance_1p.sh --data_path=../data（功能、性能测试）
 
```


## 高级参考

### 核心脚本和示例代码

```
.
├── models	
├── preprocessing
│   ├── preprocess_davis.py
│   └── preprocess_youtube.py
├── test
│   ├── env.sh
│   ├── train_full_1p.sh	
│   └── train_performance_1p.sh 
├── LICENSE
├── README.md
├── common_args.py
├── dataset_coco.py
├── dataset_davis.py
├── davis_eval.py
├── image_utils.py
├── mobilenet_v1.py
├── model_func.py
├── model_init.txt
├── modelzoo_level.txt
├── ops.py
├── osmn.py
├── osmn_coco_pretrain.py
├── osmn_eval_youtube.py
├── osmn_online_finetune.py
├── osmn_online_finetune_ytvos.py
├── osmn_train_eval.py
├── osmn_train_eval_ytvos.py
├── requirements.txt
├── util.py
├── youtube_eval.py						
└── ytvos_merge_result.py

```

### 脚本参数<a name="section6669162441511"></a>

```
    --data_path                       train data dir, default : path/to/data

```




