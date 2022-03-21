# 训练交付件模板
- [训练交付件模板](#训练交付件模板)
  - [交付件基本信息](#交付件基本信息)
  - [概述](#概述)
  - [默认配置](#默认配置)
  - [混合精度训练](#混合精度训练)
  - [开启混合精度](#开启混合精度)
  - [数据集准备](#数据集准备)
  - [快速上手](#快速上手)

## 交付件基本信息
**发布者（Publisher）：_huawei_**

**应用领域（Application Domain）：_Object Detection_**

**版本（Version）：_1.1_**

**修改时间（Modified）：_2021.05.17_**

**框架（Framework）：_TensorFlow1.15.0_**

**模型格式（Model Format）：_ckpt_**

**精度（Precision）：Mixed**

**处理器（Processor）：_昇腾910_**

**应用级别（Categories）：_Official_**

**描述（Description）：__基于TensorFlow框架的Faster-RCNN图片检测网络训练代码__**

## 概述
Faster R-CNN是截止目前，RCNN系列算法的最杰出产物，two-stage中最为经典的物体检测算法。推理第一阶段先找出图片中待检测物体的anchor矩形框（对背景、待检测物体进行二分类），第二阶段对anchor框内待检测物体进行分类。 R-CNN系列物体检测算法的思路都是，先产生一些待检测框，再对检测框进行分类。Faster R-CNN使用神经网络生成待检测框，替代了其他R-CNN算法中通过规则等产生候选框的方法，从而实现了端到端训练，并且大幅提速。 本文档描述的Faster R-CNN是基于TensorFlow实现的版本。

* 参考实现：

  url=https://github.com/tensorflow/tpu/tree/master/models/official/mask_rcnn

  branch=master

  commit_id=e7be6ecc6d99cd9a77892723a22bfc6715d3d0b9

## 默认配置

-   训练超参（8卡）：
    -   Batch size: 2
    -   Momentum: 0.9
    -   LR scheduler: cosine
    -   Learning rate\(LR\): 0.02
    -   Weight decay: 0.0001
    -   Train steps: 90000

## 混合精度训练

混合精度训练方法是通过混合使用float16和float32数据类型来加速深度神经网络训练的过程，并减少内存使用和存取，从而可以训练更大的神经网络，同时又能基本保持使用float32训练所能达到的网络精度。

## 开启混合精度

在NPU的训练config中设置precision_mode="allow_mix_precision"即可开启NPU上的混合精度模式。

例如：Estimator模式下通过NPURunConfig中的precision_mode参数设置精度模式
```python
npu_config=NPURunConfig(
  model_dir=FLAGS.model_dir,
  precision_mode="allow_mix_precision")
```

**训练环境准备**

硬件环境准备请参见[各硬件产品文档](https://ascend.huawei.com/#/document?tag=developer)。需要在硬件设备上安装固件与驱动。

## 数据集准备

数据集COCO 2017 请用户自行下载

执行以下脚本进行下载和预处理

```
download_and_preprocess_mscoco.sh <data_dir_path>
```

Data will be downloaded, preprocessed to tfrecords format and saved in the directory (on the host).

## 快速上手

1. 下载预训练模型。

   ```
   wget https://storage.googleapis.com/cloud-tpu-checkpoints/detection/classification/resnet-101-imagenet.tar.gz
   tar -xzf resnet-101-imagenet.tar.gz
   mkdir -p {backbone dir}
   mv resnet-101-imagenet {backbone dir}
   ```
   备注：其他resnet系列预训练模型也可以在https://storage.googleapis.com/cloud-tpu-checkpoints中下载

2. 开始训练。
   - 单机单卡

      bash test/[train_full_1p.sh](https://gitee.com/chen-zhuan/modelzoo/blob/master/built-in/TensorFlow/Research/cv/detection/FasterRcnn_for_TensorFlow/test/train_full_1p.sh) --ckpt_path={backbone dir} --data_path={data dir}

   - 单机8卡

      bash test/[train_full_8p.sh](https://gitee.com/chen-zhuan/modelzoo/blob/master/built-in/TensorFlow/Research/cv/detection/FasterRcnn_for_TensorFlow/test/train_full_8p.sh) --ckpt_path={backbone dir} --data_path={data dir}

