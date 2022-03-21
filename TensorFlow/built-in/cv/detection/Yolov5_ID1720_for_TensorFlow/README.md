# 训练交付件

- [交付件基本信息](#交付件基本信息)
- [概述](#概述)
- [默认配置](#默认配置)
- [混合精度训练](#混合精度训练)
- [开启混合精度](#开启混合精度)
- [训练环境准备](#训练环境准备)
- [数据集准备](#数据集准备)
- [快速上手](#快速上手)

## 交付件基本信息

**发布者（Publisher）：_huawei_**

**应用领域（Application Domain）：_Object Detection_**

**版本（Version）：_1.1_**

**修改时间（Modified）：_2021.06.23_**

**框架（Framework）：_TensorFlow1.15.0_**

**模型格式（Model Format）：_ckpt_**

**精度（Precision）：Mixed**

**处理器（Processor）：_昇腾910_**

**应用级别（Categories）：_Reserch_**

**描述（Description）：__基于TensorFlow框架的YOLOv5图片检测网络训练代码__**

## 概述

YOLO创造性的提出了one-stage，解决了传统two-stage目标检测算法普遍存在的运算速度慢的缺点，将物体分类和物体定位在一个步骤中完成。YOLO将物体检测作为回归问题求解，基于一个单独的end-to-end网络，完成从原始图像的输入到物体位置和类别的输出。

- 参考实现：https://github.com/avBuffer/Yolov5_tf

## 默认配置
- 训练超参（8卡）：
  - Batch size: `2`
  - Optimizer: `ADAM`
  - LR scheduler: `cosine`
  - Initial learning-rate: `0.01`
  - Warmup epochs: `3`
  - L2 regularization: `0.0005`
  - Train epochs: `300`

## 混合精度训练

混合精度训练方法是通过混合使用float16和float32数据类型来加速深度神经网络训练的过程，并减少内存使用和存取，从而可以训练更大的神经网络，同时又能基本保持使用float32训练所能达到的网络精度。

## 开启混合精度

在NPU的训练`config`中设置`precision_mode="allow_mix_precision"`即可开启NPU上的混合精度模式。

例如：`sess.run`模式下通过`NPURunConfig`中的`precision_mode`参数设置精度模式

```python
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")

        self.sess = tf.Session(config=npu_config_proto(config_proto=config))
```

## 训练环境准备

硬件环境准备请参见[各硬件产品文档](https://ascend.huawei.com/#/document?tag=developer)。需要在硬件设备上安装固件与驱动。

## 数据集准备

- 下载并解压COCO开源数据集
- 把`gen_anno.py`拷贝到数据集解压文件夹下，修改其中的路径`bbox_data_dir`、`img_dir`、`res_file`
- 运行`python3 gen_anno.py`，分别制作`train_annotation.txt`、`test_annotation.txt`

## 快速上手

- 单机单卡

```shell
bash test/train_full_1p.sh --data_path={data dir}
```

- 单机8卡

```shell
bash test/train_full_8p.sh --data_path={data dir}
```
