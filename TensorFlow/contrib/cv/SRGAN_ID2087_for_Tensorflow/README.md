
# SRGAN 介绍

## 基本信息

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：** 目标检测

**版本（Version）：1.1**

**修改时间（Modified） ：2021.11.2**2

**大小（Size）：77M**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：.h5**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Research**

**描述（Description）：基于TF1.15+keras2.2.4的SRGAN复现**

##  概述

SRGAN是使用了生成对抗网络来训练SRResNet,使其产生的HR图像看起来更加自然,有更好的视觉效果（SRResNet是生成网络，对抗网络是用来区分真实的HR图像和通过SRResNet还原出来的HR图像。

论文中 并没有说明确的loss，但是可以通过生成图看出效果的好坏。原论文中给出的是三个评分PSNR，SSIM，MOS

本项目 生成图效果较好。

+ 参考论文：

  [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial
Network](https://arxiv.org/pdf/1609.04802.pdf)

  




## 默认配置

+ 训练超参
        --input_dir=${data_path} \
        --output_dir=${output_path} \
        --batch_size=16 \
        --epochs=500 \
        --number_of_images=8000 \
        --train_test_ratio=0.8



##  支持特性

| 特效列表 | 是否支持 |
| -------- | -------- |
| 混合精度 | 是       |

### 混合精度训练

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。



## Rquirements

+ Keras 2.2.4
+ TensorFlow 1.15.0
numpy 1.10.4
matplotlib, skimage, scipy




## 模型训练与评测步骤

### For train

+ 在modelArts上训练，入口地址为 modelarts_entry_acc.py 文件。

+ 通过入口地址文件 执行 训练脚本 train_full_1p.sh。 

  ```python3.7 ./train.py \
        --input_dir=${data_path} \
        --output_dir=${output_path} \
        --batch_size=16 \
        --epochs=500 \
        --number_of_images=8000 \
        --train_test_ratio=0.8 \
        --model_save_dir='./model/'
  ```

+ 在脚本文件中，执行train代码。 训练使用预训练，训练数据集为reshape后的coco，训练为6400，测试为1600.

  

+ 训练的每个阶段的代码保存在 obs 的 model 文件夹中，格式为h5。






## GPU与NPU 精度与性能比对
- 精度结果比对

|精度指标项|GPU实测|NPU实测|
|---|---|---|
|ganloss1|0.005|0.007|
|ganloss2|0.003|0.005|
|ganloss3|2|2|
|discriminator_loss|0.3|0.35|

- 性能结果比对  

|性能指标项|GPU实测|NPU实测|
|---|---|---|
|一秒step个数|2.5it/s|2.37it/s|



## 迁移学习指导

+ 数据集准备

  + 数据集使用的是reshape后的coco数据集，位于：obs://ksgannn/dataset10000/tra/

+ 模型参数修改

  + 修改train_full_1p.sh中的内容。

+ 需要的预训练模型

  + 需要在根目录中放入./h5/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5,位于：obs://ksgannn/5.29/MA-new-05-29-17-24/code/h5/






## 训练过程

训练的参数 可以手动在train_performance_1p.sh中调整

```
 96%|█████████▋| 385/400 [02:44<00:06,  2.37it/s]
 96%|█████████▋| 386/400 [02:45<00:05,  2.34it/s]
 97%|█████████▋| 387/400 [02:45<00:05,  2.35it/s]
 97%|█████████▋| 388/400 [02:45<00:05,  2.37it/s]
 97%|█████████▋| 389/400 [02:46<00:04,  2.37it/s]
 98%|█████████▊| 390/400 [02:46<00:04,  2.35it/s]
 98%|█████████▊| 391/400 [02:47<00:03,  2.32it/s]
 98%|█████████▊| 392/400 [02:47<00:03,  2.34it/s]
 98%|█████████▊| 393/400 [02:48<00:02,  2.34it/s]
 98%|█████████▊| 394/400 [02:48<00:02,  2.36it/s]
 99%|█████████▉| 395/400 [02:48<00:02,  2.37it/s]
 99%|█████████▉| 396/400 [02:49<00:01,  2.37it/s]
 99%|█████████▉| 397/400 [02:49<00:01,  2.37it/s]
100%|█████████▉| 398/400 [02:50<00:00,  2.38it/s]
100%|█████████▉| 399/400 [02:50<00:00,  2.38it/s]
100%|██████████| 400/400 [02:51<00:00,  2.38it/s]
100%|██████████| 400/400 [02:51<00:00,  2.34it/s]
discriminator_loss : 0.283746
gan_loss : [0.007959357, 0.0059490325, 2.0103247]
--------------- Epoch 418 ---------------

  0%|          | 0/400 [00:00<?, ?it/s]
  0%|          | 1/400 [00:00<02:49,  2.36it/s]
  0%|          | 2/400 [00:00<02:48,  2.37it/s]
  1%|          | 3/400 [00:01<02:48,  2.36it/s]
  1%|          | 4/400 [00:01<02:46,  2.37it/s]


               