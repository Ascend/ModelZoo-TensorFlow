# 基本信息

**发布者（Publisher）：** **Huawei**

**应用领域（Application Domain）：** **image denoising**

**版本（Version）：1.0**

**修改时间（Modified） ：** **2021.11.17**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：** **Demo**

**描述（Description）：基于TensorFlow框架的图像去噪网络训练代码**

# 概述

- 模型功能：

"We apply basic statistical reasoning to signal reconstruction by machine learning -- learning to map corrupted observations to clean signals -- with a simple and powerful conclusion: it is possible to learn to restore images by only looking at corrupted examples, at performance at and sometimes exceeding training using clean data, without explicit image priors or likelihood models of the corruption. In practice, we show that a single model learns photographic noise removal, denoising synthetic Monte Carlo images, and reconstruction of undersampled MRI scans -- all corrupted by different processes -- based on noisy data only." 

模型网络利用有噪声的图像进行学习，从而将有噪声的图像转化为无噪声的clean图像。

![n2nteaser_1024width](https://gitee.com/DatalyOne/picGo-image/raw/master/202110081800288.png)

- 模型来源：Github](https://github.com/NVlabs/noise2noise)

- 参考论文： [Paper (arXiv)](https://arxiv.org/abs/1803.04189)

- 适配昇腾 AI 处理器的实现

​		https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/NOISE2NOISE_ID0800_for_TensorFlow

- 获取适配代码：

​	   通过Git获取对应commit_id的代码方法如下：

```
git clone {repository_url}    # 克隆仓库的代码
cd {repository_name}    # 切换到模型的代码仓目录
git checkout  {branch}    # 切换到对应分支
git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
```

# 支持特性

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 是       |
| 混合精度   | 是       |

# 混合精度训练

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

- 开启混合精度

```
--precision-mode		算子精度模式:allow_fp32_to_fp16/force_fp16/allow_mix_precision
```

# 训练环境准备

1. 硬件环境准备请参见各硬件产品文档"[驱动和固件安装升级指南](https://support.huawei.com/enterprise/zh/category/ai-computing-platform-pid-1557196528909)"。需要在硬件设备上安装与CANN版本配套的固件与驱动。

2. 宿主机上需要安装Docker并登录[Ascend Hub中心](https://ascendhub.huawei.com/#/detail?name=ascend-tensorflow-arm)获取镜像。

   当前模型支持的镜像列表如[表1](https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/NOISE2NOISE_ID0800_for_TensorFlow#zh-cn_topic_0000001074498056_table1519011227314)所示。

   **表 1** 镜像列表

   | *镜像名称*                                                   | *镜像版本* | *配套CANN版本*                                               |
   | ------------------------------------------------------------ | ---------- | ------------------------------------------------------------ |
   | *ARM架构：[ascend-tensorflow-arm](https://ascend.huawei.com/ascendhub/#/detail?name=ascend-tensorflow-arm)**x86架构：[ascend-tensorflow-x86](https://ascend.huawei.com/ascendhub/#/detail?name=ascend-tensorflow-x86)* | *20.2.0*   | *[20.2](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software)* |

# 训练脚本

**必须将数据集文件放置在与test文件夹同级的datasets文件夹中**

- 启动训练之前，首先要配置程序运行相关环境变量。

  环境变量配置信息参见：

  [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend 910训练平台环境变量设置?sort_id=3148819)

- 单卡训练

​		启动训练 （脚本为test/train_full_1p.sh或test/train_full_1p_mri.sh）

```
bash train_full_1p.sh    (train_full_1p_mri.sh)
```

- 8卡训练

​		启动训练 （脚本为test/train_full_8p.sh或test/train_full_8p_mri.sh）

```
bash train_full_8p.sh 	(train_full_8p_mri.sh)
```

部分参数设置请参考下文描述

# Noise2Noise 网络

## 默认配置

- 训练数据集预处理（以ImageNet2012 ILSVRC2012_img_val 数据集集为例，仅作为用户参考示例）：
  - 图像的输入尺寸为256*256
  - 图像输入格式：TFRecord
  - 图像通道设为channel_first
  - 图像归一化
- 测试数据集预处理（以kodak验证集为例，仅作为用户参考示例）
  - 图像的输入尺寸为512\*768或者768\*512
  - 图像输入格式：png
- 部分训练超参
  - Batch size: 4
  - Learning rate(LR): 0.0003
  - Iteration_count:500000
  - Eval interval:1000
  - Ramp down perc:0.3

## 数据集准备

1、用户自行准备好数据集，包括训练数据集和验证数据集。

2、数据集的处理可以参考"模型来源"

3、**将数据集放置在datasets文件夹下**

## 训练

在 ImageNet 上训练 noise2noise autoencoder

```
# try python config.py train --help for available options
python3.7.5 config.py --desc='-test' --graph-run-mode=1 --op-select-implmode=high_precision --precision-mode=allow_mix_precision train --train-tfrecords=datasets/imagenet_val_raw.tfrecords --long-train=true --noise=gaussian --hcom-parallel=False --is-distributed=False --is-loss-scale=True
```

部分参数解释：

```
--graph-run-mode    	图执行模式：0：在线推理场景，1：训练场景
--op-select-implmode	算子高精度模式：high_precision 高性能模型：high_performance 
--precision-mode		算子精度模式:allow_fp32_to_fp16/force_fp16/allow_mix_precision

--hcom-parallel			是否启用Allreduce梯度更新和前后向并行执行 
--is-distributed		是否为分布式训练
--is-loss-scale			是否启用LossScale
--long-train			False：用于测试
```

使用TensorBoard查看训练情况：

```
cd results
tensorboard --logdir .
```

训练完成后，会在results/目录下生成相应的文件夹（如00000-autoencoder-n2n）。该文件夹内包含network_final.pickle，用于保存模型结果

## 推理

```
python3.7.5 config.py --graph-run-mode=0 --op-select-implmode=high_precision --precision-mode=allow_mix_precision infer-image --network-snapshot=<path>/network_final.pickle --image=<path>/test.png --out=<path>/test_out.png
```

生成图片进行展示

## 评估

在kodak验证集上评估

```
python3.7.5 config.py --graph-run-mode=0 --op-select-implmode=high_precision --precision-mode=allow_mix_precision validate --dataset-dir=datasets/kodak --network-snapshot=<path>/network_final.pickle
```

## 性能

| Tesla V100-SXM2 | Ascend 910 |
| --------------- | ---------- |
| 31ms/step       | 23ms/step  |

注：minibatch size 为 4

## 精度

**论文复现命令**：

| Noise    | Noise2Noise | Command line                                                 |
| -------- | ----------- | ------------------------------------------------------------ |
| Gaussian | Yes         | python config.py --graph-run-mode=1 --op-select-implmode=high_precision --precision-mode=allow_mix_precision train --noise=gaussian --noise2noise=true --long-train=true --train-tfrecords=datasets/imagenet_val_raw.tfrecords --hcom-parallel=False --is-distributed=False --is-loss-scale=True |
| Gaussian | No          | python config.py --graph-run-mode=1 --op-select-implmode=high_precision --precision-mode=allow_mix_precision train --noise=gaussian --noise2noise=false --long-train=true --train-tfrecords=datasets/imagenet_val_raw.tfrecords --hcom-parallel=False --is-distributed=False --is-loss-scale=True |
| Poisson  | Yes         | python config.py --graph-run-mode=1 --op-select-implmode=high_precision --precision-mode=allow_mix_precision train --noise=poisson --noise2noise=true --long-train=true --train-tfrecords=datasets/imagenet_val_raw.tfrecords --hcom-parallel=False --is-distributed=False --is-loss-scale=True |
| Poisson  | No          | python config.py --graph-run-mode=1 --op-select-implmode=high_precision --precision-mode=allow_mix_precision train --noise=poisson --noise2noise=false --long-train=true --train-tfrecords=datasets/imagenet_val_raw.tfrecords --hcom-parallel=False --is-distributed=False --is-loss-scale=True |

**模型结果验证：**

| Noise    | Dataset | Command line                                                 | Expected PSNR (dB)        | NPU PSNR (dB)              |
| -------- | ------- | ------------------------------------------------------------ | ------------------------- | -------------------------- |
| Gaussian | kodak   | python3.7.5 config.py --graph-run-mode=0 --op-select-implmode=high_precision --precision-mode=allow_mix_precision validate --dataset-dir=datasets/kodak --noise=gaussian --network-snapshot=<.../network_final.pickle> | 32.38 (n2c) / 32.39 (n2n) | 32.40 (n2c) / 32.38 (n2n)  |
| Gaussian | bsd300  | python config.py validate --graph-run-mode=0 --op-select-implmode=high_precision --precision-mode=allow_mix_precision --dataset-dir=datasets/bsd300 --noise=gaussian --network-snapshot=<.../network_final.pickle> | 31.01 (n2c) / 31.02 (n2n) | 31.02 (n2c) / 31.01 (n2n)  |
| Poisson  | kodak   | python config.py validate --graph-run-mode=0 --op-select-implmode=high_precision --precision-mode=allow_mix_precision --dataset-dir=datasets/kodak --noise=poisson --network-snapshot=<.../network_final.pickle> | 31.66 (n2c) / 31.66 (n2n) | 31.62 (n2c) /  31.60 (n2n) |
| Poisson  | bsd300  | python config.py validate --graph-run-mode=0 --op-select-implmode=high_precision --precision-mode=allow_mix_precision --dataset-dir=datasets/bsd300 --noise=poisson --network-snapshot=<.../network_final.pickle> | 30.27 (n2c) / 30.26 (n2n) | 30.25 (n2c) /  30.23 (n2n) |

## 模型固化

```
python3.7.5 ckpt2pb.py --ckptdir=model/ckpt_npu/model.ckpt --pbdir=model/pb --input-node-name=input --output-node-name=output --pb-name=test.pb --width=768 --height=512
```

参数解释：

```
--ckptdir 			模型路径 （改为训练后生成的checkpoit文件位置）
--pbdir     		转换后的pb模型的保存路径
--pb_name			pb模型名字 默认：test
--input-node-name 	输入节点 默认：input
--output-node-names	输出节点 默认：output
--width				与训练输入的input shape保持一致
--heigh
```

## pb模型评估

```
python3.7.5 config.py --graph-run-mode=0 --op-select-implmode=high_precision --precision-mode=allow_mix_precision validate-pb --dataset-dir=datasets/kodak --pbdir=model/pb/test.pb --width=768 --height=512 --input-tensor-name=input:0 --output-tensor-name=output:0 --noise=gaussian
```

注：输入的图片会被自动填充成符合--width，--height形状的图片，因此评估结果可能会与用原图片进行评估得到的结果存在差异

## pb模型推理

```
python3.7.5 config.py --graph-run-mode=0 --op-select-implmode=high_precision --precision-mode=allow_mix_precision infer-image-pb --image=<path>/test.png --out=<path>/test_out.png  --pbdir=model/pb/test.pb --width=768 --height=512 --input-tensor-name=input:0 --output-tensor-name=output:0
```

# Noise2Noise MRI 网络

## 默认配置

- 训练数据集预处理（以IXI-T1 数据集为例，仅作为用户参考示例）：
  - 图像的输入尺寸为256*256
  - 图像输入格式：pkl
  - 图像归一化
- 测试数据集预处理（以IXI-T1 数据集为例，仅作为用户参考示例）
  - 图像的输入尺寸为256*256
  - 图像输入格式：pkl
  - 图像归一化
- 部分训练超参
  - Batch size:16
  - Max learning rate(LR): 0.001
  - Epoch: 300

## 数据集准备

1、用户自行准备好数据集，包括训练数据集和验证数据集。

2、数据集的处理可以参考"模型来源"

3、**将数据集放置在datasets文件夹中**

## 训练

```
python3.7.5 config_mri.py --graph-run-mode=1 --op-select-implmode=high_precision --precision-mode=allow_mix_precision --epoch=300 --is-distributed=False --is-loss-scale=True --hcom-parallel=False 
```

训练成功后，输出以下内容:

```
dnnlib: Running train_mri.train() on localhost...
Loading training set.
Loading dataset from datasets\ixi_train.pkl
<...long log omitted...>
Epoch 297/300: time=107.981, train_loss=0.0126064, test_db_clamped=31.72174, lr=0.000002
Epoch 298/300: time=107.477, train_loss=0.0125972, test_db_clamped=31.73622, lr=0.000001
Epoch 299/300: time=106.927, train_loss=0.0126012, test_db_clamped=31.74232, lr=0.000001
Saving final network weights.
Resetting random seed and saving a bunch of example images.
dnnlib: Finished train_mri.train() in 8h 59m 19s.
```

The expected average PSNR on the validation set (named `test_db_clamped` in code) is roughly 31.74 dB.目标精度：31.74 dB

Noise-to-noise training is enabled by default for the MRI case.  To use noise-to-clean training, edit `config_mri.py` and change `corrupt_targets=True` to `corrupt_targets=False`. 默认使用加噪声的图片作为target。

## 性能

(1) 由于 Noise2Noise MRI 网络中存在tf.complex64数据的处理，NPU暂不支持改数据格式，因此注释掉config_mri.py中的

```
train.update(post_op='fspec')
```

注释掉如上命令后，可在NPU进行训练

| Tesla V100-SXM2 | Ascend 910  |
| :-------------- | ----------- |
| 82.8s/epoch     | 54.5s/epoch |

(2)  在验证时，将相关TF代码替换为NumPy代码。开启 train.update(post_op='fspec')

| Tesla V100-SXM2 | Ascend 910  |
| --------------- | ----------- |
| 87.2s/epoch     | 60.0s/epoch |

注：batch size 为 16

## 精度

(1) 注释 train.update(post_op='fspec')

| Tesla V100-SXM2 | Ascend 910 |
| --------------- | ---------- |
| 29.5 dB         | 29.5 dB    |

(2) 开启 train.update(post_op='fspec')

| Tesla V100-SXM2 | Ascend 910 |
| --------------- | ---------- |
| 31.74 dB        | 31.37 dB   |

注：由于代码改动，精度结果可能存在差异。

## 模型固化

```
python3.7.5 ckpt2pb_mri.py --ckptdir=model/ckpt_npu/model.ckpt --pbdir=model/pb --input-node-name=input --output-node-name=output --pb-name=test_mri.pb
```

## pb模型评估

```
python3.7.5 config_mri_pb.py --graph-run-mode=0 --op-select-implmode=high_precision --precision-mode=allow_mix_precision validate --data-dir=datasets/ixi_valid.pkl --pbdir=model/pb/test_mri.pb --input-tensor-name=input:0 --output-tensor-name=output:0 --post-op=fspec
```

# om离线推理

详情请看**NOISE2NOISE_ID800_for_ACL**  README.md

