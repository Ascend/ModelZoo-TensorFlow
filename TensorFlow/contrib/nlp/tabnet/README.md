## 基本信息

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：NLP** 

**版本（Version）：1.2**

**修改时间（Modified） ：2022.11.14**

**大小（Size）：66KB**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Contrib**

**描述（Description）：基于TensorFlow框架的tabnet表格处理神经网络训练代码 ** 

## 概述

Papåer: https://arxiv.org/abs/1908.07442

This directory contains an example implementation of TabNet on the Forest Covertype dataset (https://archive.ics.uci.edu/ml/datasets/covertype).

First, run `python -m download_prepare_covertype.py` to download and prepare the Forest Covertype dataset. This command creates train.csv, val.csv and test.csv files under the "data/" directory.

To run the pipeline for training and evaluation, simply use `python -m experiment_covertype.py`. For debugging in a low-resource environment, you can use `python -m test_experiment_covertype.py`.

To modify the experiment to other tabular datasets:

- Substitute the train.csv, val.csv, and test.csv files under "data/" directory,
- Modify the data_helper function with the numerical and categorical features of the new dataset,
- Reoptimize the TabNet hyperparameters for the new dataset.

## Requirements

- Tensorflow 1.15.0
- absl-py >= 0.5.0
- numpy \=\= 1.15.1
- wget >\= 3.2
- Ascend910

## 模型训练

### 脚本和示例代码

```
tabnet
├── check_result.tf.json
├── data_helper_covertype.py
├── download_prepare_covertype.py
├── experiment_covertype.py
├── fusion_result.json
├── fusion_switch.cfg
├── myTest.py
├── requirements.txt
├── run.sh
├── tabnet README.md
├── tabnet_model.py
└── test_experiment_covertype.py
```

### 脚本参数

```
- TRAIN_FILE = "data/train_covertype.csv"
- VAL_FILE = "data/val_covertype.csv"
- TEST_FILE = "data/test_covertype.csv"
- MAX_STEPS = 10
- DISPLAY_STEP = 5
- VAL_STEP = 5
- SAVE_STEP = 40000
- INIT_LEARNING_RATE = 0.02
- DECAY_EVERY = 500
- DECAY_RATE = 0.95
- BATCH_SIZE = 32
- SPARSITY_LOSS_WEIGHT = 0.0001
- GRADIENT_THRESH = 2000.0
- SEED = 1
```

### 训练环境

- 华为NPU裸机

### 训练过程

- 首先运行 `python -m download_prepare_covertype.py` 下载数据集
- 基于[原始代码](https://github.com/google-research/google-research/tree/master/tabnet)，参考[TensorFlow 1.15网络模型迁移和训练](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/600alpha001/moddevg/tfmigr1/atlasmprtg_13_0009.html)官方文档，将其中代码适配入NPU
- 全量运行参数过大，目前仅在GPU和NPU执行了 `python -m test_experiment_covertype.py` ，同时对比了精度和性能

## 结果对比

### GPU训练结果

```
Step 1 , Step Training Time = 15.6996
Step 2 , Step Training Time = 0.0656
Step 3 , Step Training Time = 0.0643
Step 4 , Step Training Time = 0.0658
Step 5 , Training Loss = 2.1249
Step 5 , Step Training Time = 42.3971
Step 5 , Val Accuracy = 0.3649
Step 6 , Step Training Time = 0.0686
Step 7 , Step Training Time = 0.0667
Step 8 , Step Training Time = 0.0663
Step 9 , Step Training Time = 0.0665
Step 10 , Training Loss = 1.9217
Step 10 , Step Training Time = 16.2367
Step 10 , Val Accuracy = 0.3444
```

### NPU训练结果

```
Step 1 , Step Training Time = 931.2861
Step 2 , Step Training Time = 0.3418
Step 3 , Step Training Time = 0.3229
Step 4 , Step Training Time = 0.3138
Step 5 , Training Loss = 1.7622
Step 5 , Step Training Time = 253.5153
Step 5 , Val Accuracy = 0.2752
Step 6 , Step Training Time = 116.0940
Step 7 , Step Training Time = 0.4088
Step 8 , Step Training Time = 0.3603
Step 9 , Step Training Time = 0.3069
Step 10 , Training Loss = 1.7022
Step 10 , Step Training Time = 196.8820
Step 10 , Val Accuracy = 0.3621
```

### 精度与性能分析

- 不考虑NPU train和eval的首次编图耗时，目前NPU的训练耗时大约是GPU的5倍，最终精度有所提升











