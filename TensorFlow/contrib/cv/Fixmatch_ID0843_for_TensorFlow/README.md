概述

FixMatch TensorFlow实现，适配Ascend平台。


* 一种通过图像增强、一致性正则化和伪标签的半监督学习算法，实现图像分类。
* Original paper: https://arxiv.org/abs/2001.07685


## 训练环境

* TensorFlow 1.15.0
* Python 3.7.0

## 代码及路径解释


```
fixmatch
└─ ...
  ├─README.md
  ├─LICENSE
  ├─test
    ├─train_full_1p.sh    npu完整训练启动脚本
    ├─train_performance_1p.sh    npu少量训练启动脚本
  ├─installdabasets.sh 下载数据集脚本
  ├─freeze_graph.py    冻结图生成pb文件
  ...
```

## 数据集
```
选择使用 cifar10/cifar100/imagenet 数据集。
```

## 快速上手

* 安装相关依赖

```bash
  sudo apt install imagemagick
  pip install -r requirements.txt
```

* 下载数据集  
  参考如下脚本（`ML_DATA` 应该指向数据集位置） :

```bash
$ sh installdabasets.sh
```

* Codebase for ImageNet experiments located in the [imagenet subdirectory](https://github.com/google-research/fixmatch/tree/master/imagenet).
* NPU 完整训练

```bash
$ export ML_DATA="path to where you want the datasets saved"
$ export PYTHONPATH=$PYTHONPATH:.
$ cd test
$ bash train_full_1p.sh --filters=32 --dataset=cifar10.5@40-1 --train_dir=./experiments/fixmatch
```

若想更改参数设置:

```bash
$ cd test
$ bash train_full_1p.sh --help
```

* 网络参数说明:
```
parameter explain:
    --filters                set filters number, default is 32
    --dataset                dataset name and some dataset setting, default is cifar10.5@40-1
    --train_dir		         model's output place, default is ./experiments/fixmatch
    -h/--help		         show help message
```

**关于 dataset 参数格式：**Available labelled sizes are 10, 20, 30, 40, 100, 250, 1000, 4000.
For validation, available sizes are 1, 5000.
Possible shuffling seeds are 1, 2, 3, 4, 5 and 0 for no shuffling (0 is not used in practiced since data requires to be shuffled for gradient descent to work properly).

* NPU 部分训练

If you want to train the model with less_step. you can use `train_performance_1p.sh` to achieve that. For example, training a FixMatch with 32 filters on cifar10 shuffled with `seed=5`, 40 labeled samples and 1 validation and only train 1 epoch :

```bash
$ cd test
$ bash train_performance_1p.sh --filters=32 --dataset=cifar10.5@40-1 --train_dir=./experiments/fixmatch --less_step=1
```


* 导出模型

You can export from a checkpoint to a standalone GraphDef file as follow:

```bash
$ python3 freeze_graph.py --dataset=cifar10.5@40-1 --CKPT_PATH=./experiments/fixmatch/.../XXX.ckpt --OUTPUT_GRAPH=./pb_albert_model/fixmatch.pb --OUTPUT_NODE_NAMES=Softmax_2
```
args:  
`--CKPT_PATH` : Select the checkpoint file place  
`--OUTPUT_GRAPH` : Select the graph file output place and name PB_file  
`--OUTPUT_NODE_NAMES`: Fill Output_node_name, default is Softmax_2

* 监视训练进程

You can point tensorboard to the training folder (by default it is `--train_dir=./experiments`) to monitor the training process:


For example:

```bash
$ tensorboard.sh --port 6007 --logdir ./experiments
```

* 查看断点精度

We compute the median accuracy of the last 20 checkpoints in the paper, this is done through this code:

```bash
# Following the previous example in which we trained cifar10.3@250-5000, extracting accuracy:
$ ./scripts/extract_accuracy.py ./experiments/fixmatch/cifar10.d.d.d.3@40-1/CTAugment_depth2_th0.80_decay0.990/FixMatch_archresnet_batch64_confidence0.95_filters32_lr0.03_nclass10_repeat4_scales3_uratio7_wd0.0005_wu1.0/

# The command above will create a stats/accuracy.json file in the model folder.
# The format is JSON so you can either see its content as a text file or process it to your liking.
```

## 精度对比
|      | GPU   | NPU   |
| ---- | ----- | ----- |
| 精度 | 86.98 | 87.09 |

we achieve the **87.09** accuarcy in dataset CIFAR10 with only 40 labeled samples. which is consistent with the conclusion of the paper and In line with the facts, GPU is also accurate (88.61 ± 3.5) 

## 训练性能

训练性能对比：

| 平台     | Epoch | 训练性能 |
|----------|---|--------------|
| NPU      | 4 |   188 sec   |
| GPU V100 | 4 |  194 sec |

The performance of model in Npu is also astonishing, The speed of training with one card on npu is about the same as the speed of training with 5 TiTAN X on Gpu. And the following table shows the performance (epoch time: s) comparison result with v100 graphics card. (Average of **4 epochs**, without **Hot Start**)
