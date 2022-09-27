## 基本信息

**发布者（Publisher）：contrib**

**应用领域（Application Domain）： Machine Translation**

**版本（Version）：1.0**

**修改时间（Modified） ：2021.01.11**

**大小（Size）：1.5GB**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Research**

**描述（Description）：基于TensorFlow框架的Seq2Seq机器翻译网络训练代码** 

## 概述

Seq2Seq是一种循环神经网络的变种，包括编码器(Encoder)和解码器(Decoder)两部分。Seq2Seq是自然语言处理中的一种重要模型，可以用于机器翻译、对话系统、自动文摘。

Seq2Seq是一种重要的RNN模型，也称为Encoder-Decoder模型，可以理解为一种N×M的模型。模型包含两个部分：Encoder用于编码序列的信息，将任意长度的序列信息编码到一个向量c里。而Decoder是解码器，解码器得到上下文信息向量c之后可以将信息解码，并输出为序列。

- 参考论文：

  [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215v3)

- 参考实现：

  https://github.com/tensorflow/models/tree/r1.6.0

- 适配昇腾AI处理器的实现：

  [https://gitee.com/ascend/modelzoo/tree/master/contrib/TensorFlow/Research/nlp/seq2seq/Seq2Seq_ID1474_for_TensorFlow](https://gitee.com/ascend/modelzoo/tree/master/contrib/TensorFlow/Research/nlp/seq2seq/Seq2Seq_ID1474_for_TensorFlow)

## 默认配置

训练超参

- use_lstm:True
- learning_rate:0.7
- learning_rate_decay_factor:0.5
- max_gradient_norm:5.0
- batch_size:128
- size:1000
- num_layers:4
- from_vocab_size:160000
- to_vocab_size:80000
- max_train_data_size:12000000
- use_rev_sou_sen:False
- num_samples:512

- steps_per_checkpoint:200
- use_fp16:False
- _buckets:[(5, 10), (10, 15), (20, 25), (40, 50)]
- epoch:7.5

## 支持特性

| 特性列表 | 是否支持 |
| -------- | -------- |
| 混合精度 | 是       |

## 混合精度训练

昇腾910AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度

脚本已默认开启混合精度，在translate.py脚本中设置。

```python
custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
```

## 训练环境准备

1. 硬件环境准备请参见各硬件产品文档"[驱动和固件安装升级指南]( https://support.huawei.com/enterprise/zh/category/ai-computing-platform-pid-1557196528909)"。需要在硬件设备上安装与CANN版本配套的固件与驱动。
2. 软件环境：
   - CANN版本：5.0.4.alpha002
   - Tensorflow版本：1.15.0
   - Python版本：3.7.5
   - nltk

## 快速上手

我们使用了 WMT'14 英语到法语数据集。

请自行准备数据集，包括训练集和验证集。

- 训练集：training-giga-fren.tar
- 验证集：dev-v2.tgz

将数据集文件放入项目的data目录中，在训练脚本中检查数据集路径，可正常使用。

## 模型训练

- 单击“立即下载”，并选择合适的下载方式下载源码包。

- 启动训练之前，首先要配置程序运行相关环境变量。

  环境变量配置信息参见：

     [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)

- 训练

  1. 配置训练参数

     首先在脚本translate.py中，配置训练和验证数据集路径，请用户根据实际路径配置，数据集参数如下所示：

     ```python
     tf.app.flags.DEFINE_string("data_dir", "./data", "Data directory")
     ```

  2. 启动训练

     执行run_1p.sh脚本。

     ```shell
     sh run_1p.sh
     ```

- 验证

  1. 测试的时候，需要修改translate.py脚本中的参数，配置训练完成的模型文件所在路径，请用户根据实际路径进行修改。

     ```python
     tf.app.flags.DEFINE_string("train_dir", "./model", "Training directory.")
     ```

  2. 启动测试

     执行eval.sh脚本。

     ```shell
     sh eval.sh
     ```

## 高级参考

### 脚本和示例代码

```
Seq2Seq_ID1474_for_TensorFlow
│  compute_bleu.py			bleu计算
│  data_utils.py			原始数据处理
│  eval.sh					执行验证脚本
│  LICENSE					证书
│  modelzoo_level.txt		模型状态文件
│  requirements.txt			所需库文件
│  run_1p.sh				执行训练脚本
│  README.md				
│  seq2seq_model.py			seq2seq模型文件
│  translate.py				训练和验证脚本
├─data						存放原始数据文件
├─model						存放ckpt模型文件
├─override_contrib			seq2seq搭建模型所需脚本文件
│      core_rnn_cell.py
└─     override_seq2seq.py
```

### 脚本参数

```
- use_lstm:True							是否使用LSTM网络
- learning_rate:0.7						学习率
- learning_rate_decay_factor:0.5		学习率衰减指数
- max_gradient_norm:5.0					最大梯度裁剪范数				
- batch_size:128						批量大小
- size:1000								每个模型层的大小
- num_layers:4							模型层数
- from_vocab_size:160000				英语词表大小
- to_vocab_size:80000					法语词表大小
- max_train_data_size:12000000			训练数据大小限制（0：无限制）
- use_rev_sou_sen:False					如果为真，使用倒置源语句的训练集
- num_samples:512						采样softmax的样本数
- steps_per_checkpoint:200				每个检查点要执行多少训练步骤
- use_fp16:False						使用fp16而不是fp32进行训练
- _buckets:[(5, 10), (10, 15), (20, 25), (40, 50)]															预定义的桶，为了override_seq2seq.model_with_buckets()方法使用
- epoch:7.5								训练轮次
```

### 训练过程

1.  通过“模型训练”中的训练指令启动训练。
2.  参考脚本的模型存储路径为model，训练脚本log中包括如下信息。

```
global step 424600 learning rate 0.7000 step-time 0.85 perplexity 2.40
  eval: bucket 0 perplexity 3.58
  eval: bucket 1 perplexity 3.22
  eval: bucket 2 perplexity 2.77
  eval: bucket 3 perplexity 3.27
```

### 推理/验证过程

1.  通过“模型训练”中的测试指令启动测试。
2.  当前只能针对该工程训练出的checkpoint进行推理测试。
3.  推理脚本的参数train_dir可以配置为checkpoint所在的文件夹路径。
4.  测试脚本log中包括如下信息。

```
The 2995 sentence is calculated
The 2996 sentence is calculated
The 2997 sentence is calculated
The 2998 sentence is calculated
The 2999 sentence is calculated
The 3000 sentence is calculated
The average score of Bleu :  0.14748751130748788
The average score of Bleu*100 :  14.748751130748788
```

## 结果对比

### 预训练模型

- GPU：[Link(op1z)](https://pan.baidu.com/s/1yCG7fLZwREa8xhf-DdJV2w)

- NPU：[Link(dik8)](https://pan.baidu.com/s/1HCYU2SemTg8Lho7hNul-JQ)

### 困惑度（Perplexity）

GPU

seq2seq_train_perplexity

![seq2seq_train_perplexity](https://images.gitee.com/uploads/images/2021/0928/103712_21b3d5b8_5559452.jpeg "seq2seq_train_perplexity.jpg")

seq2seq_eval_perplexity

![seq2seq_eval_perplexity](https://images.gitee.com/uploads/images/2021/0928/103748_2e443d5a_5559452.jpeg "seq2seq_eval_perplexity.jpg")

NPU

seq2seq_npu_train_perplexity

![seq2seq_npu_train_perplexity](https://images.gitee.com/uploads/images/2021/1221/213023_53e1f159_5559452.jpeg)

seq2seq_npu_eval_perplexity

![seq2seq_npu_eval_perplexity](https://images.gitee.com/uploads/images/2021/1221/213159_5dcf19c3_5559452.jpeg)

### 精度（BLEU分数）

论文：34.81%

GPU：15.07%

NPU（未开混合精度）：14.74%

NPU（开启混合精度）：11.41%

### 性能

GPU：2.53 s/step

NPU：0.83 s/step