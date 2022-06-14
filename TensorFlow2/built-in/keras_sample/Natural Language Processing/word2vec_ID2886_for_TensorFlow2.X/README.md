- [基本信息](#基本信息.md)
- [概述](#概述.md)
- [训练环境准备](#训练环境准备.md)
- [快速上手](#快速上手.md)
- [迁移学习指导](#迁移学习指导.md)
- [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Natural Language Processing**

**版本（Version）：1.1**

**修改时间（Modified） ：2022.04.11**

**大小（Size）：64.2KB**

**框架（Framework）：TensorFlow_2.6.2**

**模型格式（Model Format）：h5**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Research**

**描述（Description）：基于skip-gram的词向量训练代码**

<h2 id="概述.md">概述</h2>

## 简述
    基于skip-gram的词向量训练代码。


- 参考论文：

  [Distributed representations of words and phrases and their compositionality](https://arxiv.org/abs/1310.4546)

- 参考实现：

  [https://github.com/Deermini/word2vec-tensorflow](https://github.com/Deermini/word2vec-tensorflow)

- 适配昇腾 AI 处理器的实现：skip
    
    [https://gitee.com/jelly_111/research_tf2/tree/master/word2vec_ID2886_for_TensorFlow2.X](https://gitee.com/jelly_111/research_tf2/tree/master/word2vec_ID2886_for_TensorFlow2.X)

- 通过Git获取对应commit\_id的代码方法如下：
    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    ```

## 默认配置<a name="section91661242121611"></a>
- 训练超参（单卡）：

  Batch Size = 128

  train steps = 3000000

  learning rate = 0.001



## 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 否       |
| 混合精度   | 是       |
| 数据并行   | 否       |

## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>
拉起脚本中，传入--precision_mode='allow_mix_precision'

```
./train_performance_1p.sh --help
parameter explain:
    --precision_mode         precision mode(allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision)
    --over_dump                  if or not over detection, default is False
    --data_dump_flag         data dump flag, default is False
    --data_dump_step             data dump step, default is 10
    --profiling                  if or not profiling for performance debug, default is False
    --data_path                  source data of training
    -h/--help                    show help message
```

对应代码：

```
  npu_device.global_options().precision_mode='allow_mix_precision'
  npu_device.open().as_default()
```

<h2 id="训练环境准备.md">训练环境准备</h2>

-  硬件环境和运行环境准备请参见《[CANN软件安装指南](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=installation-update)》
-  运行以下命令安装依赖。
```
pip3 install requirements.txt
```
说明：依赖配置文件requirements.txt文件位于模型的根目录

<h2 id="快速上手.md">快速上手</h2>

## 数据集准备<a name="section361114841316"></a>

1. 请用户自行下载cnews数据集

## 模型训练<a name="section715881518135"></a>
- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练。
  
  1. 启动训练之前，首先要配置程序运行相关环境变量。

    环境变量配置信息参见：

    [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

  2. 单卡训练
       
    2.1 单卡训练指令（脚本位于word2vec_ID2886_for_TensorFlow2.X/test/）,请确保下面例子中的“--data_path”修改为用户的数据集的路径。
    
      bash test/train_performance_1p.sh --data_path=/home/cnews

<h2 id="迁移学习指导.md">高级参考</h2>

- 数据集准备。

    1.  获取数据。
        请参见“快速上手”中的数据集准备。
    
- 模型训练。

    参考“模型训练”中训练步骤。

<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码
```
word2vec_ID2886_for_TensorFlow2.X
|-- LICENSE
|-- README.md
|-- requirements.txt
|-- stop_words.txt
|-- modelzoo_level.txt
|-- word2vec_chinese.py
|-- test
|   |-- train_full_1p.sh
|   |-- train_performance_1p.sh

```

## 脚本参数<a name="section6669162441511"></a>

```
--data_dir							The location of the input data.
--ckpt_path							The location of the model checkpoint files
--batch_size						Batch size for training and evaluation
--train_step						train step num
```

## 训练过程<a name="section1589455252218"></a>

1. 通过“模型训练”中的训练指令启动单卡训练。
2. 将训练脚本（train_full_1p.sh）中的data_path设置为训练数据集的路径。具体的流程参见“模型训练”的示例。
3. 模型存储路径为“curpath/output/ASCEND_DEVICE_ID”，包括训练的log文件。
4. 以多卡训练为例，loss信息在文件curpath/output/{ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。

## 推理/验证过程<a name="section1465595372416"></a>

```
 NA

```




