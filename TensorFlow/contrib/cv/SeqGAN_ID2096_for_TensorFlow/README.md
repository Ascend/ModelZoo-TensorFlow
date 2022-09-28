- [基本信息](#基本信息.md)
- [概述](#概述.md)
- [训练环境准备](#训练环境准备.md)
- [快速上手](#快速上手.md)
- [迁移学习指导](#迁移学习指导.md)
- [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Object Detection**

**版本（Version）：1.1**

**修改时间（Modified） ：2022.8.11**

**大小（Size）：NULL**

**框架（Framework）：TensorFlow_1.15**

**模型格式（Model Format）：ckpt**

**精度（Precision）：FP32**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架具有策略梯度的序列生成对抗网络网络训练代码**

<h2 id="概述.md">概述</h2>

## 简述<a name="section194554031510"></a>

SeqGan具有策略梯度的序列生成对抗网络,SeqGAN的插图在真实数据和生成的数据上进行训练。通过策略梯度训练，其中最终奖励信号由插图提供，并通过蒙特卡洛搜索传递回中间动作值,实验分为两个阶段。在第一阶段，使用预言机模型和最大似然估计提供的正数据进行监督学习。在第二阶段，使用对抗训练来改进生成器。

- 参考论文：
  
  [http://arxiv.org/abs/1609.05473](SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient)

- 参考实现：

  https://github.com/LantaoYu/SeqGAN

- 适配昇腾 AI 处理器的实现：
  
  https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/SeqGAN_ID2096_for_TensorFlow

- 通过Git获取对应commit\_id的代码方法如下：
  
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    

## 默认配置<a name="section91661242121611"></a>

-   训练超参（单卡）：
    - Batch size: 64
    - TOTAL_BATCH = 200
    - positive_file = 'save/real_data.txt'
    - negative_file = 'save/generator_sample.txt'
    - eval_file = 'save/eval_file.txt'
    - generated_num = 10000


## 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 是       |
| 混合精度   | 否      |
| 数据并行   | 是       |


## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

拉起脚本中，

```
 ./train_full_1p.sh --help

parameter explain:
    --precision_mode         #precision mode(allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision)
    --data_path              # dataset of training
    --output_path            # output of training
    --train_steps            # max_step for training
    --train_epochs           # max_epoch for training
    --batch_size             # batch size
    -h/--help                show help message
```

混合精度相关代码示例:

 ```
    precision_mode="allow_mix_precision"

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

1、数据集见训练脚本代码。路径/save/target_params.pkl

2、SeqGAN训练的模型及数据集可以参考"简述 -> 参考实现"


## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练。

    - 启动训练之前，首先要配置程序运行相关环境变量。

      环境变量配置信息参见：

      [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)

    - 单卡训练


              1. 配置训练参数。
            
                 在`sequence_gan.py`中，配置checkpoint保存路径，请用户根据实际路径配置，参数如下所示：
            
                 ```
                 TOTAL_BATCH = 200
                 positive_file = 'save/real_data.txt'
                 negative_file = 'save/generator_sample.txt'
                 eval_file = 'save/eval_file.txt'
                 generated_num = 10000

                 ```
            
              2. 启动训练。
            
                 ```
                 python3 sequence_gan.py 

                 ```
            


<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码

```
SeqGAN_ID2096_for_TensorFlow 
└─
  ├─README.md
  ├─LICENSE  
  ├─save
  |     |--target_params.pkl   训练需要的数据集
  ├─dataloader.py     训练数据工具类 
  ├─discriminator.py    
  ├─modelarts_entry_acc.py   
  ├─rollout.py     
  ├─sequence_gan.py     模型训练程序入口
  ├─target_lstm.py
  ├─test
  |     |--train_full_1p.sh  单卡full shell
```

## 脚本参数<a name="section6669162441511"></a>

```
--TOTAL_BATCH = 200
--positive_file = 'save/real_data.txt'
--negative_file = 'save/generator_sample.txt'
--eval_file = 'save/eval_file.txt'
--generated_num = 10000
--EMB_DIM = 32 # embedding dimension
--HIDDEN_DIM = 32 # hidden state dimension of lstm cell
--SEQ_LENGTH = 20 # sequence length
--START_TOKEN = 0
--PRE_EPOCH_NUM = 120 # supervise (maximum likelihood estimation) epochs
--SEED = 88
--BATCH_SIZE = 64

```

## 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡，8卡网络训练。模型存储路径为${cur_path}/output/$ASCEND_DEVICE_ID，包括训练的log以及checkpoints文件。以1卡训练为例，loss信息在文件${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。