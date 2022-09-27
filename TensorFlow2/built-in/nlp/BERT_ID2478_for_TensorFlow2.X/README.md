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

**修改时间（Modified） ：2022.6.11**

**大小（Size）：44KB**

**框架（Framework）：TensorFlow_2.6.2**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的BertLarge自然语言处理网络训练代码**

<h2 id="概述.md">概述</h2>

## 简述<a name="section194554031510"></a>

BERT模型的全称是：BidirectionalEncoder Representations from Transformer。BERT模型的目标是利用大规模无标注语料训练、获得文本的包含丰富语义信息的Representation，即：文本的语义表示，然后将文本的语义表示在特定NLP任务中作微调，最终应用于该NLP任务。

- 参考论文：

  [https://arxiv.org/abs/1810.04805](https://gitee.com/link?target=https%3A%2F%2Farxiv.org%2Fabs%2F1810.04805)

- 参考实现：

  https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/LanguageModeling/BERT

- 适配昇腾 AI 处理器的实现：
  
  https://gitee.com/ascend/ModelZoo-TensorFlow/blob/master/TensorFlow2/built-in/nlp/BERT_ID2478_for_TensorFlow2.X

- 通过Git获取对应commit\_id的代码方法如下：
  
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    

## 默认配置<a name="section91661242121611"></a>

-   网络结构
    - 24-layer, 1024-hidden, 16-heads, 340M parameters
-   训练超参（单卡）：
    - Batch size: 24
    - max_predictions_per_seq: 76
    - max_seq_length: 512
    - Learning rate(LR): 0.000058711
    - Weight decay: 0.01
    - beta_1: 0.91063
    - beta_2: 0.96497
    - Train epoch: 32


## 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 是       |
| 混合精度   | 是       |
| 数据并行   | 是       |


## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

拉起脚本中，传入--precision_mode='allow_mix_precision'

```
 ./train_performance_1p_16bs.sh --help

parameter explain:
    --precision_mode         precision mode(allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision)
    --over_dump                  if or not over detection, default is False
    --data_dump_flag         data dump flag, default is False
    --data_dump_step             data dump step, default is 10
    --profiling                  if or not profiling for performance debug, default is False
    --data_path                  source data of training
    -h/--help                    show help message
```

相关代码示例:

```
flags.DEFINE_string(name='precision_mode', default= 'allow_fp32_to_fp16',
                    help='allow_fp32_to_fp16/force_fp16/ ' 
                    'must_keep_origin_dtype/allow_mix_precision.')

npu_device.global_options().precision_mode=FLAGS.precision_mode
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

1、用户自行准备好数据集，本网络包括Bert的Pre-training和Fine tuning任务

2、Pre-training任务使用的数据集是wikipedia，Fine tuning使用的数据集是SQuAD 1.1 和 SQuAD 2.0

3、Bert训练的模型及数据集可以参考"简述 -> 参考实现"



## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练。

    - 启动训练之前，首先要配置程序运行相关环境变量。

      环境变量配置信息参见：

      [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)

    - 单卡训练

         网络共包含10个训练，其中Pre-training 6个训练任务， Fine tuning4个训练任务。

         **Pre-training 任务**:

         1. Lamb优化器

            - phase1

              ```shell
              # bertbase模型
              bash train_performance_bertbase_8p_1024bs_lamb_phase1.sh --data_path=/home
              
              # bertlarge模型
              bash train_performance_bertlarge_8p_512bs_lamb_phase1.sh --data_path=/home
              ```

            - phase2

              ```shell
              # bertbase模型
              bash train_performance_bertbase_8p_192bs_lamb_phase2.sh --data_path=/home
              
              # bertlarge模型
              bash train_performance_bertlarge_8p_96bs_lamb_phase2.sh --data_path=/home
              ```

              

         2. Adam优化器

         ```
         # bertbase模型
         bash train_performance_bertbase_8p_192bs_adam.sh --data_path=/home
         
         # bertlarge模型
         bash train_performance_bertlarge_8p_128bs_adam.sh --data_path=/home
         ```

         **Fine tuning**

         - SQuAD1.1

           ```
           # bertbase模型
           bash train_performance_squad1.1_base_bs24_1p.sh --data_path=/home
           
           # bertlarge模型
           bash train_performance_squad1.1_large_bs24_1p.sh --data_path=/home
           ```

           

         - SQuAD2.0

           ```
           # bertbase模型
           bash train_performance_squad2.0_base_bs24_1p.sh --data_path=/home
           
           # bertlarge模型
           bash train_performance_squad2.0_large_bs24_1p.sh --data_path=/home
           ```

           


<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码

```
|--LICENSE
|--README.md									#说明文档
|--input_pipeline.py
|--model_saving_utils.py
|--run_pretraining.py
|--run_squad.py  									#训练代码
|--requirements.txt		   						#所需依赖
|--squad_lib.py
|--test			           						#训练脚本目录
|	|--train_performance_bertlarge_8p_512bs_lamb_phase1.sh
|	|--train_performance_bertlarge_8p_96bs_lamb_phase2.sh
|	|--train_performance_bertlarge_8p_128bs_adam.sh
|	|--train_performance_squad1.1_large_bs24_1p.sh
|	|--train_performance_squad2.0_large_bs24_1p.sh
```

## 脚本参数<a name="section6669162441511"></a>

```
--data_path						# the path to train data
--epochs						# epochs of training
--ckpt_save_path				# directory to ckpt
--batch_size					# batch size for 1p
--log_steps						# log frequency
--bert_config_file
--precision_mode				# the path to save over dump data
--over_dump						# if or not over detection, default is False
--data_dump_flag				# data dump flag, default is False
--data_dump_step				# data dump step, default is 10
--profiling						# if or not profiling for performance debug, default is False
--profiling_dump_path			# the path to save profiling data
--over_dump_path				# the path to save over dump data
--data_dump_path				# the path to save dump data
--use_mixlist					# use_mixlist flag, default is False
--fusion_off_flag				# fusion_off flag, default is False
--mixlist_file					# mixlist file name, default is ops_info.json
--fusion_off_file				# fusion_off file name, default is fusion_switch.cfg
```

## 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡，8卡网络训练。模型存储路径为${cur_path}/output/$ASCEND_DEVICE_ID，包括训练的log以及checkpoints文件。以8卡训练为例，loss信息在文件${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。