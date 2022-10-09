- [基本信息](#基本信息.md)
- [概述](#概述.md)
- [训练环境准备](#训练环境准备.md)
- [快速上手](#快速上手.md)
- [迁移学习指导](#迁移学习指导.md)
- [高级参考](#高级参考.md)
## 基本信息

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Natural Language Processing**

**版本（Version）：1.1**

**修改时间（Modified） ：2022.6.13**

**大小（Size）：381KB**

**框架（Framework）：TensorFlow_2.6.2**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的Roberta_ZH预训练及微调代码**

## 概述

在自然语言处理领域中，预训练语言模型（Pre-trained Language Models）已成为非常重要的基础技术。为了进一步促进中文信息处理的研究发展，基于全词遮罩（Whole Word Masking）技术的中文预训练模型BERT-wwm应运而生，以及与此技术密切相关的模型：BERT-wwm-ext，RoBERTa-wwm-ext，RoBERTa-wwm-ext-large, RBT3, RBTL3。

本项目包含两个场景：Roberta-pretrain与Roberta-finetune，两个场景共用本项目下的代码，通过./Roberta_Series_for_TensorFlow2.X/test路径下不同的脚本拉起训练，详见“脚本和示例代码”；两个场景所用的数据集并不相同，详见“数据集准备”。

- 参考论文：

  https://arxiv.org/abs/1906.08101

- 参考实现：

  https://github.com/brightmart/roberta_zh

- 适配昇腾 AI 处理器的实现：
  
  https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow2/built-in/nlp/Roberta_Series_for_TensorFlow2.X

- 通过Git获取对应commit\_id的代码方法如下：
  
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    

#### 默认配置<a name="section91661242121611"></a>

-   网络结构：
    -   roberta-xlarge
-   pretrain训练超参（多卡）：
    -   batch size: 16
    -   learning rate: 2e-5
    -   max_predictions_per_seq: 23
    -   max_seq_length: 256
    -   save_checkpoints_step: 10000
-   finetune训练超参（单卡）：
    -   batch size: 32
    -   learning rate: 3e-5
    -   max_seq_length:128


#### 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持（pretrain) | 是否支持(finetune) |
| ---------- | ------------------- | ------------------ |
| 分布式训练 | 是                  | 否                 |
| 混合精度   | 是                  | 是                 |
| 数据并行   | 否                  | 否                 |


#### 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

#### 开启混合精度<a name="section20779114113713"></a>

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

## 训练环境准备

-  硬件环境和运行环境准备请参见《[CANN软件安装指南](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=installation-update)》
-  运行以下命令安装依赖。
```
pip3 install requirements.txt
```
说明：依赖配置文件requirements.txt文件位于模型的根目录

## 快速上手

#### 数据集准备<a name="section361114841316"></a>

- Roberta-pretrain使用的数据集是通用语料数据，数据集路径：https://github.com/brightmart/nlp_chinese_corpus，请用户自行下载数据集。

  通过网络目录下的create_pretrain_data.sh脚本将原始的txt文本数据转化为tfrecord数据，请用户参考“参考实现”中的数据集转化方法进行转化，示例如下所示：

  ```
  如将1到10个txt转化为tfrecord数据：
  nohup bash create_pretrain_data.sh 1 10 &
  注：在我们的实验中使用15%的比例做全词遮蔽，模型学习难度大、收敛困难，所以我们用了10%的比例
  ```

  转化后的数据集目录参考如下：

  ```
  └──wiki_zh.tfrecord
  ```

  

- Roberta-finetune使用的数据集为XNLI数据集，请用户参考“参考实现”中的数据集下载链接自行下载

  数据集目录参考如下：

  ```
  ├──README.md
  ├──train.tf_record
  ├──xnli.dev.jsonl
  ├──xnli.dev.tsv
  ├──xnli.test.jsonl
  ├──xnli.test.tsv
  └──multinli
  	└──multinli.train.zh.tsv
  ```

  用户需自行下载预训练模型，请用户参考“参考实现”中的模型下载链接自行下载，本项目中使用的模型为 RoBERTa-wwm-ext-large 

模型目录参考如下：

```
chinese_roberta_wwm_large_ext_L-24_H-1024_A-16
	├──bert_config.json
    ├──bert_model.ckpt.data-00000-of-00001
    ├──bert_model.ckpt.index
    ├──bert_model.ckpt.meta
    └──vocab.txt
```



#### 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。

- 开始训练Robert-pretrain。

    1. 启动训练之前，首先要配置程序运行相关环境变量。
    
       环境变量配置信息参见：
    
       [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)
    
    2. 单卡训练
    
       单卡训练指令（脚本位于Roberta_Series_for_TensorFlow2.X/test/train_ID3214_Roberta_Pretrain_performance_1p.sh）,需要先使用cd命令进入test目录下，再使用下面的命令启动训练。请确保下面例子中的“--data_path”修改为用户的数据路径,这里选择将数据文件夹data放在/home目录下。
    
       ```
       bash train_ID3214_Roberta_Pretrain_performance_1p.sh --data_path=/home/data
       ```
    
    3. 多卡训练
    
       多卡训练指令（以8卡为例脚本位于Roberta_Series_for_TensorFlow2.X/test/train_ID3214_Roberta_Pretrain_full_8p.sh）,需要先使用cd命令进入test目录下，再使用下面的命令启动训练。请确保下面例子中的“--data_path”修改为用户的数据路径,这里选择将Roberta-pretrain的训练数据文件夹data放在/home目录下，运行以下命令拉起训练：
    
       ```
       bash train_ID3214_Roberta_Pretrain_full_8p.sh --data_path=/home/data
       ```
    
- 开始训练Robert-finetune

    1. 启动训练之前，首先要配置程序运行相关环境变量。
    
    	环境变量配置信息参见：
    
       [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)
       
    2. 单卡训练
    
        单卡训练指令（脚本位于Roberta_Series_for_TensorFlow2.X/test/train_ID3222_Roberta_Finetune_performance_1p.sh）,需要先使用cd命令进入test目录下，再使用下面的命令启动训练。请确保下面例子中的“--data_path”修改为用户的数据路径,这里选择将Roberta-finetune的训练数据文件夹data放在/home目录下，将模型文件夹ckpt放在/home目录下，运行以下命令拉起训练：
    
        ```
        bash train_ID3222_Roberta_Finetune_performance_1p.sh --data_path=/home/data --ckpt_path=/home/ckpt
        ```
    
        


## 高级参考

#### 脚本和示例代码

```
├──resources
|	└──vocab.txt
├──LICENSE
├──README.md													#说明文档
├──bert_config_large.json										#BERT配置文件
├──create_pretrain_data.sh
├──create_pretaining_data.sh
├──modeling.py
├──modelzoo_level.txt
├──optimization.py
├──optimization_finetuning.py  				
├──requirements.txt		   										#所需依赖
├──run_classifier.py											#finetune
├──run_pretraining.py											#pretrain
├──tokenization.py
└──test			           										#训练脚本目录
	├──1p.json
	├──8p.json
	├──set_ranktable.py
	├──train_ID3214_Roberta_Pretrain_full_8p.sh					#robert-pretrain 全量 8P脚本
	├──train_ID3214_Roberta_Pretrain_performance_16p.sh			#robert-pretrain performance 16p脚本
	├──train_ID3214_Roberta_Pretrain_performance_1p.sh			#robert-pretrain performance 1p脚本
	├──train_ID3214_Roberta_Pretrain_performance_8p.sh			#robert-pretrain performance 8p脚本
	├──train_ID3222_Roberta_Finetune_full_1p.sh					#robert-finetune 全量 1p脚本
	└──train_ID3222_Roberta_Finetune_performance_1p.sh			#robert-finetune performance 1p脚本
```

#### 脚本参数<a name="section6669162441511"></a>

1、Roberta-pretrain脚本参数

```
--input_file					# the path to train data
--output_dir					# path of saving outputs
--do_train						# if or not training
--do_eval						# if or not evaluation
--bert_config_file				# path of bert config file
--train_batch_size				# training batch size
--max_seq_length				# max sequence length
--max_predictions_per_seq		# max predictions per sequence
--num_train_steps				# training steps
--num_warmup_steps				# warmup steps
--learning_rate					# learning rate, default is 2e-5
--save_checkpoints_steps		# default is 10000, saving a checkpoint each 10000 steps
--distributed					# default is False
```

2、Roberta-finetune脚本参数

```
--task_name						# in this case is xnli, it can be changed to other task names
--do_train						# if or not training
--do_eval						# if or not evaluation
--data_dir						# data path
--vocab_file					# vocab file path
--bert_config_file				# path of bert config file
--init_checkpoint				# initial checkpoints for training
--max_seq_length				# max sequence length
--train_batch_size				# training batch size
--learning_rate					# learning rate, default is 3e-5
--num_train_epochs				# in performance is 1, in full is 2
--output_dir					# the path of saving outputs
```

#### 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡或者多卡的full或performance训练。
单卡和多卡通过运行不同脚本启动，支持Roberta-pretrain的单卡，8卡网络训练、Roberta-finetune的单卡网络训练。
模型存储路径为${cur_path}/output/$ASCEND_DEVICE_ID，包括训练的log以及checkpoints文件。
以单卡训练为例，训练打屏日志在文件${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。