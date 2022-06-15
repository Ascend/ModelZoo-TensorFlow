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

**修改时间（Modified） ：2022.6.13**

**大小（Size）：1.66MB**

**框架（Framework）：TensorFlow_2.6.2**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的AlBert微调代码**

<h2 id="概述.md">概述</h2>

## 简述<a name="section194554031510"></a>

ALBERT，一种轻量级BERT，用于语言表征的自监督学习。

本项目中包含使用ALBERT V2 Base和ALBERT V2 Large两种预训练模型的finetune下游分类任务，两个场景使用相同的数据及代码，使用的预训练模型不相同。

- 参考论文：

  https://arxiv.org/abs/1909.11942

- 参考实现：

  https://github.com/tensorflow/models/tree/r2.4.0/official/nlp/albert

- 适配昇腾 AI 处理器的实现：
  
  https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow2/built-in/nlp/albert-FineTune_Series_for_TensorFlow2.X

- 通过Git获取对应commit\_id的代码方法如下：
  
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    

## 默认配置<a name="section91661242121611"></a>

-   网络结构：
    -   ALBERT V2 Base: 12-layer, 768-hidden, 12-heads, 12M parameters
    -   ALBERT V2 Large: 24-layer, 1024-hidden, 16-heads, 18M parameters
-   基于ALBERT V2 Base的finetune训练超参（单卡）：
    -   attention_probs_dropout_prob: 0
    -   hidden_act: "gelu"
    -   hidden_dropout_prob: 0
    -   embedding_size: 128
    -   hidden_size: 768
    -   initializer_range: 0.02
    -   intermediate_size: 3072
    -   max_position_embeddings: 512
    -   num_attention_heads: 12
    -   num_hidden_layers: 12
    -   num_hidden_grops: 1
    -   net_structure_type: 0
    -   gap_size: 0
    -   num_memory_blocks: 0
    -   inner_group_num: 1
    -   down_scale_factor: 1
    -   type_vocab_size: 2
    -   vocab_size: 30000
    -   train_batch_size: 4
    -   eval_batch_size: 4
    -   learning_rate: 2e-5
-   基于ALBERT V2 Large的finetune训练超参（单卡）：
    -   attention_probs_dropout_prob: 0
    -   hidden_act: "gelu"
    -   hidden_dropout_prob: 0
    -   embedding_size: 128
    -   hidden_size: 1024
    -   initializer_range: 0.02
    -   intermediate_size: 4096
    -   max_position_embeddings: 512
    -   num_attention_heads: 16
    -   num_hidden_layers: 24
    -   num_hidden_grops: 1
    -   net_structure_type: 0
    -   gap_size: 0
    -   num_memory_blocks: 0
    -   inner_group_num: 1
    -   down_scale_factor: 1
    -   type_vocab_size: 2
    -   vocab_size: 30000
    -   train_batch_size: 4
    -   eval_batch_size: 4
    -   learning_rate: 4e-6


## 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持（ALBERT V2 Base finetune) | 是否支持(ALBERT V2 Large finetune) |
| ---------- | ---------------------------------- | ---------------------------------- |
| 分布式训练 | 否                                 | 否                                 |
| 混合精度   | 是                                 | 是                                 |
| 数据并行   | 否                                 | 否                                 |


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

1、ALBERT V2 Base finetune和ALBERT V2 Large finetune所使用的数据集均为GLUE数据集，请用户参考“参考实现”中数据集下载方式下载原始GLUE数据集。

下载的原始数据集需要通过./data/create_finetune_data.py转化成tf_record数据，详细请参考“参考实现”中的数据集转换部分，示例代码如下：

```shell
export GLUE_DIR=~/glue											# GLUE原始数据文件夹
export ALBERT_DIR=/albert/checkpoints/albert_v2_base			# Albert模型文件

export TASK_NAME=MNLI											# 以task MNLI为例
export OUTPUT_DIR=/some_bucket/datasets							# 保存路径
python ../data/create_finetuning_data.py \
 --input_data_dir=${GLUE_DIR}/${TASK_NAME}/ \
 --sp_model_file=${ALBERT_DIR}/30k-clean.model \
 --train_data_output_path=${OUTPUT_DIR}/${TASK_NAME}_train.tf_record \
 --eval_data_output_path=${OUTPUT_DIR}/${TASK_NAME}_eval.tf_record \
 --meta_data_file_path=${OUTPUT_DIR}/${TASK_NAME}_meta_data \
 --fine_tuning_task_type=classification --max_seq_length=128 \
 --classification_task_name=${TASK_NAME} \
 --tokenization=SentencePiece
```

转化后的数据集目录参考如下：

```
MNLI_dataset
	├── MNLI_eval.tf_record
	├── MNLI_meta_data
	└── MNLI_train.tf_record
```

2、ALBERT V2 Base finetune使用的模型文件为ALBERT V2 Base，请用户参考“参考实现”中的模型下载链接自行下载对应模型

3、ALBERT V2 Large finetune使用的模型文件为ALBERT V2 Large，请用户参考“参考实现”中的模型下载链接自行下载对应模型

4、请用户将前几步下载并转换的数据集和下载的模型文件放在同一路径下，最终的数据集目录参考如下：

```
albert_data
	├── albert_base
	|	├── 30k-clean.model
	|	├── 30k-clean.vocab
	|	├── albert_config.json
	|	├── bert_model.ckpt.data-00000-of-00001
	|	└── bert_model.ckpt.index
	├── albert_large
	|	├── 30k-clean.model
	|	├── 30k-clean.vocab
	|	├── albert_config.json
	|	├── bert_model.ckpt.data-00000-of-00001
	|	└── bert_model.ckpt.index
	└── MNLI_dataset
		├── MNLI_eval.tf_record
		├── MNLI_meta_data
		└── MNLI_train.tf_record
```



## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。

- 开始训练ALBERT V2 Base finetune。

    1. 启动训练之前，首先要配置程序运行相关环境变量。
    
       环境变量配置信息参见：
    
       [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)
    
    2. 单卡训练
    
       单卡训练指令（脚本位于albert-FineTune_Series_for_TensorFlow2.X/test/train_ID3243_albert_FineTune_GLUE_AlbertBase_full_1p.sh）,需要先使用cd命令进入test目录下，再使用下面的命令启动训练。请确保下面例子中的“--data_path”修改为用户的数据路径,这里选择将数据文件夹albert_data放在/home目录下，运行以下命令拉起训练：
    
       ```
       bash train_ID3243_albert_FineTune_GLUE_AlbertBase_full_1p.sh --data_path=/home/albert_data
       ```
    
- 开始训练ALBERT V2 Large finetune。

    1. 启动训练之前，首先要配置程序运行相关环境变量。
    
    	环境变量配置信息参见：
    
       [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)
       
    2. 单卡训练
    
        单卡训练指令（脚本位于albert-FineTune_Series_for_TensorFlow2.X/test/train_ID3244_albert_FineTune_GLUE_AlbertLarge_full_1p.sh）,需要先使用cd命令进入test目录下，再使用下面的命令启动训练。请确保下面例子中的“--data_path”修改为用户的数据路径,这里选择将数据文件夹albert_data放在/home目录下，运行以下命令拉起训练：
    
        ```
        bash train_ID3244_albert_FineTune_GLUE_AlbertLarge_full_1p.sh --data_path=/home/albert_data
        ```
    
        


<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码

```
├── configs
|	└── ops_info.json
├──	official
|	└── ...
├── LICENSE
├── README.md																	# 说明文档				
├── requirements.txt		   													# 所需依赖
└── test			           													# 训练脚本目录
	├── train_ID3243_albert_FineTune_GLUE_AlbertBase_full_1p.sh					# ALBERT V2 Base finetune full 1p训练拉起脚本
	├── train_ID3243_albert_FineTune_GLUE_AlbertBase_performance_1p.sh			# ALBERT V2 Base finetune performance 1p训练拉起脚本
	├── train_ID3244_albert_FineTune_GLUE_AlbertLarge_full_1p.sh				# ALBERT V2 Large finetune full 1p训练拉起脚本
	└── train_ID3244_albert_FineTune_GLUE_AlbertLarge_performance_1p.sh			## ALBERT V2 Large finetune performance 1p训练拉起脚本
```

## 脚本参数<a name="section6669162441511"></a>

1、ALBERT V2 Base finetune及ALBERT V2 Large finetune脚本参数

```
--mode							# default is "train_and_eval"
--input_meta_data_path			# path of input meta data
--train_data_path				# path of training data
--eval_data_path				# path of evaluation data
--bert_config_file				# path of bert config file
--init_checkpoint				# initial checkpoint for training
--train_batch_size				# training batch size
--eval_batch_size				# evaluation batch size
--steps_per_loop				# steps per loop, default is 1
--log_steps						# default is 98175, print log each 98175 steps
--learning_rate					# learning rate
--num_train_epochs				# training epochs
--model_dir						# path of saving model
--distribution_strategy			# default is one_device
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
--auto_tune						# auto tune flag, default is false
```



## 训练过程<a name="section1589455252218"></a>

```
通过“模型训练”中的训练指令可以启动两个场景的单卡full或performance训练。
模型存储路径为${cur_path}/output/$ASCEND_DEVICE_ID，包括训练的log以及checkpoints文件。
以单卡训练为例，训练打屏日志在文件${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。
```