# <font face="微软雅黑">

中文|[English](README_EN.md)

# BERT DownStream TensorFlow离线任务

***
此链接提供BERT DownStream模型在NPU上离线推理的脚本和方法

* [x] BERT DownStream推理,  基于[BERT](https://github.com/google-research/bert) 

***

## 注意
**此案例仅为您学习Ascend软件栈提供参考，不用于商业目的。**

在开始之前，请注意以下适配条件。如果不匹配，可能导致运行失败。

| Conditions | Need |
| --- | --- |
| CANN版本 | >=5.0.3 |
| 芯片平台| Ascend310/Ascend310P3 |
| 第三方依赖| 请参考 'requirements.txt' |

## 快速指南

### 1. 拷贝代码

```shell
git clone https://gitee.com/ascend/ModelZoo-TensorFlow.git
cd Modelzoo-TensorFlow/ACL_TensorFlow/contrib/nlp/BERT_for_ACL
```

### 2. 下载数据集和预处理

请自行下载数据集, 更多详情见: [data](./data)
请自行下载vocab.txt and bert_config.json , 更多详情见: [config](./config/README.md)

### 3. 获取训练好的checkpoint文件或者pb模型

获取训练好的checkpoint文件或者pb模型, 更多详情见: [ckpt](./save/ckpt/README.md) 或者 [models](./save/model/README.md)

### 4. 编译程序
编译推理工具, 更多详情见: [xacl_fmk](./xacl_fmk/README.md)
将xacl工具放至当前位置。

### 5. 离线推理

**BERT**
***
* BERT使用bert做为模型的名称, 每个下游任务各自做为模型名称。
* BERT支持ner, squad1.1, mrpc, cola, mnli and tnews等下游任务
* 改变模型入参，以支持不同的任务
* 只在BERT Base和BERT Large被测试过
***

**环境变量设置**

  请参考[说明](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719)，设置环境变量

**预处理**
* --data_dir为存放下游任务数据集的绝对路径, 同时，确保**predict**文件在此路径下，例如'dev.tsv'
* --output_dir和--data_dir一样, 预处理脚本会将text文件转换为bin文件并保存至此路径
* --vocab_file, --bert_config_file, --do_lower_case, --max_seq_length, --doc_stride等不变
* --model_name=bert 当下游任务为BERT时，模型名称为bert
* --task_name为下游所需的任务名, 仅支持ner, squad(squad1.1), mrpc, cola, mnli 和 tnews 任务
* 更多数据集和任务详情见README.md中各个数据集路径下的下载链接
```Bash
python3 xnlp_fmk.py \
    --data_dir=./data/SQuAD1.1 \
    --output_dir=./data/SQuAD1.1 \
    --vocab_file=./config/uncased_L-24_H-1024_A-16/vocab.txt \
    --bert_config_file=./config/uncased_L-24_H-1024_A-16/bert_config.json \
    --max_seq_length=384 \
    --do_lower_case=True \
    --model_name=bert \
    --task_name=squad \
    --action_type=preprocess

```

**冻结pb模型**
* --output_dir为相对路径, 冻结脚本会将checkpoint文件转换成pb模型文件至此路径下
* --checkpoint_dir路径下包含 'checkpoint', 'ckpt.data', 'ckpt.index' and 'ckpt.meta'等文件
* --pb_model_file 为真实模型文件名称
* --predict_batch_size入参为实际batch size值,或者赋'None'来做为动态batch size
* 其它参数同上
```Bash
python3 xnlp_fmk.py \
    --output_dir=./save/model \
    --vocab_file=./config/uncased_L-24_H-1024_A-16/vocab.txt \
    --bert_config_file=./config/uncased_L-24_H-1024_A-16/bert_config.json \
    --pb_model_file=./save/model/BERT_LARGE_SQuAD_BatchSize_None.pb \
    --checkpoint_dir=./save/ckpt/bert_large_squad \
    --max_seq_length=384 \
    --model_name=bert \
    --task_name=squad \
    --action_type=freeze

```

**pb模型转om**
* 重命名--om_model_file为真实的om离线模型
* 依据实际情况修改--soc_version, --in_nodes, --out_nodes等入参
* 如果需要的话，可添加另外的atc参数。例如, --precision_mode
* 修改--predict_batch_size入参为实际batch size值, 当前仅支持固态batch size
* 其它参数同上
```Bash
python3 xnlp_fmk.py \
    --output_dir=./save/model \
    --pb_model_file=./save/model/BERT_LARGE_SQuAD_BatchSize_None.pb \
    --om_model_file=./save/model/BERT_LARGE_SQuAD_BatchSize_1.om \
    --predict_batch_size=1 \
    --soc_version="Ascend310" \
    --in_nodes="\"input_ids:1,384;input_mask:1,384;segment_ids:1,384\"" \
    --out_nodes="\"logits:0\"" \
    --max_seq_length=384 \
    --model_name=bert \
    --task_name=squad \
    --action_type=atc

```

**运行离线推理**
* 修改 --output_dir改为相对路径, 脚本会将输出bin文件并保存至此路径下
* 编译推理工具，并将其放至当前路径下，更多详情见: [xacl_fmk](./xacl_fmk/README.md)
* 其它参数同上
```Bash
python3 xnlp_fmk.py \
    --data_dir=./data/SQuAD1.1 \
    --output_dir=./save/output \
    --om_model_file=./save/model/BERT_LARGE_SQuAD_BatchSize_1.om \
    --predict_batch_size=1 \
    --model_name=bert \
    --task_name=squad \
    --action_type=npu

```

**后处理**
* 修改 --output_dir改为相对路径, 脚本会保存精度结果至此路径下
* 其它参数同上
```Bash
python3 xnlp_fmk.py \
    --data_dir=./data/SQuAD1.1 \
    --output_dir=./save/output \
    --vocab_file=./config/uncased_L-24_H-1024_A-16/vocab.txt \
    --bert_config_file=./config/uncased_L-24_H-1024_A-16/bert_config.json \
    --om_model_file=./save/model/BERT_LARGE_SQuAD_BatchSize_1.om \
    --predict_batch_size=1 \
    --do_lower_case=True \
    --max_seq_length=384 \
    --model_name=bert \
    --task_name=squad \
    --action_type=postprocess

```

## 其他用法
**pb模型转换为pbtxt**
* --output_dir改为相对路径, 脚本将pb模型转换成pbtxt，并保存至此路径下
*--pb_model_file入参改为实际模型名称
* 其它参数同上
```Bash
python3 xnlp_fmk.py \
    --output_dir=./save/model \
    --pb_model_file=./save/model/BERT_LARGE_SQuAD_BatchSize_1.pb \
    --model_name=bert \
    --task_name=squad \
    --action_type=pbtxt

```

**pb模型推理**
* 依据实际情况，修改--in_nodes, --out_nodes 
* 其它参数同上
```Bash
python3 xnlp_fmk.py \
    --data_dir=./data/SQuAD1.1 \
    --output_dir=./save/output \
    --pb_model_file=./save/model/BERT_LARGE_SQuAD_BatchSize_1.pb \
    --predict_batch_size=1 \
    --in_nodes="\"input_ids:1,384;input_mask:1,384;segment_ids:1,384\"" \
    --out_nodes="\"logits:0\"" \
    --model_name=bert \
    --task_name=squad \
    --action_type=cpu

```

## 参考

[1] https://arxiv.org/abs/1810.04805

[2] https://github.com/google-research/bert

[3] https://github.com/kyzhouhzau/BERT-NER

[4] https://github.com/zjy-ucas/ChineseNER

[5] https://github.com/hanxiao/bert-as-service

[6] https://github.com/macanv/BERT-BiLSTM-CRF-NER

[7] https://github.com/tensorflow/tensor2tensor

[8] https://github.com/google-research/albert

[9] https://github.com/brightmart/albert_zh

[10] https://github.com/HqWei/Sentiment-Analysis

[11] https://gitee.com/wang-bain/xacl_fmk

[12] https://github.com/brightmart/roberta_zh

[13] https://github.com/dmis-lab/biobert

[14] https://github.com/Songweiping/GRU4Rec_TensorFlow

# </font>
