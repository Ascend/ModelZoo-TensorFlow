中文|[English](README_EN.md)
# <font face="微软雅黑">

# RoBERTa TensorFlow离线推理

***
此链接提供RoBERTa TensorFlow模型在NPU上离线推理的脚本和方法

* [x] RoBERTa 离线推理的连接 [RoBERTa](https://github.com/brightmart/roberta_zh) 

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
git clone https://gitee.com/ascend/modelzoo.git
cd Modelzoo-TensorFlow/ACL_TensorFlow/contrib/nlp/RoBERTa_for_ACL
```

### 2. 下载数据集和预处理

请自行下载数据集, 并放在data目录下(若目录不存在请在子项目根目录下自行创建),

请自行下载vocab.txt 和 bert_config.json 文件, 更多详细信息请参见: [config](./config/README.md)

### 3. 获取微调的检查点文件或pb模型

获取微调的检查点文件或pb模型, 更多详细信息请参见: [ckpt](./save/ckpt/README.md) or [models](./save/model/README.md)

### 4. 编译程序
编译推理应用程序, 更多详细信息请参见: [xacl_fmk](./xacl_fmk/README.md)
将xacl放在当前字典中

### 5. 离线推理

**RoBERTa**
***
* RoBERTa将roberta用作model_name参数，将每个下游任务名称用作task_name
* RoBERTa支持lcqmc任务
* 更改不同任务的参数
* 仅对RoBERTa Base进行了测试
***
**环境变量设置**
请参考[说明](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719)，设置环境变量

**预处理**
* 将--data_dir更改为每个下游任务数据集的实际路径，并确保路径下的**predict**文件，如“dev.tsv”
* 将--output_dir更改为与--data_dir相同，预处理脚本将把文本转换为该路径下的bin文件
* 通过微调参数使--vocab_file、--bert_config_file和--do_lower_case、--max_seq_length、--doc_stride等保持不变
* 执行roberta任务时保留--model_name=roberta
* 将--task_name更改为要执行的下游任务，仅支持lcqmc任务
* 更多数据集和任务详细信息，如下载链接，请参阅自述文件。每个数据集路径中的readme.md
```Bash
python3 xnlp_fmk.py \
    --data_dir=./data/LCQMC \
    --output_dir=./data/LCQMC \
    --vocab_file=./config/roberta_large/vocab.txt \
    --bert_config_file=./config/roberta_large/bert_config_large.json \
    --do_lower_case=True \
    --model_name=roberta \
    --task_name=lcqmc \
    --action_type=preprocess

```

**固定pb模型**
* 将--output_dir更改为实际路径，冻结脚本将把检查点文件转换为该路径下的pb模型文件
* 将--checkpoint_dir更改为检查点文件的实际路径，包括“checkpoint”、“ckpt”。数据'，'ckpt。索引'和'ckpt.meta'
* 将--pb_model_file重命名为真正的pb模型文件名
* 将--predict_batch_size更改为实际批次大小，或为动态批次指定“无”
* 保持其他参数与上一步相同
```Bash
python3 xnlp_fmk.py \
    --output_dir=./save/model \
    --bert_config_file=./config/roberta_large/bert_config_large.json \
    --pb_model_file=./save/model/ROBERTA_LARGE_LCQMC_BatchSize_None.pb \
    --checkpoint_dir=./save/ckpt/roberta_large_lcqmc \
    --model_name=roberta \
    --task_name=lcqmc \
    --action_type=freeze

```

**将pb转换为om**
* 将--om_model_file重命名为实际的om模型文件名
* 根据实际情况更改--soc_version、--in_nodes、--out_nodes
* 如果需要，可以添加其他atc参数，例如--precision_mode
* 将--predict_batch_size更改为实际批量大小，当前仅支持静态批量大小
* 保持其他参数与上一步相同
```Bash
python3 xnlp_fmk.py \
    --output_dir=./save/model \
    --pb_model_file=./save/model/ROBERTA_LARGE_LCQMC_BatchSize_None.pb \
    --om_model_file=./save/model/ROBERTA_LARGE_LCQMC_BatchSize_1.om \
    --predict_batch_size=1 \
    --soc_version="Ascend310" \
    --in_nodes="\"input_ids:1,128;input_mask:1,128;segment_ids:1,128\"" \
    --out_nodes="\"logits:0\"" \
    --model_name=roberta \
    --task_name=lcqmc \
    --action_type=atc

```

**运行推断**
* 将--output_dir更改为实际路径，脚本将在该路径下保存输出bin文件
* 构建推理应用程序并将其置于当前路径下，更多详细信息请参见：[xacl_fmk]（./xacl_5mk/README.md）
* 保持其他参数与上一步相同
```Bash
python3 xnlp_fmk.py \
    --data_dir=./data/LCQMC \
    --output_dir=./save/output \
    --om_model_file=./save/model/ROBERTA_LARGE_LCQMC_BatchSize_1.om \
    --predict_batch_size=1 \
    --model_name=roberta \
    --task_name=lcqmc \
    --action_type=npu

```

**后期处理**
* 将--output_dir更改为实际路径，脚本将在此路径下保存精度结果文件
* 保持其他参数与上一步相同
```Bash
python3 xnlp_fmk.py \
    --data_dir=./data/LCQMC \
    --output_dir=./save/output \
    --vocab_file=./config/roberta_large/vocab.txt \
    --om_model_file=./save/model/ROBERTA_LARGE_LCQMC_BatchSize_1.om \
    --predict_batch_size=1 \
    --do_lower_case=True \
    --model_name=roberta \
    --task_name=lcqmc \
    --action_type=postprocess

```

## 其他用途
**将pb转换为pbtxt**
* 将--output_dir更改为实际路径，convert脚本将在此路径下将pb模型文件转换为pbtxt模型文件
* 将--pb_model_file重命名为真正的pb模型文件名
* 保持其他参数与上一步相同
```Bash
python3 xnlp_fmk.py \
    --output_dir=./save/model \
    --pb_model_file=./save/model/ROBERTA_LARGE_LCQMC_BatchSize_1.pb \
    --model_name=roberta \
    --task_name=lcqmc \
    --action_type=pbtxt

```

**通过pb模型运行推断**
* 根据实际情况更改--in_nodes、--out_nodes
* 保持其他参数与上一步相同
```Bash
python3 xnlp_fmk.py \
    --data_dir=./data/GAD \
    --output_dir=./save/output \
    --pb_model_file=./save/model/ROBERTA_LARGE_LCQMC_BatchSize_1.pb \
    --predict_batch_size=1 \
    --in_nodes="\"input_ids:1,128;input_mask:1,128;segment_ids:1,128\"" \
    --out_nodes="\"logits:0\"" \
    --model_name=roberta \
    --task_name=lcqmc \
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
