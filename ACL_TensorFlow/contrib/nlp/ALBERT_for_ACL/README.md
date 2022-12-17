# <font face="微软雅黑">

中文|[English](README_EN.md)

# ALBERT DownStream TensorFlow离线推理

***
此链接提供ALBERT DownStream TensorFlow模型在NPU上离线推理的脚本和方法

* [x] 英文版ALBERT DownStream TensorFlow, 见[albert](https://github.com/google-research/albert) 

* [x] 中文版ALBERT DownStream TensorFlow, 见[albert_zh](https://github.com/brightmart/albert_zh) 
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
cd Modelzoo-TensorFlow/ACL_TensorFlow/contrib/nlp/ALBERT_for_ACL
```

### 2. 下载数据集和预处理

请自行下载数据集, 更多详情见: [data](./data)
请自行下载 **vocab.txt and bert_config.json** , 更多详情见: [config](./config/README.md)

### 3. 获取训练好的checkpoint文件，或者pb模型。

[pb模型下载链接](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/2022-12-12_tf/ALBERT_SQuAD1_for_ACL/ALBERT_EN_BASE_SQuAD1.1_BatchSize_None.pb)

### 4. 编译程序
编译推理工具, 更多详情见: [xacl_fmk](./xacl_fmk/README.md)
将xacl工具放至当前位置。

### 5. 离线推理

**ALBERT_en**
***
* ALBERT_en使用albert_en做为模型的名称, 每个下游任务各自做为模型名称。
* ALBERT_en支持 cola, mnli, mrpc, race and squad1.1等下游任务。
* ALBERT_en支持spm_model 或者 vocab.txt前处理
* 改变模型入参，以支持不同的任务
* 只在ALBERT_en Base上测试过
***
**环境变量设置**

  请参考[说明](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719)，设置环境变量


**预处理**
* --data_dir：每个任务数据集的实际路径, 并且确保 **predict** 文件在当前路径下，例如'dev.tsv'
* --output_dir的传参与--data_dir相同, 预处理脚本会将文本转换为该路径下的bin文件
* ALBERT_en支持spm_model或vocab.txt做为预处理, --spm_model_file：使用 spm_model or --vocab_file：使用 vocab.txt
* --bert_config_file, --do_lower_case, --max_seq_length, --doc_stride等参数进行微调
* --model_name：当进行ALBERT_en任务时，参数为albert_en
* --task_name：任务名,仅支持cola, mnli, mrpc, race and squad(for squad1.1)任务
* 更多数据集和任务详细信息，如下载链接，请参阅自述文件。每个数据集路径中的README.md各个数据集路径下的下载链接
```Bash
python3 xnlp_fmk.py \
    --data_dir=./data/CoLA \
    --output_dir=./data/CoLA \
    --spm_model_file=./config/albert_en_config/30k-clean.model \
    --bert_config_file=./config/albert_en_config/albert_en_base_config.json \
    --do_lower_case=True \
    --model_name=albert_en \
    --task_name=cola \
    --action_type=preprocess

```

**冻结pb模型**
* --output_dir：在此路径下，冻结脚本会把checkpoint文件转成Pb模型
* --checkpoint_dir：checkpoint文件, 包括 'checkpoint', 'ckpt.data', 'ckpt.index' 和 'ckpt.meta'
* --pb_model_file：pb模型文件名
* --predict_batch_size：实际batch_size值 or 'None'来表示动态batch
* 其它参数同上
```Bash
python3 xnlp_fmk.py \
    --output_dir=./save/model \
    --checkpoint_dir=./save/ckpt/albert_en_base_cola \
    --pb_model_file=./save/model/ALBERT_EN_BASE_CoLA_BatchSize_1.pb \
    --bert_config_file=./config/albert_en_config/albert_en_base_config.json \
    --predict_batch_size=1 \
    --model_name=albert_en \
    --task_name=cola \
    --action_type=freeze

```

**离线模型转换**
* --om_model_file：om模型名
* --soc_version, --in_nodes, --out_nodes：根据实际情况传参
* 添加额外需要的atc参数，例如： --precision_mode
* --predict_batch_size ：实际batch, 当前仅支持静态batch
* 其它参数同上
```Bash
python3 xnlp_fmk.py \
    --output_dir=./save/model \
    --pb_model_file=./save/model/ALBERT_EN_BASE_CoLA_BatchSize_1.pb \
    --om_model_file=./save/model/ALBERT_EN_BASE_CoLA_BatchSize_1.om \
    --soc_version="Ascend310" \
    --in_nodes="\"input_ids:1,128;input_mask:1,128;segment_ids:1,128\"" \
    --out_nodes="\"logits:0\"" \
    --predict_batch_size=1 \
    --model_name=albert_en \
    --task_name=cola \
    --action_type=atc

```

**运行推理**
* --output_dir：脚本将在该路径下保存输出bin文件
* 构建推理应用程序并将其置于当前路径下，详情见: [xacl_fmk](./xacl_fmk/README.md)
* 其它参数同上
```Bash
python3 xnlp_fmk.py \
    --data_dir=./data/CoLA \
    --output_dir=./save/output \
    --om_model_file=./save/model/ALBERT_EN_BASE_CoLA_BatchSize_1.om \
    --predict_batch_size=1 \
    --model_name=albert_en \
    --task_name=cola \
    --action_type=npu

```

**后处理**
* --output_dir：脚本将在该路径下保存精度结果文件
* 其它参数同上
```Bash
python3 xnlp_fmk.py \
    --data_dir=./data/CoLA \
    --output_dir=./save/output \
    --spm_model_file=./config/albert_en_config/30k-clean.model \
    --om_model_file=./save/model/ALBERT_EN_BASE_CoLA_BatchSize_1.om \
    --predict_batch_size=1 \
    --do_lower_case=True \
    --model_name=albert_en \
    --task_name=cola \
    --action_type=postprocess

```

**ALBERT_zh**
***
* ALBERT_zh使用albert_zh做为模型的名称, 每个下游任务各自做为模型名称。
* ALBERT_zh支持afqmc,cmnli,csl,iflytek,tnews和wsc任务
* 改变模型入参，以支持不同的任务
* 仅ALBERT_zh Tiny测试过
***
**预处理**
* --data_dir：每个任务数据集的实际路径, 并且确保 **predict** 文件在当前路径下，例如'dev.tsv'
* --output_dir的传参与--data_dir相同, 预处理脚本会将文本转换为该路径下的bin文件
* --vocab_file, --bert_config_file, --do_lower_case, --max_seq_length, --doc_stride等参数进行微调
* --model_name：当进行ALBERT_en任务时，参数为albert_en
* --task_name为下游所需的任务名, 仅支持afqmc, cmnli, csl, iflytek, tnews 和 wsc 任务
```Bash
python3 xnlp_fmk.py \
    --data_dir=./data/TNEWS \
    --output_dir=./data/TNEWS \
    --vocab_file=./config/albert_zh_config/vocab.txt \
    --bert_config_file=./config/albert_zh_config/albert_zh_tiny_config.json \
    --model_name=albert_zh \
    --task_name=tnews \
    --action_type=preprocess

```

**冻结pb模型**
* --output_dir：在此路径下，冻结脚本会把checkpoint文件转成Pb模型
* --checkpoint_dir：checkpoint文件, 包括 'checkpoint', 'ckpt.data', 'ckpt.index' 和 'ckpt.meta'
* --pb_model_file: pb模型文件名
* --predict_batch_size：实际batch size值,或者以'None'来做为动态batch size
* 其它参数同上
```Bash
python3 xnlp_fmk.py \
    --output_dir=./save/model \
    --pb_model_file=./save/model/ALBERT_ZH_TINY_TNEWS_BatchSize_None.pb \
    --checkpoint_dir=./save/ckpt/albert_zh_tiny_tnews \
    --model_name=albert_zh \
    --task_name=tnews \
    --action_type=freeze

```

**pb模型转om**
* --om_model_file：om模型名
* --soc_version, --in_nodes, --out_nodes ：根据实际情况传参
* 添加额外需要的atc参数，例如： --precision_mode
* --predict_batch_size ：实际batch, 当前仅支持静态batch
* 其它参数同上
```Bash
python3 xnlp_fmk.py \
    --output_dir=./save/model \
    --pb_model_file=./save/model/ALBERT_ZH_TINY_TNEWS_BatchSize_None.pb \
    --om_model_file=./save/model/ALBERT_ZH_TINY_TNEWS_BatchSize_1.om \
    --predict_batch_size=1 \
    --soc_version="Ascend310" \
    --in_nodes="\"input_ids:1,128;input_mask:1,128;segment_ids:1,128\"" \
    --out_nodes="\"logits:0\"" \
    --model_name=albert_zh \
    --task_name=tnews \
    --action_type=atc

```

**运行离线推理**
* --output_dir：脚本将在该路径下保存输出bin文件
* 构建推理应用程序并将其置于当前路径下，详情见: [xacl_fmk](./xacl_fmk/README.md)
* 其它参数同上
```Bash
python3 xnlp_fmk.py \
    --data_dir=./data/TNEWS \
    --output_dir=./save/output \
    --om_model_file=./save/model/ALBERT_ZH_TINY_TNEWS_BatchSize_1.om \
    --predict_batch_size=1 \
    --model_name=albert_zh \
    --task_name=tnews \
    --action_type=npu

```

**后处理**
* --output_dir:脚本将在该路径下保存精度结果文件
* 其它参数同上
```Bash
python3 xnlp_fmk.py \
    --data_dir=./data/TNEWS \
    --output_dir=./save/output \
    --vocab_file=./config/albert_zh_config/vocab.txt \
    --om_model_file=./save/model/ALBERT_ZH_TINY_TNEWS_BatchSize_1.om \
    --predict_batch_size=1 \
    --model_name=albert_zh \
    --task_name=tnews \
    --action_type=postprocess

```

## 其他用法
**将pb模型转换为pbtxt**
* --output_dir：在此路径下，脚本会将pb模型转为pbtxt模型文件
*--pb_model_file：pb模型文件名
* 其它参数同上
```Bash
python3 xnlp_fmk.py \
    --output_dir=./save/model \
    --pb_model_file=./save/model/ALBERT_EN_BASE_CoLA_BatchSize_1.pb \
    --model_name=albert_en \
    --task_name=cola \
    --action_type=pbtxt

```

**pb模型推理**
* --in_nodes, --out_nodes：根据实际情况传参
* 其它参数同上
```Bash
python3 xnlp_fmk.py \
    --data_dir=./data/Cola \
    --output_dir=./save/output \
    --pb_model_file=./save/model/ALBERT_EN_BASE_CoLA_BatchSize_1.pb \
    --predict_batch_size=1 \
    --in_nodes="\"input_ids:1,128;input_mask:1,128;segment_ids:1,128\"" \
    --out_nodes="\"logits:0\"" \
    --model_name=albert_en \
    --task_name=cola \
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
