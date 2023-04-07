# <font face="微软雅黑">

English|[中文](README.md)

# ALBERT DownStream Inference for TensorFlow

***
This repository provides a script and recipe to Inference the ALBERT DownStream Inference

* [x] ALBERT DownStream Inference for English, based on [albert](https://github.com/google-research/albert) 

* [x] ALBERT DownStream Inference for Chinese, based on [albert_zh](https://github.com/brightmart/albert_zh) 
***

## Notice
**This sample only provides reference for you to learn the Ascend software stack and is not for commercial purposes.**

Before starting, please pay attention to the following adaptation conditions. If they do not match, may leading in failure.

| Conditions | Need |
| --- | --- |
| CANN Version | >=5.0.3 |
| Chip Platform| Ascend310/Ascend310P3 |
| 3rd Party Requirements| Please follow the 'requirements.txt' |

## Quick Start Guide

### 1. Clone the respository
```shell
git clone https://gitee.com/ascend/ModelZoo-TensorFlow.git
cd Modelzoo-TensorFlow/ACL_TensorFlow/contrib/nlp/ALBERT_for_ACL
```

### 2. Download and preprocess the dataset

Download the dataset and save it to the data directory. If the directory does not exist, create it in the root directory of the subproject.

Download the vocab.txt and bert_config.json by yourself, more details see: [config](./config/README.md)

### 3. Obtain the fine-tuned checkpoint files or pb model

[pb download link](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/2022-12-12_tf/ALBERT_SQuAD1_for_ACL/ALBERT_EN_BASE_SQuAD1.1_BatchSize_None.pb)

### 4. Build the program
Build the inference application, more details see: [xacl_fmk](./xacl_fmk/README.md)
Put xacl to the current dictory.

### 5. Offline Inference

**ALBERT_en**
***
* ALBERT_en use albert_en for model_name parameter, each downstream task name for task_name
* ALBERT_en support cola, mnli, mrpc, race and squad1.1 tasks
* ALBERT_en support spm_model or vocab.txt to do preprocess
* Change the parameters for different tasks
* Only ALBERT_en Base has been tested
***
**Configure the env**

  Please follow the [guide](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719) to set the envs


**PreProcess**
* Change --data_dir to the real path of each downstream task dataset, and make sure the **predict** file under the path such as 'dev.tsv'
* Change --output_dir to the same with --data_dir, and preprocess script will convert text to bin files under this path
* ALBERT_en support spm_model or vocab.txt to do preprocess, --spm_model_file when using spm_model or --vocab_file when using vocab.txt
* Keep the --bert_config_file, --do_lower_case, --max_seq_length, --doc_stride, etc. the same with fine-tuning parameters
* Keep the --model_name=albert_en when do the ALBERT_en tasks
* Change --task_name to the downstream task you want to do, only support cola, mnli, mrpc, race and squad(for squad1.1) tasks
* More datasets and tasks details like download link see README.md in each datasets' path
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

**Freeze pb model**
* Change --output_dir to the real path, and freeze script will convert checkpoint files to pb model file under this path
* Change --checkpoint_dir to the real path of checkpoint files, include 'checkpoint', 'ckpt.data', 'ckpt.index' and 'ckpt.meta'
* Rename --pb_model_file to the real pb model file name
* Change --predict_batch_size to the real batch size, or give 'None' for dynamic batch
* Keep other parameters the same as the previous step
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

**Convert pb to om**
* Rename --om_model_file to the real om model file name
* Change the --soc_version, --in_nodes, --out_nodes according to the actual situation
* Add additional atc parameters if you need, e.g., --precision_mode
* Change --predict_batch_size to the real batch size, currently only support static batch size
* Keep other parameters the same as the previous step
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

**Run the inference**
* Change --output_dir to the real path and script will save the output bin file under this path
* Build the inference application and put it under current path, more details see: [xacl_fmk](./xacl_fmk/README.md)
* Keep other parameters the same as the previous step
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

**PostProcess**
* Change --output_dir to the real path and script will save the precision result file under this path
* Keep other parameters the same as the previous step
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
* ALBERT_zh use albert_zh for model_name parameter, each downstream task name for task_name
* ALBERT_zh support afqmc, cmnli, csl, iflytek, tnews , lcqmc and wsc tasks
* Change the parameters for different tasks
* Only ALBERT_zh Tiny has been tested
***
**PreProcess**
* Change --data_dir to the real path of each downstream task dataset, and make sure the **predict** file under the path such as 'dev.tsv'
* Change --output_dir to the same with --data_dir, and preprocess script will convert text to bin files under this path
* Keep the --vocab_file, --bert_config_file, --do_lower_case, --max_seq_length, --doc_stride, etc. the same with fine-tuning parameters
* Keep the --model_name=albert_zh when do the ALBERT_zh tasks
* Change --task_name to the downstream task you want to do, only support afqmc, cmnli, csl, iflytek, tnews, lcqmc and wsc tasks
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

**Freeze pb model**
* Change --output_dir to the real path, and freeze script will convert checkpoint files to pb model file under this path
* Change --checkpoint_dir to the real path of checkpoint files, include 'checkpoint', 'ckpt.data', 'ckpt.index' and 'ckpt.meta'
* Rename --pb_model_file to the real pb model file name
* Change --predict_batch_size to the real batch size, or give 'None' for dynamic batch
* Keep other parameters the same as the previous step
```Bash
python3 xnlp_fmk.py \
    --output_dir=./save/model \
    --pb_model_file=./save/model/ALBERT_ZH_TINY_TNEWS_BatchSize_None.pb \
    --checkpoint_dir=./save/ckpt/albert_zh_tiny_tnews \
    --model_name=albert_zh \
    --task_name=tnews \
    --action_type=freeze

```

**Convert pb to om**
* Rename --om_model_file to the real om model file name
* Change the --soc_version, --in_nodes, --out_nodes according to the actual situation
* Add additional atc parameters if you need, e.g., --precision_mode
* Change --predict_batch_size to the real batch size, currently only support static batch size
* Keep other parameters the same as the previous step
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

**Run the inference**
* Change --output_dir to the real path and script will save the output bin file under this path
* Build the inference application and put it under current path, more details see: [xacl_fmk](./xacl_fmk/README.md)
* Keep other parameters the same as the previous step
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

**PostProcess**
* Change --output_dir to the real path and script will save the precision result file under this path
* Keep other parameters the same as the previous step
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

## Other Usages
**Convert pb to pbtxt**
* Change --output_dir to the real path, and convert script will convert pb model file to pbtxt model file under this path
* Rename --pb_model_file to the real pb model file name
* Keep other parameters the same as the previous step
```Bash
python3 xnlp_fmk.py \
    --output_dir=./save/model \
    --pb_model_file=./save/model/ALBERT_EN_BASE_CoLA_BatchSize_1.pb \
    --model_name=albert_en \
    --task_name=cola \
    --action_type=pbtxt

```

**Run the inference by pb model**
* Change the --in_nodes, --out_nodes according to the actual situation
* Keep other parameters the same as the previous step
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

## Reference

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
