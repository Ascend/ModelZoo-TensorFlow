# <font face="微软雅黑">

# BioBERT DownStream Inference for TensorFlow

***
This repository provides a script and recipe to Inference the BioBERT DownStream Inference

* [x] BioBERT DownStream Inference, based on [BioBERT](https://github.com/dmis-lab/biobert) 

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
cd Modelzoo-TensorFlow/ACL_TensorFlow/contrib/nlp/BioBERT_for_ACL
```

### 2. Download and preprocess the dataset

Download the dataset by yourself, more details see: data
Download the vocab.txt and bert_config.json by yourself, more details see: [config](./config/README.md)

### 3. Obtain the fine-tuned checkpoint files or pb model

Obtain the fine-tuned checkpoint files or pb model, more details see: [ckpt](./save/ckpt/README.md) or [models](./save/model/README.md)

### 4. Build the program
Build the inference application, more details see: [xacl_fmk](./xacl_fmk/README.md)
Put xacl to the current dictory.

### 5. Offline Inference

**BioBERT**
***
* BioBERT use biobert for model_name parameter, each downstream task name for task_name
* BioBERT support ner and re tasks
* Change the parameters for different tasks
* Only BioBERT Base and BioBERT Large has been tested
***
**Configure the env**
```
export install_path=/usr/local/Ascend
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg:$PYTHONPATH
export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
export ASCEND_OPP_PATH=${install_path}/opp
```

**PreProcess**
* Change --data_dir to the real path of each downstream task dataset, and make sure the **predict** file under the path such as 'dev.tsv'
* Change --output_dir to the same with --data_dir, and preprocess script will convert text to bin files under this path
* Keep the --vocab_file, --bert_config_file, --do_lower_case, --max_seq_length, --doc_stride, etc. the same with fine-tuning parameters
* Keep the --model_name=biobert when do the BioBERT tasks
* Change --task_name to the downstream task you want to do, only support ner and re tasks
* More datasets and tasks details like download link see README.md in each datasets' path
```Bash
python3 xnlp_fmk.py \
    --data_dir=./data/NCBI \
    --output_dir=./data/NCBI \
    --vocab_file=./config/biobert_base/vocab.txt \
    --bert_config_file=./config/biobert_base/bert_config.json \
    --do_lower_case=False \
    --model_name=biobert \
    --task_name=ner \
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
    --bert_config_file=./config/biobert_base/bert_config.json \
    --pb_model_file=./save/model/BIOBERT_BASE_NER_BatchSize_None.pb \
    --checkpoint_dir=./save/ckpt/biobert_base_ner \
    --model_name=biobert \
    --task_name=ner \
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
    --pb_model_file=./save/model/BIOBERT_BASE_NER_BatchSize_None.pb \
    --om_model_file=./save/model/BIOBERT_BASE_NER_BatchSize_1.om \
    --predict_batch_size=1 \
    --soc_version="Ascend310" \
    --in_nodes="\"input_ids:1,128;input_mask:1,128;segment_ids:1,128\"" \
    --out_nodes="\"logits:0\"" \
    --model_name=biobert \
    --task_name=ner \
    --action_type=atc

```

**Run the inference**
* Change --output_dir to the real path and script will save the output bin file under this path
* Build the inference application and put it under current path, more details see: [xacl_fmk](./xacl_fmk/README.md)
* Keep other parameters the same as the previous step
```Bash
python3 xnlp_fmk.py \
    --data_dir=./data/NCBI \
    --output_dir=./save/output \
    --om_model_file=./save/model/BIOBERT_BASE_NER_BatchSize_1.om \
    --predict_batch_size=1 \
    --model_name=biobert \
    --task_name=ner \
    --action_type=npu

```

**PostProcess**
* Change --output_dir to the real path and script will save the precision result file under this path
* Keep other parameters the same as the previous step
```Bash
python3 xnlp_fmk.py \
    --data_dir=./data/NCBI \
    --output_dir=./save/output \
    --vocab_file=./config/biobert_base/vocab.txt \
    --om_model_file=./save/model/BIOBERT_BASE_NER_BatchSize_1.om \
    --predict_batch_size=1 \
    --do_lower_case=False \
    --model_name=biobert \
    --task_name=ner \
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
    --pb_model_file=./save/model/BIOBERT_BASE_RE_BatchSize_None.pb \
    --model_name=biobert \
    --task_name=re \
    --action_type=pbtxt

```

**Run the inference by pb model**
* Change the --in_nodes, --out_nodes according to the actual situation
* Keep other parameters the same as the previous step
```Bash
python3 xnlp_fmk.py \
    --data_dir=./data/GAD \
    --output_dir=./save/output \
    --pb_model_file=./save/model/BIOBERT_BASE_RE_BatchSize_None.pb \
    --predict_batch_size=1 \
    --in_nodes="\"input_ids:1,128;input_mask:1,128;segment_ids:1,128\"" \
    --out_nodes="\"logits:0\"" \
    --model_name=biobert \
    --task_name=re \
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
