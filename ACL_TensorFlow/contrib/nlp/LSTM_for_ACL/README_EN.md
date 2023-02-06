English|[中文](README.md)
# <font face="微软雅黑">

# LSTM Inference for TensorFlow

***
This repository provides a script and recipe to Inference the LSTM Inference

* [x] LSTM Inference, based on [Sentiment Analysis with Word Embedding](https://github.com/HqWei/Sentiment-Analysis)

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
cd Modelzoo-TensorFlow/ACL_TensorFlow/contrib/nlp/LSTM_for_ACL
```

### 2. Download and preprocess the dataset

Download the dataset and save it to the data directory. If the directory does not exist, create it in the root directory of the subproject.

### 3. Obtain the fine-tuned checkpoint files or pb model

[pb download link](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/2022-12-12_tf/LSTM_for_ACL/LSTM_IMDB_BatchSize_24.pb)

### 4. Build the program
Build the inference application, more details see: [xacl_fmk](./xacl_fmk/README.md)
Put xacl to the current dictory.

### 5. Offline Inference

**LSTM**
***
* LSTM use lstm for model_name parameter, imdb for task_name
* LSTM in LSTM_for_ACL max_seq_len=250
* LSTM in LSTM_for_ACL use static batch size, set predict_batch_size=24 as input parameter
***
**Configure the env**

  Please follow the [guide](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719) to set the envs


**PreProcess**
* Change --data_dir to the real path of each downstream task dataset, and make sure the **predict** file under the path
* Change --output_dir to the same with --data_dir, and preprocess script will convert text to bin files under this path
* Keep the --model_name=lstm when do the LSTM tasks
* Change --task_name to the task you want to do, only support imdb tasks
* More datasets and tasks details like download link see README.md in each datasets' path
```Bash
python3 xnlp_fmk.py \
    --data_dir=./data/IMDB \
    --output_dir=./data/IMDB \
    --model_name=lstm \
    --task_name=imdb \
    --action_type=preprocess

```

**Freeze pb model**
* Change --output_dir to the real path, and freeze script will convert checkpoint files to pb model file under this path
* Change --checkpoint_dir to the real path of checkpoint files, include 'checkpoint', 'ckpt.data', 'ckpt.index' and 'ckpt.meta'
* Rename --pb_model_file to the real pb model file name
* Change --predict_batch_size to the real batch size, only 24 has been tested
* Keep other parameters the same as the previous step
```Bash
python3 xnlp_fmk.py \
    --data_dir=./data/IMDB \
    --output_dir=./save/model \
    --pb_model_file=./save/model/LSTM_IMDB_BatchSize_24.pb \
    --checkpoint_dir=./save/ckpt/lstm_imdb \
    --predict_batch_size=24 \
    --model_name=lstm \
    --task_name=imdb \
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
    --data_dir=./data/IMDB \
    --output_dir=./save/model \
    --pb_model_file=./save/model/LSTM_IMDB_BatchSize_24.pb \
    --om_model_file=./save/model/LSTM_IMDB_BatchSize_24.om \
    --predict_batch_size=24 \
    --soc_version="Ascend310" \
    --in_nodes="\"input_ids:24,250\"" \
    --out_nodes="\"logits:0\"" \
    --model_name=lstm \
    --task_name=imdb \
    --action_type=atc

```

**Run the inference**
* Change --output_dir to the real path and script will save the output bin file under this path
* Build the inference application and put it under current path, more details see: [xacl_fmk](./xacl_fmk/README.md)
* Keep other parameters the same as the previous step
```Bash
python3 xnlp_fmk.py \
    --data_dir=./data/IMDB \
    --output_dir=./save/output \
    --om_model_file=./save/model/LSTM_IMDB_BatchSize_24.om \
    --predict_batch_size=24 \
    --model_name=lstm \
    --task_name=imdb \
    --action_type=npu

```

**PostProcess**
* Change --output_dir to the real path and script will save the precision result file under this path
* Keep other parameters the same as the previous step
```Bash
python3 xnlp_fmk.py \
    --data_dir=./data/IMDB \
    --output_dir=./save/output \
    --om_model_file=./save/model/LSTM_IMDB_BatchSize_24.om \
    --predict_batch_size=24 \
    --model_name=lstm \
    --task_name=imdb \
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
    --pb_model_file=./save/model/LSTM_IMDB_BatchSize_24.pb \
    --model_name=lstm \
    --task_name=imdb \
    --action_type=pbtxt

```

**Run the inference by pb model**
* Change the --in_nodes, --out_nodes according to the actual situation
* Keep other parameters the same as the previous step
```Bash
python3 xnlp_fmk.py \
    --data_dir=./data/GAD \
    --output_dir=./save/output \
    --pb_model_file=./save/model/LSTM_IMDB_BatchSize_24.pb \
    --predict_batch_size=24 \
    --in_nodes="\"input_ids:24,250\"" \
    --out_nodes="\"logits:0\"" \
    --model_name=lstm \
    --task_name=imdb \
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
