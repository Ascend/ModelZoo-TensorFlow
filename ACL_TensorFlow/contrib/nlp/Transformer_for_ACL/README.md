# <font face="微软雅黑">

# Transformer Translation Inference for TensorFlow

***
This repository provides a script and recipe to Inference the Transformer Translation Inference

* [x] Transformer Translation Inference, Based on [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor) 

***

## Notice
**This sample only provides reference for you to learn the Ascend software stack and is not for commercial purposes.**

Before starting, please pay attention to the following adaptation conditions. If they do not match, may leading in failure.

| Conditions | Need |
| --- | --- |
| CANN Version | >=5.0.3 |
| Chip Platform| Ascend310/Ascend710 |
| 3rd Party Requirements| Please follow the 'requirements.txt' |

## Quick Start Guide

### 1. Clone the respository
```shell
git clone https://gitee.com/ascend/ModelZoo-TensorFlow.git
cd Modelzoo-TensorFlow/ACL_TensorFlow/contrib/nlp/Transformer_for_ACL
```

### 2. Download and preprocess the dataset

Download the dataset by yourself, more details see: [data](./data/WMT32K/README.md)
Obtain the vocab.translate_ende_wmt32k.subwords by yourself, more details see: [config](./config/README.md)

### 3. Obtain the pb model

Obtain the pb model in Ascend ModelZoo: [Transformer](https://www.hiascend.com/zh/software/modelzoo/detail/1/4aa974b3f2fb4e02a84abbf16b56f032)

### 4. Build the program
Build the inference application, more details see: [xacl_fmk](./xacl_fmk/README.md)
Put xacl to the current dictory.

### 5. Offline Inference

**Transformer**
***
* Transformer use transformer for model_name parameter, translation for task_name
* Change the parameters for different tasks
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
```Bash
python3 xnlp_fmk.py \
    --data_dir=./data/WMT32K \
    --output_dir=./data/WMT32K \
    --vocab_file=./config/translate_ende_wmt32k/vocab.translate_ende_wmt32k.32768.subwords \
    --model_name=transformer \
    --task_name=translation \
    --action_type=preprocess

```

**Convert pb to om**
```Bash
python3 xnlp_fmk.py \
    --output_dir=./save/model \
    --vocab_file=./config/translate_ende_wmt32k/vocab.translate_ende_wmt32k.32768.subwords \
    --pb_model_file=./save/model/TRANSFORMER_TRANSLATION_BatchSize_1.pb \
    --om_model_file=./save/model/TRANSFORMER_TRANSLATION_BatchSize_1.om \
    --predict_batch_size=1 \
    --soc_version="Ascend310" \
    --in_nodes="\"input:1,128\"" \
    --out_nodes="\"transformer/strided_slice_11:0\"" \
    --precision_mode="allow_mix_precision" \
    --op_select_implmode="high_precision" \
    --model_name=transformer \
    --task_name=translation \
    --action_type=atc

```

**Run the inference**
```Bash
python3 xnlp_fmk.py \
    --data_dir=./data/WMT32K \
    --output_dir=./save/output \
    --om_model_file=./save/model/TRANSFORMER_TRANSLATION_BatchSize_1.om \
    --predict_batch_size=1 \
    --model_name=transformer \
    --task_name=translation \
    --action_type=npu

```

**PostProcess**
```Bash
python3 xnlp_fmk.py \
    --data_dir=./data/WMT32K \
    --output_dir=./save/output \
    --vocab_file=./config/translate_ende_wmt32k/vocab.translate_ende_wmt32k.32768.subwords \
    --om_model_file=./save/model/TRANSFORMER_TRANSLATION_BatchSize_1.om \
    --predict_batch_size=1 \
    --model_name=transformer \
    --task_name=translation \
    --action_type=postprocess

```

## Other Usages
**Convert pb to pbtxt**
```Bash
python3 xnlp_fmk.py \
    --output_dir=./save/model \
    --pb_model_file=./save/model/BIOBERT_BASE_RE_BatchSize_None.pb \
    --model_name=biobert \
    --task_name=re \
    --action_type=pbtxt

```

**Run the inference by pb model**
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
