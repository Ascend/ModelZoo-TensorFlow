# <font face="微软雅黑">
English|[中文](README.md)

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
| Chip Platform| Ascend310/Ascend310P3 |
| 3rd Party Requirements| Please follow the 'requirements.txt' |

## Quick Start Guide

### 1. Clone the respository
```shell
git clone https://gitee.com/ascend/ModelZoo-TensorFlow.git
cd Modelzoo-TensorFlow/ACL_TensorFlow/contrib/nlp/Transformer_for_ACL
```

### 2. Download and preprocess the dataset

Download the dataset and save it to the data directory. If the directory does not exist, create it in the root directory of the subproject.

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
 Please follow the [guide](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719) to set the envs

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
atc
    --framework 3 \
    --soc_version "Ascend310" \
    --model ./save/model/TRANSFORMER_TRANSLATION_BatchSize_1.pb \
    --input_shape "input:1,128" \
    --output ./save/model/TRANSFORMER_TRANSLATION_BatchSize_1 \
    --out_nodes "transformer/strided_slice_11:0" \
    --op_select_implmode "high_precision" \
    --precision_mode "allow_mix_precision"
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
