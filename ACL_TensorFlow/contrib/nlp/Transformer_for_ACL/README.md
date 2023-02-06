# <font face="微软雅黑">
中文|[English](README_EN.md)

# Transformer Translation TensorFlow离线推理

***
此链接提供Transformer Translation TensorFlow模型在NPU上离线推理的脚本和方法

* [x] Transformer Translation 推理, 基于 [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor) 

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
cd Modelzoo-TensorFlow/ACL_TensorFlow/contrib/nlp/Transformer_for_ACL
```

### 2. 下载数据集和预处理

请自行下载数据集, 并放在data目录下(若目录不存在请在子项目根目录下自行创建),

请自行获取vocab.translate_ende_wmt32k.subwords, 更多详细信息请参见: [config](./config/README.md)

### 3. 获取pb模型

在Ascend ModelZoo中获取pb模型: [Transformer](https://www.hiascend.com/zh/software/modelzoo/detail/1/4aa974b3f2fb4e02a84abbf16b56f032)

### 4. 编译程序
构建推理应用程序,并将其放至当前路径下，更多详细信息请参见: [xacl_fmk](./xacl_fmk/README.md)


### 5. 离线推理

**Transformer**
***
* Transformer将Transformer用于model_name参数，转换为task_name
* 更改不同任务的参数
***
**环境变量设置**
请参考[说明](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719)，设置环境变量

**预处理**
```Bash
python3 xnlp_fmk.py \
    --data_dir=./data/WMT32K \
    --output_dir=./data/WMT32K \
    --vocab_file=./config/translate_ende_wmt32k/vocab.translate_ende_wmt32k.32768.subwords \
    --model_name=transformer \
    --task_name=translation \
    --action_type=preprocess

```

**Pb模型转换为om模型**
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

**运行推理**
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

**后期处理**
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

## 其他用途
**将pb转换为pbtxt**
```Bash
python3 xnlp_fmk.py \
    --output_dir=./save/model \
    --pb_model_file=./save/model/BIOBERT_BASE_RE_BatchSize_None.pb \
    --model_name=biobert \
    --task_name=re \
    --action_type=pbtxt

```

**通过pb模型运行推理**
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
