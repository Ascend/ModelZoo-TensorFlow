中文|[English](README_EN.md)
# <font face="微软雅黑">

# LSTM TensorFlow离线推理

***
此链接提供LSTM TensorFlow模型在NPU上离线推理的脚本和方法

* [x] LSTM 推理, 基于 [Sentiment Analysis with Word Embedding](https://github.com/HqWei/Sentiment-Analysis)

***

## Notice
**此案例仅为您学习Ascend软件栈提供参考，不用于商业目的。此案例仅为您学习Ascend软件栈提供参考，不用于商业目的。**

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
cd Modelzoo-TensorFlow/ACL_TensorFlow/contrib/nlp/LSTM_for_ACL
```

### 2. 下载数据集和预处理

请自行下载数据集, 并放在data目录下(若目录不存在请在子项目根目录下自行创建),

### 3. 获取checkpoint文件或pb模型

[pb模型下载链接](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/2022-12-12_tf/LSTM_for_ACL/LSTM_IMDB_BatchSize_24.pb)

### 4. 编译程序
编译推理程序, 详情见: [xacl_fmk](./xacl_fmk/README.md)
将xacl放到当前文件夹

### 5. 离线推理

**LSTM**
***
* LSTM 将lstm用于modelname参数，imdb用于task_name参数
* LSTM 在 LSTM_for_ACL 中： max_seq_len=250
* LSTM 在 LSTM_for_ACL 中：使用静态batch，将predict_batch_size=24设置为输入参数
***
**环境变量设置**

  请参考[说明](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719)，设置环境变量

**预处理**
* --data_dir：每个任务数据集的实际路径, 并且确保 **predict** 文件在当前路径下
* --output_dir 的传参与--data_dir相同, 预处理脚本会将文本转换为该路径下的bin文件
* --model_name：当进行LSTM任务，参数为 lstm
* --task_name：任务名, 仅支持imdb任务
* 更多数据集和任务详细信息，如下载链接，请参阅自述文件。每个数据集路径中的README.md
```Bash
python3 xnlp_fmk.py \
    --data_dir=./data/IMDB \
    --output_dir=./data/IMDB \
    --model_name=lstm \
    --task_name=imdb \
    --action_type=preprocess

```

**冻结Pb模型**
* --output_dir：在此路径下，冻结脚本会把checkpoint文件转成Pb模型
* --checkpoint_dir：checkpoint文件, 包括 'checkpoint', 'ckpt.data', 'ckpt.index' 和 'ckpt.meta'
* --pb_model_file：pb模型文件名
* --predict_batch_size：与实际批量相比，仅测试了24个
* 保持其他参数与上一步相同
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

**Pb模型转换为om模型 **
* --om_model_file：om模型名
* --soc_version, --in_nodes, --out_nodes ：根据实际情况传参
* 添加额外需要的atc参数，例如： --precision_mode
* --predict_batch_size ：实际batch, 当前仅支持静态batch
* 保持其他参数与上一步相同
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

**运行推理**
* -output_dir ：脚本将在该路径下保存输出bin文件
* 构建推理应用程序并将其置于当前路径下，详情见: [xacl_fmk](./xacl_fmk/README.md)
* 保持其他参数与上一步相同
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

**后处理**
* --output_dir:脚本将在该路径下保存精度结果文件
* 保持其他参数与上一步相同
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

## 其他用途
**将pb模型转换为pbtxt**
* --output_dir：在此路径下，脚本会将pb模型转为pbtxt模型文件
* --pb_model_file：pb模型文件名
* 保持其他参数与上一步相同
```Bash
python3 xnlp_fmk.py \
    --output_dir=./save/model \
    --pb_model_file=./save/model/LSTM_IMDB_BatchSize_24.pb \
    --model_name=lstm \
    --task_name=imdb \
    --action_type=pbtxt

```

**通过pb模型运行推理**
* --in_nodes, --out_nodes：根据实际情况传参
* 保持其他参数与上一步相同
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
