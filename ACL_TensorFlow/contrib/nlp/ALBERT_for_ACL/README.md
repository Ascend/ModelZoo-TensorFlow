# <font face="微软雅黑">

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

获取训练好的checkpoint文件，或者pb模型, 更多详情见: [ckpt](./save/ckpt/README.md) or [models](./save/model/README.md)

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

**冻结pb模型**
* Change --output_dir to the real path, and freeze script will convert checkpoint files to pb model file under this path
* Change --checkpoint_dir to the real path of checkpoint files, include 'checkpoint', 'ckpt.data', 'ckpt.index' and 'ckpt.meta'
* 重命名 --pb_model_file to the real pb model file name
* 变化  --predict_batch_size to the real batch size, or give 'None' for dynamic batch
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
* Rename --om_model_file to the real om model file name
* Change the --soc_version, --in_nodes, --out_nodes according to the actual situation
* Add additional atc parameters if you need, e.g., --precision_mode
* Change --predict_batch_size to the real batch size, currently only support static batch size
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

**Run the inference**
* Change --output_dir to the real path and script will save the output bin file under this path
* Build the inference application and put it under current path, more details see: [xacl_fmk](./xacl_fmk/README.md)
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
* 修改--output_dir入参为绝对路径， script will save the precision result file under this path脚本将保存精度结果文件至此
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
* ALBERT_zh use albert_zh for model_name parameter, each downstream task name for task_name
* ALBERT_zh support afqmc, cmnli, csl, iflytek, tnews and wsc tasks
* Change the parameters for different tasks
* Only ALBERT_zh Tiny has been tested
***
**PreProcess**
* Change --data_dir to the real path of each downstream task dataset, and make sure the **predict** file under the path such as 'dev.tsv'
* Change --output_dir to the same with --data_dir, and preprocess script will convert text to bin files under this path
* Keep the --vocab_file, --bert_config_file, --do_lower_case, --max_seq_length, --doc_stride, etc. the same with fine-tuning parameters
* Keep the --model_name=albert_zh when do the ALBERT_zh tasks
* Change --task_name to the downstream task you want to do, only support afqmc, cmnli, csl, iflytek, tnews and wsc tasks
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
* Change --output_dir to the real path, and freeze script will convert checkpoint files to pb model file under this path
* Change --checkpoint_dir to the real path of checkpoint files, include 'checkpoint', 'ckpt.data', 'ckpt.index' and 'ckpt.meta'
* Rename --pb_model_file to the real pb model file name
* Change --predict_batch_size to the real batch size, or give 'None' for dynamic batch
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
* Rename --om_model_file to the real om model file name
* Change the --soc_version, --in_nodes, --out_nodes according to the actual situation
* Add additional atc parameters if you need, e.g., --precision_mode
* Change --predict_batch_size to the real batch size, currently only support static batch size
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
* Change --output_dir to the real path and script will save the output bin file under this path
* Build the inference application and put it under current path, 更多详情见: [xacl_fmk](./xacl_fmk/README.md)
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
* Change --output_dir to the real path and script will save the precision result file under this path
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
**pb模型转换为pbtxt**
* --output_dir改为相对路径, 脚本将pb模型转换成pbtxt，并保存至此路径下
*--pb_model_file入参改为实际模型名称
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
* 依据实际情况，修改--in_nodes, --out_nodes 
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
