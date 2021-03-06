-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [快速上手](#快速上手.md)
-   [迁移学习指导](#迁移学习指导.md)
-   [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Natural Language Processing**

**版本（Version）：1.1**

**修改时间（Modified） ：2021.7.17**

**大小（Size）：1.3G**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的BERT网络在Squad v1.1数据集上的finetune代码** 

<h2 id="概述.md">概述</h2>

BERT是一种与训练语言表示的方法，这意味着我们在大型文本语料库（如维基百科）上训练一个通用的”语言理解“模型，然后将该模型用于我们关心的下游NLP任务（如问答）。该工程提供了在Squad v1.1数据集上finetune的方法。

- 参考论文：

    [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805).

- 参考实现：

  https://github.com/google-research/bert

- 适配昇腾 AI 处理器的实现：
  
  https://gitee.com/ascend/modelzoo/tree/master/built-in/TensorFlow/Official/nlp/Bertsquad_ID0495_for_TensorFlow


- 通过Git获取对应commit\_id的代码方法如下：
  
    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    ```

## 默认配置<a name="section91661242121611"></a>

- 训练超参

  - train_batch_size: 32
  - learning_rate: 3e-5
  - num_train_epochs: 2.0
  - max_seq_length: 384
  - doc_stride:128


## 支持特性<a name="section1899153513554"></a>

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 是    |
| 混合精度  | 是    |
| 并行数据  | 是    |

## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

脚本已默认开启混合精度，设置precision_mode参数的脚本参考如下。

  ```
  run_config = NPURunConfig(
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        iterations_per_loop=FLAGS.iterations_per_loop,
        session_config=config,
        precision_mode="allow_mix_precision",
        keep_checkpoint_max=5)
  ```


<h2 id="训练环境准备.md">训练环境准备</h2>

1.  硬件环境准备请参见各硬件产品文档"[驱动和固件安装升级指南]( https://support.huawei.com/enterprise/zh/category/ai-computing-platform-pid-1557196528909)"。需要在硬件设备上安装与CANN版本配套的固件与驱动。
2.  宿主机上需要安装Docker并登录[Ascend Hub中心](https://ascendhub.huawei.com/#/detail?name=ascend-tensorflow-arm)获取镜像。

    当前模型支持的镜像列表如[表1](#zh-cn_topic_0000001074498056_table1519011227314)所示。

    **表 1** 镜像列表

    <a name="zh-cn_topic_0000001074498056_table1519011227314"></a>
    
    <table><thead align="left"><tr id="zh-cn_topic_0000001074498056_row0190152218319"><th class="cellrowborder" valign="top" width="47.32%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001074498056_p1419132211315"><a name="zh-cn_topic_0000001074498056_p1419132211315"></a><a name="zh-cn_topic_0000001074498056_p1419132211315"></a><em id="i1522884921219"><a name="i1522884921219"></a><a name="i1522884921219"></a>镜像名称</em></p>
    </th>
    <th class="cellrowborder" valign="top" width="25.52%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001074498056_p75071327115313"><a name="zh-cn_topic_0000001074498056_p75071327115313"></a><a name="zh-cn_topic_0000001074498056_p75071327115313"></a><em id="i1522994919122"><a name="i1522994919122"></a><a name="i1522994919122"></a>镜像版本</em></p>
    </th>
    <th class="cellrowborder" valign="top" width="27.16%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001074498056_p1024411406234"><a name="zh-cn_topic_0000001074498056_p1024411406234"></a><a name="zh-cn_topic_0000001074498056_p1024411406234"></a><em id="i723012493123"><a name="i723012493123"></a><a name="i723012493123"></a>配套CANN版本</em></p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="zh-cn_topic_0000001074498056_row71915221134"><td class="cellrowborder" valign="top" width="47.32%" headers="mcps1.2.4.1.1 "><a name="zh-cn_topic_0000001074498056_ul81691515131910"></a><a name="zh-cn_topic_0000001074498056_ul81691515131910"></a><ul id="zh-cn_topic_0000001074498056_ul81691515131910"><li><em id="i82326495129"><a name="i82326495129"></a><a name="i82326495129"></a>ARM架构：<a href="https://ascend.huawei.com/ascendhub/#/detail?name=ascend-tensorflow-arm" target="_blank" rel="noopener noreferrer">ascend-tensorflow-arm</a></em></li><li><em id="i18233184918125"><a name="i18233184918125"></a><a name="i18233184918125"></a>x86架构：<a href="https://ascend.huawei.com/ascendhub/#/detail?name=ascend-tensorflow-x86" target="_blank" rel="noopener noreferrer">ascend-tensorflow-x86</a></em></li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="25.52%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001074498056_p1450714271532"><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><em id="i72359495125"><a name="i72359495125"></a><a name="i72359495125"></a>20.2.0</em></p>
    </td>
    <td class="cellrowborder" valign="top" width="27.16%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001074498056_p18244640152312"><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><em id="i162363492129"><a name="i162363492129"></a><a name="i162363492129"></a><a href="https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software" target="_blank" rel="noopener noreferrer">20.2</a></em></p>
    </td>
    </tr>
    </tbody>
    </table>


<h2 id="快速上手.md">快速上手</h2>

- 数据集准备及预训练模型准备
1. 模型训练使用Squad v1.1数据集，参考源代码提供路径下载。
2. 预处理模型使用BERT-Base，Uncased，参考源代码提供路径下载。下载的文件夹中应该含有预处理模型，vocab.txt和bert_config.json。
3. 数据集和预处理模型下载完成后，放入模型目录下，在训练脚本中指定数据集和模型路径，可正常使用。

## 模型训练

- 单击“立即下载”，并选择合适的下载方式下载源码包。

- 启动训练之前，首先要配置程序运行相关环境变量。

  环境变量配置信息参见：

     [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

  将环境变量配置到test/train_*.sh中

- 单卡训练 

  启动单卡训练

  修改test/train_ID0495_Bert-Squad_performance_1p.sh中的data_path为squad v1.1数据集以及预训练模型的路径，然后执行：
  
  ```
  cd test
  bash train_ID0495_Bert-Squad_performance_1p.sh
  ```
  
- 8卡训练

  启动8卡训练

  修改test/train_ID0495_Bert-Squad_performance_1p.sh中的data_path为squad v1.1数据集以及预训练模型的路径，然后执行：
  
  ```
  cd test
  bash train_ID0495_Bert-Squad_performance_8p.sh
  ```
  
  

<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码<a name="section08421615141513"></a>

```
└─Bertsquad_for_TensorFlow
    ├─configs
    |     ├─rank_table_8p.json
    ├─test
    |     ├─train_ID0495_Bert-Squad_performance_1p.sh
    |     ├─train_ID0495_Bert-Squad_performance_8p.sh
    |     ├─train_ID0495_Bert-Squad_full_1p.sh
    |     ├─train_ID0495_Bert-Squad_full_8p.sh
    |     ├─train_ID3082_BertLarge-Squad_performance_1p.sh
    |     └─train_ID3082_BertLarge-Squad_performance_8p.sh
    |     ├─train_ID3082_BertLarge-Squad_full_1p.sh
    |     └─train_ID3082_BertLarge-Squad_full_8p.sh
    ├─utils
    |   ├─create_glue_data.py
    |   ├─create_pretraining_data.py
    |   ├─create_squad_data.py
    |   └─utils.py
    ├─CONTRIBUTING.md
    ├─create_pretraining_data.py
    ├─evaluate-v1.1.py
    ├─extract_features.py
    ├─gpu_environment.py
    ├─LICENSE
    ├─modeling.py
    ├─modeling_test.py
    ├─multilingual.md
    ├─optimization.py
    ├─optimization_test.py
    ├─README.md
    ├─run.sh
    ├─run_classifier.py
    ├─run_classifier_with_tfhub.py
    ├─run_pretraining.py
    ├─run_squad.py
    ├─tokenization.py
    └─tokenization_test.py
```

## 脚本参数<a name="section6669162441511"></a>

```
python3 run_squad.py \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --do_train=True \
  --train_file=$SQUAD_DIR/train-v1.1.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/dev-v1.1.json \
  --train_batch_size=32 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=./output
```

## 训练过程<a name="section1589455252218"></a>

1.  通过“模型训练”中的训练指令启动单卡训练和8卡训练。

2.  训练日志及结果见test/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log。



## 推理/验证过程<a name="section1465595372416"></a>

```
python3 evaluate-v1.1.py dataset/dev-v1.1.json .test/output/${ASCEND_DEVICE_ID}/predictions.json

##predict result for 1p:
{"exact_match": 79.87701040681173, "f1": 87.42429097480438}

##predict result for 8p:
{"exact_match": 79.2620624408704, "f1": 86.8726952549086}
```
