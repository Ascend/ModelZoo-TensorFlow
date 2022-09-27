<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Natural Language Processing** 

**版本（Version）：1.1**

**修改时间（Modified） ：2021.05.26**

**大小（Size）：1M**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的Bert下游任务学习算法训练代码** 

<h2 id="概述.md">概述</h2>

使用BERT进行命名实体识别。 

- 参考论文：

    https://arxiv.org/abs/1810.04805

- 参考实现：

   https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT

- 适配昇腾 AI 处理器的实现：
  
  
  https://gitee.com/ascend/modelzoo/tree/master/built-in/TensorFlow/Research/nlp/Bert_Nv_Finetune_MPRC_ID1641_for_TensorFlow


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

  - Batch size: 32
  - Learning rate(LR): 1e-6
  - Optimizer: AdamOptimizer
  - Warmup_proportion:0.1
  - Train epoch: 12
  - Max_seq_length:128


## 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 否       |
| 混合精度   | 是       |
| 并行数据   | 否       |

## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

脚本已默认开启混合精度，设置precision_mode参数的脚本参考如下。

  ```
   config = tf.ConfigProto()
   custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
   custom_op.name = "NpuOptimizer"
   custom_op.parameter_map["use_off_line"].b = True
   if FLAGS.precision_mode is not None:
          custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes(FLAGS.precision_mode)
   config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
   session_config = npu_config_proto(config_proto=config)
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

- 数据集准备
1. 该下游任务，支持两种数据集（MRPC和MNLI），用户自行下载并处理，也可以参考源 https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT 中的数据集说明下载

2. 预训练模型准备，预训练后的bert模型，可参考源链接下载。


## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。

- 启动训练之前，首先要配置程序运行相关环境变量。

  环境变量配置信息参见：

     [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)

- 单卡训练 

  1. 配置训练参数。

     首先在脚本test/train_mrpc_full_1p.sh中，配置MRPC训练数据集和预训练模型路径，请用户根据实际路径配置，数据集和预训练模型相关参数如下所示：

     ```
      --data_dir=../data     //数据集路径
      --vocab_file=$bertmodelpath/vocab.txt \
      --bert_config_file=$bertmodelpath/bert_config.json \
      --init_checkpoint=$bertmodelpath/bert_model.ckpt \
     ```

     如果是MNLI数据集test/train_MNLI_full_1p.sh中，配置MN,LI训练数据集和预训练模型路径，请用户根据实际路径配置，数据集和预训练模型相关参数如下所示：

     ```
      --data_dir=../data     //数据集路径
      --vocab_file=$bertmodelpath/vocab.txt \
      --bert_config_file=$bertmodelpath/bert_config.json \
      --init_checkpoint=$bertmodelpath/bert_model.ckpt \
     ```

  2. 启动训练。

     启动单卡训练 （脚本为test/train_mrpc_full_1p.sh或者train_MNLI_full_1p） 

     ```
     bash train_mrpc_full_1p.sh  或者  bash train_mnli_full_1p
     ```


<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码<a name="section08421615141513"></a>

```
BERT-NER
|____ run_classifier.py	    # 下游任务脚本
|____ conlleval.pl		    # eval code
├── test                    # 训练脚本所在目录
│    ├──train_mrpc_full_1p.sh           #mrpc数据集，单p训练脚本
│    ├──train_mnli_full_1p.sh           #mnli数据集，单p训练脚本

```

## 脚本参数<a name="section6669162441511"></a>

```
 --task_name=$task_name \               下游任务名称，支持mnli和mrpc
--precision_mode=$precision_mode \      精度模型，默认allow_mix_precision
--do_train=True   \                     是否执行训练，默认True
--do_eval=True   \                      是否执行eval，默认True
--data_dir=${data_path}/Glue/$task_name \         数据集路径
--vocab_file=$bertmodelpath/vocab.txt \
--bert_config_file=$bertmodelpath/bert_config.json \
--init_checkpoint=$bertmodelpath/bert_model.ckpt \
--train_batch_size=$train_batch_size \
--learning_rate=$learning_rate   \      初始学习率
--num_train_epochs=$num_train_epochs    \    训练epoch数
--output_dir=./output/result_dir        保存路径
```

## 训练过程<a name="section1589455252218"></a>

NA
