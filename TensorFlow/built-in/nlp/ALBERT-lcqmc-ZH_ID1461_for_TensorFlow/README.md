- [基本信息](#基本信息.md)
- [概述](#概述.md)
- [训练环境准备](#训练环境准备.md)
- [快速上手](#快速上手.md)
- [迁移学习指导](#迁移学习指导.md)
- [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>
**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Natural Language Processing**

**版本（Version）：1.1**

**修改时间（Modified） ：2021.04.16**

**大小（Size）：1M**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的Albert_ZH代码**

<h2 id="概述.md">概述</h2>
  在预训练自然语言表示时增加模型大小通常会提高下游任务的性能。 但是，由于GPU / TPU内存的限制和更长的训练时间，在某些时候，进一步的模型增加变得更加困难。 为了解决这些问题，我们提出了两种参数减少技术，以降低内存消耗并提高BERT的训练速度。 全面的经验证据表明，与原始BERT相比，我们提出的方法所导致的模型可扩展性更好。 我们还使用了一个自我监督的损失，该损失集中于对句子之间的连贯性进行建模，并表明它始终可以通过多句子输入来帮助下游任务。 因此，我们的最佳模型在GLUE，RACE和\ squad基准上建立了最新的技术成果，而与BERT-large相比，参数更少。    

- 参考论文：
    https://paperswithcode.com/paper/albert-a-lite-bert-for-self-supervised

- 参考实现：
  https://github.com/brightmart/albert_zh


- 适配昇腾 AI 处理器的实现：   
    https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/built-in/nlp/ALBERT-lcqmc-ZH_ID1461_for_TensorFlow

- 通过Git获取对应commit\_id的代码方法如下：
     
    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
        
    ```


## 默认配置【深加工】

-   网络结构
    - 6层，隐层大小1024，残差和层归一化使用Pre-norm方式。

    - 使用方差缩放（Variance Scaling）的均匀分布进行参数初始化。

    - bias和层归一化的beta初始化为0，层归一化的gamma初始化为1。

-   训练数据集预处理（当前代码以wmt16 en-de训练集为例，仅作为用户参考示例）

    - Transformer采用翻译文本输入，padding到最大长度128；

-   训练超参（单卡）：
    -   Learning rate(LR): 2.0 
    -   Batch size: 40
    -   Label smoothing: 0.1
    -   num_units: 1024
    -   num_layers: 6
    -   attention.num_heads: 16 


## 支持特性【深加工】

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 是    |
| 混合精度  | 是    |
| 数据并行  | 是    |

## 混合精度训练【深加工】

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度【深加工】


```
run_config = NPURunConfig( 
        hcom_parallel = True, 
        enable_data_pre_proc = True, 
        keep_checkpoint_max=5, 
        save_checkpoints_steps=self.args.nsteps_per_epoch, 
        session_config = self.sess.estimator_config, 
        model_dir = self.args.log_dir, 
        iterations_per_loop=self.args.iterations_per_loop, 
        precision_mode='allow_mix_precision' 
     ）
```

<h2 id="训练环境准备.md">训练环境准备</h2>

-  硬件环境和运行环境准备请参见《[CANN软件安装指南](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=installation-update)》
-  运行以下命令安装依赖。
```
pip3 install requirements.txt
```
说明：依赖配置文件requirements.txt文件位于模型的根目录

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
## 数据集准备<a name="section361114841316"></a>

1. 模型使用数据集 LCQMC，参考源代码提供路径下载。
2. 模型使用预训练模型 albert_tiny_zh，参考源代码提供路径下载。
## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。

- 开始训练。

    1. 启动训练之前，首先要配置程序运行相关环境变量。

       环境变量配置信息参见：

          [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)
    2. 配置configs/transformer_big.yml中的网络参数，请用户根据自己需求进行配置；
    
    3. 配置train-ende.sh中`DATA_PATH`训练参数（脚本位于Transformer_for_TensorFlow目录下），请用户根据实际数据集路径进行配置，确保`DATA_PATH`下存在已准备的数据集，如下所示（仅供参考）：
               
        ```
                DATA_PATH="../wmt-ende"
                VOCAB_SOURCE=${DATA_PATH}/vocab.share
                VOCAB_TARGET=${DATA_PATH}/vocab.share
                TRAIN_FILES=${DATA_PATH}/concat128/train.l128.tfrecord-001-of-016
                for idx in {002..016}; do
                  TRAIN_FILES="${TRAIN_FILES},${DATA_PATH}/concat128/train.l128.tfrecord-${idx}-of-016"
                done
                DEV_SOURCES=${DATA_PATH}/dev2010.tok.zh
                DEV_TARGETS=${DATA_PATH}/dev2010.tok.en
        ```

    4.  单卡训练。
    
    单卡训练指令如下（脚本位于Transformer_for_TensorFlow/transformer_1p目录下）：
    
    ```
            bash transformer_1p/transformer_main_1p.sh
    ```


    5.  8卡训练。
        
        8卡训练指令如下（脚本位于Transformer_for_TensorFlow/transformer_8p目录下）：
        ```
            bash transformer_8p/transformer_8p.sh 
        ```


-  开始推理。

    1. 配置inference.sh中的参数，'DATA_PATH', 'TEST_SOURCES', 'MODEL_DIR' 和'output'请用户设置为自己的路径；
       
       
        ```
        DATA_PATH="../wmt-ende"
        TEST_SOURCES="${DATA_PATH}/tfrecord/newstest2014.l128.tfrecord"
        MODEL_DIR="file://PATH_TO_BE_CONFIGURED"
        .    .    .
        output: ./output-0603
        ```

    2. 推理生成翻译结果，该脚本生成的为词id的表现形式；

        bash inference.sh

    3. 生成翻译结果的文本形式。

        REF_DATA：newstest2014.tok.de

        EVAL_OUTPUT：inference.sh的生成的output文件

        VOCAB_FILE：vocab.share

        bash scripts/process_output.sh REF_DATA EVAL_OUTPUT VOCAB_FILE

        例如：

        bash scripts/process_output.sh /data/wmt-ende/newstest2014.tok.de output-0603 /data/wmt-ende/vocab.share

    4. 测试BLEU值。

        4.1 multi-bleu.perl脚本可以通过 [perl script](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl) 下载；
    
        4.2 执行指令：

            perl multi-bleu.perl REF_DATA.forbleu < EVAL_OUTPUT.forbleu
            
            例如：
            
            perl multi-bleu.perl /data/wmt-ende/newstest2014.tok.de.forbleu < output-0603.forbleu

<h2 id="高级参考.md">高级参考</h2>
## 脚本和示例代码<a name="section08421615141513"></a>

```

└─Transformer 
    ├─README.md 
    ├─configs
        ├─transformer_big.yml
        └─...
    ├─transformer_1p 
        |─new_rank_table_1p.json 
        └─transformer_main_1p.sh 
    ├─transformer_8p 
        ├─transformer_8p.sh 
        ├─transformer_p1 
            ├─new_rank_table_8p.json 
            └─transformer_main_p1.sh 
        ├─transformer_p2 
            ├─new_rank_table_8p.json 
            └─transformer_main_p2.sh 
        ├─... 
    ├─noahnmt 
        ├─bin 
            ├─train.py 
            ├─infer.py 
            └─... 
        ├─data 
            ├─text_input_pipeline.py 
            ├─input_pipeline.py 
            └─... 
        ├─attentions 
            ├─attention.py 
            ├─sum_attention.py 
            └─... 
        ├─decoders 
            ├─decoder.py 
            ├─attention_decoder.py 
            └─... 
        ├─encoders 
            ├─encoder.py 
            ├─transformer_encoder.py 
            └─... 
        ├─hooks 
            ├─metrics_hook.py 
            ├─train_hooks.py 
            └─... 
        ├─inference 
            ├─beam_search.py 
            ├─inference.py 
            └─... 
        ├─layers 
            ├─nmt_estimator.py 
            ├─rnn_cell.py 
            └─... 
        ├─metrics 
            ├─multi_bleu.py 
            ├─metric_specs.py 
            └─... 
        ├─utils 
            ├─trainer_lib.py 
            └─... 
        ├─models 
            ├─seq2seq_model.py 
            └─... 
            ├─__init__.py 
            ├─configurable.py 
            └─graph_module.py 
    ├─inference.sh 
    ├─new_rank_table_8p.json 
    ├─create_training_data_concat.py 
    └─train-ende.sh
```


## 脚本参数<a name="section6669162441511"></a>

```
#Training 
--config_paths                 Path for training config file. 
--model_params                 Model parameters for training the model, like learning_rate,dropout_rate. 
--metrics                      Metrics for model eval. 
--input_pipeline_train         Dataset input pipeline for training. 
--input_pipeline_dev           Dataset input pipeline for dev. 
--train_steps                  Training steps, default is 300000. 
--keep_checkpoint_max          Number of the checkpoint kept in the checkpoint dir. 
--batch_size                   Batch size for training, default is 40. 
--model_dir                    Model dir for saving checkpoint 
--use_fp16                     Whether to use fp16, default is True. 
```
## 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡训练。
将训练脚本（train_full_1p.sh）中的data_path设置为训练数据集的路径。具体的流程参见“模型训练”的示例。