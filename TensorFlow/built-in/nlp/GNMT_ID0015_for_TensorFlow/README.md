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

**修改时间（Modified） ：2021.08.04**

**大小（Size）：624M**

**框架（Framework）：TensorFlow_1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的GNMT v2网络机器翻译网络训练代码**

<h2 id="概述.md">概述</h2>

GNMT v2 模型类似于谷歌的机器翻译网络GNMT模型。两种模型之间最重要的区别在于注意力机制。在GNMT v2模型中，解码器第一个 LSTM 层的输出进入注意力模块，然后重新加权的上下文在当前时间步与解码器中所有后续 LSTM 层的输入连接。

-   参考论文：

    https://arxiv.org/abs/1609.08144
-   参考实现：

    https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Translation/GNMT
-   适配昇腾 AI 处理器的实现：
    
    
     https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/built-in/nlp/GNMT_ID0015_for_TensorFlow
        

-   通过Git获取对应commit\_id的代码方法如下：
    
    
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    

## 默认配置<a name="section91661242121611"></a>

-   网络结构
    -   初始学习率为0.0005，使用Cosine learning rate 
    -   优化器：Adam 
    -   单卡batchsize：128 
    -   8卡batchsize：128*8 
    -   总Epoch数设置为6
    -   Weight decay为0.1
    -   Label smoothing参数为0.1

-   训练数据集：
    -   `scripts/wmt16_en_de.sh`自动下载和预处理训练和测试数据集
    -   原始数据使用[Moses](https://github.com/moses-smt/mosesdecoder)预处理 ，首先通过启动[Moses 标记器](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/tokenizer.perl) （标记器将文本分解为单个单词）
    -   然后通过启动 [clean-corpus-n.perl](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/training/clean-corpus-n.perl) 删除无效句子并按序列长度进行初始过滤
    -   丢弃所有不能被 latin-1 编码器解码的句子对
    -   使用 32,000 次合并操作（command ）构建共享词汇表`subword-nmt learn-bpe`，然后将生成的词汇表应用于训练、验证和测试语料库（command `subword-nmt apply-bpe`）
    
-   测试数据集：
    -   同训练数据集

## 支持特性<a name="section1899153513554"></a>

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 是    |
| 混合精度  | 是    |
| 数据并行  | 是    |


## 混合精度训练<a name="section168064817164"></a>

 混合精度训练昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

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

## 数据集准备<a name="section361114841316"></a>

1. 训练数据集可选用WMT16 English-German， 测试数据集可选用newstest2014 ，用户自行准备好数据集
2. 数据集的下载及处理，请用户参考”概述--> 参考实现“ 开源代码处理
3. 数据集处理后放在模型目录下，在训练脚本中指定数据集路径，可正常使用

## 模型训练<a name="section715881518135"></a>

-  单击“立即下载”，并选择合适的下载方式下载源码包。
-  开始训练    
   
    1. 启动训练之前，首先要配置程序运行相关环境变量。

       环境变量配置信息参见：

          [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)
    
    2. 单卡训练 

        2.1 配置train_full_1p.sh脚本中`data_dir`（脚本路径GNMT_ID0015_for_TensorFlow/test/train_full_1p.sh）,请用户根据实际路径配置，数据集参数如下所示：

            --data_dir=/data/wmt16_de_en 

        2.2 单p指令如下:

            bash train_full_1p.sh

    3. 8卡训练  
    
        3.1 配置train_full_8p.sh脚本中`data_dir`（脚本路径GNMT_ID0015_for_TensorFlow/test/train_full_8p.sh）,请用户根据实际路径配置，数据集参数如下所示：
            
        
           --data_dir=/data/wmt16_de_en
        
        3.2 8p指令如下: 
            
       
           bash train_full_8p.sh
   
-  验证。

    1. 修改train_full_1p.sh脚本中的mode参数为mode=infer：
    
       
        ```
        --mode=infer
        ```



<h2 id="迁移学习指导.md">迁移学习指导</h2>

- 数据集准备。

    1.  获取数据。
        请参见“快速上手”中的数据集准备。
    3.  数据目录结构如下：
        
        ```
        |--data
        |  |--wmt16_de_en
        |  |  |--train.tok.clean.bpe.32000
        |  |  |--newstest2014.tok.bpe.32000
        
        ```
    
-   模型训练。

    参考“模型训练”中训练步骤。

-   模型评估。

    参考“模型训练”中验证步骤。

<h2 id="高级参考.md">高级参考</h2>

脚本和示例代码

```
.
├── Dockerfile                           // 构建容器文件
├── LICENSE
├── NOTICE
├── README.md
├── attention_wrapper.py
├── beam_search_decoder.py
├── benchmark_hooks.py
├── block_lstm.py
├── configs
│   ├── rank_table_16p.json              // 16P分布式json文件
│   └── rank_table_8p.json               // 8P分布式json文件
├── estimator.py                         // 训练和推理功能 
├── gnmt.sh
├── gnmt_model.py
├── hook.py
├── model.py
├── model_helper.py
├── modelzoo_level.txt
├── nmt.py                                // 训练入口文件
├── ops_info.json
├── scripts
│   ├── docker
│   │   ├── build.sh									 		// 用于构建 GNMT 容器的脚本
│   │   └── interactive.sh                // 交互式运行 GNMT 容器的脚本
│   ├── filter_dataset.py
│   ├── parse_log.py                      // 用于从训练日志中检索 JSON 格式信息的脚本
│   ├── translate.py                      // nmt.py用于基准测试和运行推理
│   ├── verify_dataset.sh
│   └── wmt16_en_de.sh                    // 用于下载和预处理数据集的脚本
├── test
│   ├── aic-ascend910-ops-info.json
│   ├── env.sh
│   ├── launch.sh
│   ├── sigmoid.py
│   ├── train_full_1p.sh
│   ├── train_full_8p.sh
│   ├── train_performance_1p.sh
│   └── train_performance_8p.sh
├── utils
│   ├── __init__.py
│   ├── evaluation_utils.py
│   ├── iterator_utils.py
│   ├── math_utils.py
│   ├── misc_utils.py
│   ├── nmt_utils.py
│   └── vocab_utils.py
└── variable_mgr
    ├── BUILD
    ├── __init__.py
    ├── allreduce.py
    ├── batch_allreduce.py
    ├── constants.py
    ├── variable_mgr.py
    └── variable_mgr_util.py
```


## 脚本参数<a name="section6669162441511"></a>

```
  --learning_rate LEARNING_RATE
                        Learning rate.
  --warmup_steps WARMUP_STEPS
                        How many steps we inverse-decay learning.
  --max_train_epochs MAX_TRAIN_EPOCHS
                        Max number of epochs.
  --target_bleu TARGET_BLEU
                        Target bleu.
  --data_dir DATA_DIR   Training/eval data directory.
  --translate_file TRANSLATE_FILE
                        File to translate, works only with translate mode
  --output_dir OUTPUT_DIR
                        Store log/model files.
  --batch_size BATCH_SIZE
                        Total batch size.
  --log_step_count_steps LOG_STEP_COUNT_STEPS
                        The frequency, in number of global steps, that the
                        global step and the loss will be logged during training
  --num_gpus NUM_GPUS   Number of gpus in each worker.
  --random_seed RANDOM_SEED
                        Random seed (>0, set a specific seed).
  --ckpt CKPT           Checkpoint file to load a model for inference.
                        (defaults to newest checkpoint)
  --infer_batch_size INFER_BATCH_SIZE
                        Batch size for inference mode.
  --beam_width BEAM_WIDTH
                        beam width when using beam search decoder. If 0, use
                        standard decoder with greedy helper.
  --amp                 use amp for training and inference
  --mode {train_and_eval,infer,translate}

```

## 训练过程<a name="section1589455252218"></a>

1. 通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡，8卡网络训练。 
2. 将训练脚本（train_full_1p.sh,train_full_8p.sh）中的data_dir设置为训练数据集的路径。具体的流程参见“模型训练”的示例。 
3. 模型存储路径为“${cur_path}/output/$ASCEND_DEVICE_ID”，包括训练的log以及checkpoints文件。
4. 以单卡训练为例，loss信息在文件${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中，示例如下。 

```
training time for epoch 1: 29.37 mins (2918.36 sent/sec, 139640.48 tokens/sec)
[...]
bleu is 20.50000
```

## 推理/验证过程<a name="section1465595372416"></a>

1. 在6个 epoch训练执行完成后： 
    方法一：参照“模型训练”中的测试流程，需要修改脚本启动参数（脚本位于test/train_full_1p.sh）将mode设置为infer，然后执行脚本。 
    
    方法二：在训练过程中将mode修改为train_and_eval，训练过程会在每个训练时期后自动运行评估并输出 BLEU 分数。 
    
    `bash train_full_1p.sh `
    
    该脚本会自动执行验证流程，验证结果会输出到 ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log文件中，示例如下。 
    

```
eval time for epoch 1: 1.57 mins (78.48 sent/sec, 4283.88 tokens/sec)
```
