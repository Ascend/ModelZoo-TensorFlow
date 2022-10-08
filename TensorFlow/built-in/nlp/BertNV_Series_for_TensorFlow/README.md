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

**修改时间（Modified） ：2020.10.14**

**大小（Size）：1331.2M**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Benchmark**

**描述（Description）：基于TensorFlow框架的BERT预训练及下游任务代码**

<h2 id="概述.md">概述</h2>

   BERT是谷歌2018年推出的预训练语言模型结构，通过自监督训练实现对语义语境相关的编码，是目前众多NLP应用的基石。

-   参考论文：

    [Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv 
    preprint arXiv:1810.04805.](https://arxiv.org/pdf/1810.04805.pdf)
    
-   参考实现：

    [https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT)

-   适配昇腾 AI 处理器的实现：
    
    https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/built-in/nlp/BertNV_Series_for_TensorFlow
    
- 通过Git获取对应commit\_id的代码方法如下：

  ```
  git clone {repository_url}    # 克隆仓库的代码
  cd {repository_name}    # 切换到模型的代码仓目录
  git checkout  {branch}    # 切换到对应分支
  git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
  cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```

## 默认配置<a name="section91661242121611"></a>

-   网络结构
    - 24-layer, 1024-hidden, 16-heads, 340M parameters
-   训练超参（单卡）：
    - Batch size: 24
    - max_predictions_per_seq: 80
    - max_seq_length: 512
    - Learning rate(LR): 5e-5, polynomial decay
    - optimizer: Adam 
    - Weight decay: 0.01
    - beta_1: 0.9
    - beta_2: 0.999
    - Train epoch: 1

## 支持特性<a name="section1899153513554"></a>

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 是    |
| 混合精度  | 是    |
| 数据并行  | 是    |


## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

开启混合精度相关代码示例。

```
    run_config = NPURunConfig(
            model_dir=self.config.model_dir,
            session_config=session_config,
            keep_checkpoint_max=5,
            save_checkpoints_steps=5000,
            enable_data_pre_proc=True,
            iterations_per_loop=iterations_per_loop,
            precision_mode='allow_mix_precision',
            hcom_parallel=True      
        ）
```


<h2 id="训练环境准备.md">训练环境准备</h2>

1.  硬件环境准备请参见各硬件产品文档"[驱动和固件安装升级指南]( https://support.huawei.com/enterprise/zh/category/ai-computing-platform-pid-1557196528909)"。需要在硬件设备上安装与CANN版本配套的固件与驱动。
2.  宿主机上需要安装Docker并登录[Ascend Hub中心](https://ascendhub.huawei.com/#/detail?name=ascend-tensorflow-arm)获取镜像。

    当前模型支持的镜像列表如[表1](#zh-cn_topic_0000001074498056_table1519011227314)所示。

    **表 1** _镜像列表_

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

1、用户自行准备好数据集，本网络包括Bert的Pre-training和Fine tuning任务

2、Pre-training任务使用的数据集是wikipedia-en，Fine tuning使用的数据集是MPRC、MNLI、CoLA、SQuAD1.1和SQuAD2.0。

```shell
#为了提升训练的端到端效率，SQuAD1.1和SQuAD2.0均提前做了转tfrecord的处理，转换方式如下：
cd ${work_path}
python3 ${work_path}/src/utils/create_squad_data.py --train_file=${data_path}/train-v1.1.json  \
                                                    --predict_file=${data_path}/dev-v1.1.json  \
                                                    --vocab_file=${model_path}/vocab.txt
```

3、Bert训练的模型及数据集可以参考"概述 -> 参考实现"

## 模型训练<a name="section715881518135"></a>

#### 模型训练

- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练。

    - 启动训练之前，首先要配置程序运行相关环境变量。

      环境变量配置信息参见：

      [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

    - 单卡训练

      网络共包含18个训练，其中Pre-training 8个训练任务, Fine tuning 10个训练任务。
    
      **Pre-training 任务**:

      ```shell
      #ID0060: num_hidden_layers=12 max_seq_length=128 optimizer=Adam
      bash train_ID0060_BertBase_performance_1p.sh --data_path=/home
      
      #ID3067: num_hidden_layers=24 max_seq_length=128 optimizer=Adam
      bash train_ID3067_BertLarge-128_performance_1p.sh --data_path=/home
      
      #ID3068: num_hidden_layers=24 max_seq_length=512 optimizer=lamb phase2
      bash train_ID3068_BertLarge-512_performance_1p.sh --data_path=/home
      
      #ID3069: num_hidden_layers=12 max_seq_length=512 optimizer=lamb phase2
      bash train_ID3069_BertBase-512_performance_1p.sh --data_path=/home
      
      #ID3206: num_hidden_layers=12 max_seq_length=512 optimizer=Adam
      bash train_ID3206_BertBase-512_performance_1p.sh --data_path=/home
      
      #ID3207: num_hidden_layers=24 max_seq_length=512 optimizer=Adam
      bash train_ID3207_BertLarge-512_performance_1p.sh --data_path=/home
      
      #ID3208: num_hidden_layers=12 max_seq_length=128 optimizer=lamb phase1
      bash train_ID3208_BertBase-128_performance_1p.sh --data_path=/home
      
      #ID3209: num_hidden_layers=24 max_seq_length=128 optimizer=lamb phase1
      bash train_ID3209_BertLarge-128_performance_1p.sh --data_path=/home 
      ```
    
      **Fine tuning 任务**:

      ```shell
      #ID1641: MRPC num_hidden_layers=24 max_seq_length=128 optimizer=Adam
      bash train_ID1641_BertLarge-128_performance_1p.sh --data_path=/home
      
      #ID3232: MRPC num_hidden_layers=12 max_seq_length=128 optimizer=Adam
      bash train_ID3232_BertBase-128_performance_1p.sh --data_path=/home
      
      #ID1642: MNLI num_hidden_layers=24 max_seq_length=128 optimizer=Adam
      bash train_ID1642_BertLarge-128_performance_1p.sh --data_path=/home
      
      #ID3233: MNLI num_hidden_layers=12 max_seq_length=128 optimizer=Adam
      bash train_ID3233_BertBase-128_performance_1p.sh --data_path=/home
      
      #ID1643: CoLA num_hidden_layers=24 max_seq_length=128 optimizer=Adam
      bash train_ID1643_BertLarge-128_performance_1p.sh --data_path=/home
      
      #ID3234: CoLA num_hidden_layers=12 max_seq_length=128 optimizer=Adam
      bash train_ID3234_BertBase-128_performance_1p.sh --data_path=/home
      
      #ID3217: SQuAD1.1 num_hidden_layers=12 max_seq_length=384 optimizer=Adam
      bash train_ID3217_BertBase-Squad1.1_performance_1p.sh --data_path=/home
      
      #ID3218: SQuAD1.1 num_hidden_layers=24 max_seq_length=384 optimizer=Adam
      bash train_ID3218_BertLarge-Squad1.1_performance_1p.sh --data_path=/home
      
      #ID3219: SQuAD2.0 num_hidden_layers=12 max_seq_length=384 optimizer=Adam
      bash train_ID3219_BertBase-Squad2.0_performance_1p.sh --data_path=/home 
      
      #ID3220: SQuAD2.0 num_hidden_layers=24 max_seq_length=384 optimizer=Adam
      bash train_ID3220_BertLarge-Squad2.0_performance_1p.sh --data_path=/home 
      ```

#### 分布式插件使能分布式

ID0060网络分布式统一训练脚本`./test/train_ID0060_BertBase_performance_distribute.sh`, 该脚本由`./test/train_ID0060_BertBase_performance_8p.sh`修改而来, 具体差异可自行比对, 分布式插件屏蔽了多P 执行过程中rank_table.json和环境变量的差异, 多P可以共有一个脚本, 具体超参请用户根据实际情况修改

训练前请下载工具并根据说明完成配置

工具路径: https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/Tools/ascend_distribute


- 8p训练
```
python3 $path/distrbute_npu.py --np 8 --env 10.10.10.10:8 --train_command "bash train_ID0060_BertBase_performance_distribute.sh --data_path=/npu/traindata"
```


- 16p训练

```
python3 $path/distrbute_npu.py --np 16 --env 10.10.10.10:8,10.10.10.11:8 --train_command "bash train_ID0060_BertBase_performance_distribute.sh --data_path=/npu/traindata"
```


<h2 id="高级参考.md">高级参考</h2>

    脚本和示例代码
    ├── configs  
    │    ├──8p.json              //8p rank table配置文件
    │    ├──bert_base_config.json                //bert large模型配置文件
    │    ├──bert_large_config.json               //bert base模型配置文件
    │    ├──bert_base_vocab.txt                 //bert base中文词表
    ├── src
    │    ├──utils
    │    │    ├──create_pretraining_data.py            //生成预训练数据脚本
    │    │    ├──create_glue_data.py                   //glue数据集转tfrecord脚本
    │    │    ├──create_squad_data.py                  //squad数据集转tfrecord脚本
    │    │    ├──dllogger_class.py                     //生成与训练数据脚本
    │    │    ├──gpu_affinity.py                       //设置gpu亲和性
    │    │    ├──utils.py                              //公共脚本
    │    ├──gpu_environment.py                     //原始gpu_environment设置
    │    ├──modeling.py                           //NEZHA模型脚本
    │    ├──optimization.py                       //优化器脚本
    │    ├──extract_features.py                   //特征抽取脚本
    │    ├──fp16_utils.py                       //fp16 utils脚本
    │    ├──fused_layer_norm.py                     //layer norm融合脚本
    │    ├──run_pretraining.py                    //预训练启动脚本
    │    ├──run_classifier.py                    //下游任务分类脚本
    │    ├──run_squad.py                         //下游任务squad脚本
    │    ├──tf_metrics.py                        //tf metrics脚本
    │    ├──tokenization.py                      //分词器脚本
    ├── CONTRIBUTING.md                             //CONTRIBUTING.md
    ├── LICENCE                                   //LICENCE
    ├── NOTICE                                   //NOTICE
    ├── README.md                                 //说明文档


## 脚本参数<a name="section6669162441511"></a>


```
 --train_batch_size=128 \           # 每个NPU训练的batch size
 --learning_rate=1e-4 \             # 学习率
 --num_warmup_steps=10000 \         # 初始warmup训练epoch数
 --num_train_steps=500000 \         # 训练次数
 --input_files_dir=xxxx \           # 训练数据集路径
 --eval_files_dir=xxxx \            # 验证数据集路径  
 --iterations_per_loop=100 \        # NPU运行时，device端下沉次数
    
```


## 训练过程<a name="section1589455252218"></a>

通过“快速上手”中的训练指令启动训练。

```
I0521 19:45:05.731803 281473752813584 basic_session_run_hooks.py:692] global_step/sec: 2.451
I0521 19:45:05.732023 281473228546064 basic_session_run_hooks.py:260] global_step = 1323600, masked_lm_loss = 0.7687549, next_sentence_loss = 0.005564222, total_loss = 0.7743191 (81.600 sec)
I0521 19:45:05.732058 281473117769744 basic_session_run_hooks.py:260] global_step = 1323600, masked_lm_loss = 0.74314255, next_sentence_loss = 0.023222845, total_loss = 0.7663654 (81.600 sec)
2020-05-21 19:45:05.732132: I tf_adapter/kernels/geop_npu.cc:526] [GEOP] RunGraphAsync callback, status:0, kernel_name:GeOp15_0[ 2409us]
I0521 19:45:05.732016 281473584246800 basic_session_run_hooks.py:692] global_step/sec: 2.451
I0521 19:45:05.732048 281472971046928 basic_session_run_hooks.py:692] global_step/sec: 2.451
loss_scale: loss_scale:[65536.0] 
2020-05-21 19:45:05.732378: I tf_adapter/kernels/geop_npu.cc:526] [GEOP] RunGraphAsync callback, status:0, kernel_name:GeOp15_0[ 2445us]
loss_scale:[65536.0] 
I0521 19:45:05.732480 281473752813584 basic_session_run_hooks.py:260] global_step = 1323600, masked_lm_loss = 0.94164073, next_sentence_loss = 0.023505606, total_loss = 0.96514636 (81.600 sec)
I0521 19:45:05.732715 281473584246800 basic_session_run_hooks.py:260] global_step = 1323600, masked_lm_loss = 0.738043, next_sentence_loss = 0.03810045, total_loss = 0.77614343 (81.599 sec)
I0521 19:45:05.732658 281473385623568 basic_session_run_hooks.py:692] global_step/sec: 2.451
I0521 19:45:05.732574 281473416220688 basic_session_run_hooks.py:692] global_step/sec: 2.45098
I0521 19:45:05.732777 281472971046928 basic_session_run_hooks.py:260] global_step = 1323600, masked_lm_loss = 0.7797201, next_sentence_loss = 0.05669275, total_loss = 0.8364129 (81.600 sec)loss_scale: [65536.0]
loss_scale:[65536.0] 
I0521 19:45:05.733291 281473385623568 basic_session_run_hooks.py:260] global_step = 1323600, masked_lm_loss = 0.8004036, next_sentence_loss = 0.12787658, total_loss = 0.9282802 (81.600 sec)[65536.0]

```

## 多机训练过程<a name="section1589455252298"></a>

以 test/train_ID3067_BertLarge-128_full_32p.sh为例，此脚本适用于32p（4*8）多机集群场景

```
1.修改脚本 test/train_ID3067_BertLarge-128_full_32p.sh
    注释如下脚本
    nohup python3 set_ranktable.py --npu_nums=$linux_num --conf_path=$conf_path
    自行配置rank_table.json路径，如下
    export RANK_TABLE_FILE=$cur_path/rank_table.json

2.多机集群环境同步代码及数据集，保证集群环境各服务器代码及数据集一致

3.多机环境分别拉起脚本,如下格式，server_index为服务器顺序标识，32p场景值为0-3，servers_num为服务器数量，32p场景值为4，devices_num为单台服务器使用卡数，默认为8，$data_path为数据集路径
    bash train_ID3067_BertLarge-128_full_32p.sh  --server_index=$server_index  --servers_num=4 --devices_num=8 --data_path=$data_path

    example：
    服务器1 :bash train_ID3067_BertLarge-128_full_32p.sh  --server_index=0  --servers_num=4 --devices_num=8 --data_path=$data_path
    服务器2 :bash train_ID3067_BertLarge-128_full_32p.sh  --server_index=1  --servers_num=4 --devices_num=8 --data_path=$data_path
    服务器3 :bash train_ID3067_BertLarge-128_full_32p.sh  --server_index=2  --servers_num=4 --devices_num=8 --data_path=$data_path
    服务器4 :bash train_ID3067_BertLarge-128_full_32p.sh  --server_index=3  --servers_num=4 --devices_num=8 --data_path=$data_path 
    
 ```