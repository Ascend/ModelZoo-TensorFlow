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

**修改时间（Modified） ：2021.8.15**

**大小（Size）：210KB**

**框架（Framework）：TensorFlow_2.6.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的BertLarge自然语言处理网络训练代码**

<h2 id="概述.md">概述</h2>

## 简述

BERT模型的全称是：BidirectionalEncoder **Representations** from Transformer。BERT模型的目标是利用大规模无标注语料训练、获得文本的包含丰富语义信息的Representation，即：文本的语义表示，然后将文本的语义表示在特定NLP任务中作微调，最终应用于该NLP任务。

- 论文路径

  https://arxiv.org/abs/1810.04805

- 开源代码路径

  https://github.com/sgpyc/training/tree/bert_tf2/language_model/tensorflow2/bert

-   适配昇腾 AI 处理器的实现：
    
    https://gitee.com/ascend/modelzoo/tree/master/built-in/TensorFlow/Official/nlp/BertLarge_ID0634_for_TensorFlow2.X

-   通过Git获取对应commit\_id的代码方法如下：
    
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    

## 默认配置<a name="section91661242121611"></a>
-   网络结构
    -   24-layer, 1024-hidden, 16-heads, 340M parameters
    
-   训练超参（单卡）：
    -   Batch size: 24
    -   max_predictions_per_seq: 76
    -   max_seq_length: 512
    -   Learning rate\(LR\): 0.000058711
    -   Weight decay: 0.01
    -   beta_1:  0.91063
    -   beta_2: 0.96497
    -   Train epoch: 32


## 支持特性<a name="section1899153513554"></a>

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 是    |
| 混合精度  | 是    |
| 数据并行  | 是    |


## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>
拉起脚本中，传入--precision_mode='allow_mix_precision'

```
 ./train_performance_1p_24bs.sh --help

parameter explain:
    --precision_mode         precision mode(allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision)
    --over_dump                  if or not over detection, default is False
    --data_dump_flag         data dump flag, default is False
    --data_dump_step             data dump step, default is 10
    --profiling                  if or not profiling for performance debug, default is False
    --data_path                  source data of training
    -h/--help                    show help message
```

相关代码示例:

```
flags.DEFINE_string(name='precision_mode', default= 'allow_fp32_to_fp16',
                    help='allow_fp32_to_fp16/force_fp16/ ' 
                    'must_keep_origin_dtype/allow_mix_precision.')

npu_device.global_options().precision_mode=FLAGS.precision_mode
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

## 数据集准备<a name="section361114841316"></a>

1、用户自行准备好数据集，包括训练数据集和验证数据集。使用的数据集是wikipedia

2、训练的数据集放在train目录，验证的数据集放在eval目录

3、bert 预训练的模型及数据集可以参考"简述->开源代码路径处理"

数据集目录参考如下：

```
├── tfrecord
│    ├──train   ├──part-00XXX.tfrecord
│    │          │ ......         
│    │          ├──part-00XXX.tfrecord
│    ├──tf2_ckpt├──model.ckpt-28252.data-000000-of-000001
│    |          ├──model.ckpt-28252.index
│    ├──val     ├──eval.tfrecord
                ├──eval_10k.tfrecord
```

4、数据集pack (仅在使用pack策略进行训练时执行)

若训练时使用pack策略（参看“模型训练” - 开始训练- 4. pack策略），须将数据集进行处理，生成为pack后的数据集。再使用pack后的数据集进行训练。数据集转换脚本在bert/data_pack/目录下。数据集pack过后，在指定的目录下生成“strategy_record”开头的一系列文件。
进行数据集pack时，需要将训练（train）的数据集与验证（eval）的数据集分别进行pack，生成在新的目录中。未pack的数据集为tfrecord格式，需先使用bert/data_pack目录下 "bert_data/record_to_binary.py" 将其批量转换为bin文件：

python3 bert_data/record_to_binary.py --tf-record-glob="path/to/your/unpacked/data/part*.tfrecord" --output-path="path/to/store/binery/files"

然后再使用bert/data_pack目录下 "pack_pretraining_data.py" ,将bin文件转化为pack后的数据集。

python3 pack_pretraining_data.py --input-glob="path/to/store/binery/files" --output-dir="packed/data/folder"

文件夹路径需要自己创建。

## 模型训练<a name="section715881518135"></a>
- 下载训练脚本。
- 检查并修改configs/目录下8卡IP的json配置文件“rank_table_8p.json"。
  
```
{
  "server_count":"1",
  "server_list":[
    {
      "server_id":"10.147.179.27",
      "device":[
        {
          "device_id":"0",
          "device_ip":"192.168.100.100",
          "rank_id":"0"
        },
        {
          "device_id":"1",
          "device_ip":"192.168.101.100",
          "rank_id":"1"
        },
        {
          "device_id":"2",
          "device_ip":"192.168.102.100",
          "rank_id":"2"
        },
        {
          "device_id":"3",
          "device_ip":"192.168.103.100",
          "rank_id":"3"
        },
        {
          "device_id":"4",
          "device_ip":"192.168.100.101",
          "rank_id":"4"
        },
        {
          "device_id":"5",
          "device_ip":"192.168.101.101",
          "rank_id":"5"
        },
        {
          "device_id":"6",
          "device_ip":"192.168.102.101",
          "rank_id":"6"
        },
        {
          "device_id":"7",
          "device_ip":"192.168.103.101",
          "rank_id":"7"
        }
      ]
    }
  ],
  "status":"completed",
  "version":"1.0"
}

```

- 开始训练。

    1. 启动训练之前，首先要配置程序运行相关环境变量。

       环境变量配置信息参见：

          [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)


    2. 单卡训练
       
        2.1 单卡训练指令（脚本位于BertLarge_TF2.x_for_Tensorflow/test/train_performance_1p_24bs.sh）,请确保下面例子中的“--data_path”修改为用户的tfrecord的路径,这里选择将tfrecord文件夹放在home目录下。
        
            bash test/train_performance_1p_24bs.sh --data_path=/home/tfrecord --precision_mode=allow_mix_precision
    
    3. 8卡训练
    
        3.1 8卡训练指令（脚本位于BertLarge_TF2.x_for_Tensorflow/test/train_performance_8p_192bs.sh),请确保下面例子中的“--data_path”修改为用户的tfrecord的路径。
    
            bash test/train_performance_8p_192bs.sh --data_path=/home/tfrecord --precision_mode=allow_mix_precision

    4. pack策略
    
        4.1 含pack策略的训练脚本（./test/目录下名字带有"_packed"的脚本即为相应包含pack策略的训练脚本）
        使用pack策略进行训练时，需使用pack过后的数据集（train、eval）及对应的预训练模型。若无对应tensorflow-v2版本packed预训练模型，可由tensorflow-v1版本进行转换得来。模型转换相关脚本为bert/tf2_encoder_checkpoint_converter.py，详见：脚本和事例代码 - 模型转换脚本

<h2 id="迁移学习指导.md">高级参考</h2>

## 脚本和事例代码

```
|--bert			#网络代码目录
|   |--tf2_common
|   |--modeling
|   |--data_pack        #pack脚本及说明所在目录
|	|--......
|--configs		#配置文件目录
|   |--bert_config.json
|   |--rank_table_8p.json
|--test			#训练脚本目录
|	|--train_performance_1p_24bs.sh
|	|--train_performance_8p_192bs.sh
|   |--......
```

添加：

模型转换脚本（仅tensorflow_v1版本checkpoint转化为tensorflow_v2版本时使用）

tensorflow-v1的checkpoint与tensorflow-v2的checkpoint从结构和使用上具有较大差异。迁移原有tensorflow-v1生成的checkpoint， 使其转换为tensorflow-v2环境中可使用的checkpoint， 需使用转换脚本， 脚本位置：“./bert/tf2_encoder_checkpoint_converter.py”。
脚本使用示例： python3 tf2_encoder_checkpoint_converter.py --bert_config_file=/path/to/your/tensorflow_v1/bert_config.json  --checkpoint_to_convert=/path/to/your/tensorflow_v1/model.ckpt-28252 --converted_checkpoint_path=/path/to/save/output_ckpt/output_ckpt

## 脚本参数<a name="section6669162441511"></a>

```
	--max_seq_length		      		The maximum total input sequence length after WordPiece tokenizationTrue,default:128
    --max_predictions_per_seq           Maximum predictions per sequence_output,default:20  
    --train_batch_size					Total batch size for training,default:32
    --num_steps_per_epoch		        Total number of training steps to run per epoch,default:1000
    --warmup_steps						Warmup steps for optimizer,default:10000
    --start_warmup_step					The starting step of warmup,default:0
	--stop_steps						The number of steps to stop training,default:None
	--do_eval							Whether to run eval,default:False
	--device_warmup						Whether or not to enable device warmup,default:False
	--steps_between_eval				Steps between an eval. Is multiple of steps per loop,default:10000
	--steps_before_eval_start			Steps before starting eval,default:0
	--num_eval_samples					Number of eval samples,default:10000
	--eval_batch_size					Total batch size for training,default:32
	--weight_decay_rate					The weight_decay_rate value for the optimizer,default:0.01
	--beta_1							The beta_1 value for the optimizer,default:0.9
	--beta_2							The beta_2 value for the optimizer,default:0.999
	--precision_mode         			precision mode,default:allow_fp32_to_fp16
    --over_dump		         			if or not over detection, default is False
    --data_dump_flag	     			data dump flag, default is False
    --data_dump_step		 			data dump step, default is 10
    --profiling		         			if or not profiling for performance debug, default is False
    --data_path		         			source data of training
```

## 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡，8卡网络训练。模型存储路径为${cur_path}/output/$ASCEND_DEVICE_ID，包括训练的log以及checkpoints文件。以8卡训练为例，loss信息在文件${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。






