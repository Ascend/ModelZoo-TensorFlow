- [基本信息](#基本信息.md)
- [概述](#概述.md)
- [训练环境准备](#训练环境准备.md)
- [快速上手](#快速上手.md)
- [迁移学习指导](#迁移学习指导.md)
- [高级参考](#高级参考.md)

<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Image Classification**

**版本（Version）：1.1**

**修改时间（Modified） ：2022.4.8**

**大小（Size）：210KB**

**框架（Framework）：TensorFlow_2.6**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的Involutional网络训练代码**

<h2 id="概述.md">概述</h2>

- Involutional neural networks 由**Inverting the Inherence of Convolution**卷积, 即Involution构成。Involution kernel具有位置特定且与通道无关的特点。

- 参考论文：

  https://arxiv.org/abs/2103.06255

- 参考实现：

  https://github.com/keras-team/keras-io/blob/master/examples/vision/involution.py

- 适配昇腾 AI 处理器的实现：

  https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow2/built-in/keras_sample/cv/involution_ID2515_for_TensorFlow2.X

- 通过Git获取对应commit\_id的代码方法如下：

        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换


## 默认配置<a name="section91661242121611"></a>
-   网络结构
    _________________________________________________________________
    Model: "inv_model"
    _________________________________________________________________
    Layer (type) Output Shape Param

    input_1 (InputLayer) [(None, 32, 32, 3)] 0
    _________________________________________________________________
    inv_1 (Involution) ((None, 32, 32, 3), (None 26
    _________________________________________________________________
    re_lu_3 (ReLU) (None, 32, 32, 3) 0
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 16, 16, 3) 0
    _________________________________________________________________
    inv_2 (Involution) ((None, 16, 16, 3), (None 26
    _________________________________________________________________
    re_lu_4 (ReLU) (None, 16, 16, 3) 0
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 8, 8, 3) 0
    _________________________________________________________________
    inv_3 (Involution) ((None, 8, 8, 3), (None, 26
    _________________________________________________________________
    re_lu_5 (ReLU) (None, 8, 8, 3) 0
    _________________________________________________________________
    flatten_1 (Flatten) (None, 192) 0
    _________________________________________________________________
    dense_2 (Dense) (None, 64) 12352
    _________________________________________________________________
    dense_3 (Dense) (None, 10) 650

    Total params: 13,080 Trainable params: 13,074 Non-trainable params: 6
    _________________________________________________________________

-   训练超参（单卡）：
    -   Batch size: 256
    -   Train epoch: 200


## 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 否       |
| 混合精度   | 否       |
| 数据并行   | 否       |


## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>
拉起脚本中，默认开启混合精度传入，即precision_mode='allow_mix_precision'

```
 ./train_performance_1p.sh --help

parameter explain:
    --precision_mode         precision mode(allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision)
    --over_dump              if or not over detection, default is False
    --data_dump_flag         data dump flag, default is False
    --data_dump_step         data dump step, default is 10
    --profiling              if or not profiling for performance debug, default is False
    --data_path              source data of training
    --max_step               # of step for training
    --learning_rate          learning rate
    --batch                  batch size
    --modeldir               model dir
    --save_interval          save interval for ckpt
    --loss_scale             enable loss scale ,default is False
    -h/--help                show help message
```

相关代码示例:

```
npu_device.global_options().precision_mode=FLAGS.precision_mode
```

<h2 id="训练环境准备.md">训练环境准备</h2>

1. 硬件环境和运行环境准备请参见《[CANN软件安装指南](https://support.huawei.com/enterprise/zh/category/ai-computing-platform-pid-1557196528909)》
2. 宿主机上需要安装Docker并登录[Ascend Hub中心](https://ascendhub.huawei.com/#/detail?name=ascend-tensorflow-arm)获取镜像。

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
    <td class="cellrowborder" valign="top" width="25.52%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001074498056_p1450714271532"><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><em id="i72359495125"><a name="i72359495125"></a><a name="i72359495125"></a>21.09</em></p>
    </td>
    <td class="cellrowborder" valign="top" width="27.16%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001074498056_p18244640152312"><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><em id="i162363492129"><a name="i162363492129"></a><a name="i162363492129"></a><a href="https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software" target="_blank" rel="noopener noreferrer">20.2</a></em></p>
    </td>
    </tr>
    </tbody>
    </table>

3. 运行以下命令安装依赖。
```
pip install requirements.txt
```
说明：依赖配置文件requirements.txt文件位于模型的根目录

<h2 id="快速上手.md">快速上手</h2>

## 数据集准备<a name="section361114841316"></a>

1、用户自行准备好数据集，包括训练数据集和验证数据集。使用的数据集是wikipedia

2、训练的数据集放在train目录，验证的数据集放在eval目录

3、bert 预训练的模型及数据集可以参考"简述->开源代码路径处理"

数据集目录参考如下：

```

├─data
│  └─cifar-10-batches-py
│    ├──batchex.meta
│    ├──data_batch_1
│    ├──data_batch_2
│    ├──data_batch_3
│    ├──data_batch_4
│    ├──data_batch_5
│    ├──readme.html
│    └─test_batch
```



## 模型训练<a name="section715881518135"></a>
- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练。

    1. 启动训练之前，首先要配置程序运行相关环境变量。

       环境变量配置信息参见：

          [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)


    2. 单卡训练

        2. 1单卡训练指令（脚本位于BertLarge_TF2.x_for_Tensorflow/test/train_full_1p_16bs.sh）,请确保下面例子中的“--data_path”修改为用户的data的路径,这里选择将data文件夹放在home目录下。训练默认开启混合精度，即precision_mode='allow_mix_precision'

            bash train_full_1p_static.sh --data_path=/home/data




<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码<a name="section08421615141513"></a>

```
|--configs		#网络配置
|   |--ops_info.json
|--test			#训练脚本目录
|	|--train_full_1p_static.sh   # 全量静态训练
|	|--train_performance_1p.sh
|	|--train_performance_1p_inv.sh
|	|--train_performance_1p_static.sh
|--involution.py #网络脚本
|--......
```

## 脚本参数<a name="section6669162441511"></a>

```
  parser.add_argument('--data_path', default="../cifar-10-batches-py/", help="""directory to data""")
  parser.add_argument('--batch_size', default=128, type=int, help="""batch size for 1p""")
  parser.add_argument('--epochs', default=10, type=int, help="""epochs""")
  parser.add_argument('--Drop_Reminder', dest="Drop_Reminder", type=ast.literal_eval, help='static or not')
  parser.add_argument('--save_h5', dest="save_h5", type=ast.literal_eval, help='whether save h5 file after training')
  parser.add_argument('--network', default="convolution", help='train network, only "convolution" or "involution"')
  #===============================NPU Migration=========================================
  parser.add_argument("--log_steps", default=50, type=int, help="TimeHis log Step.")
  parser.add_argument('--precision_mode', default="allow_mix_precision", type=str,help='the path to save over dump data')
  parser.add_argument('--over_dump', dest='over_dump', type=ast.literal_eval, help='if or not over detection, default is False')
  parser.add_argument('--data_dump_flag', dest='data_dump_flag', type=ast.literal_eval, help='data dump flag, default is False')
  parser.add_argument('--data_dump_step', default="10", help='data dump step, default is 10')
  parser.add_argument('--profiling', dest='profiling', type=ast.literal_eval,help='if or not profiling for performance debug, default is False')
  parser.add_argument('--profiling_dump_path', default="/home/data", type=str,help='the path to save profiling data')
  parser.add_argument('--over_dump_path', default="/home/data", type=str,help='the path to save over dump data')
  parser.add_argument('--data_dump_path', default="/home/data", type=str,help='the path to save dump data')
  parser.add_argument('--use_mixlist', dest='use_mixlist', type=ast.literal_eval, help='use_mixlist flag, default is False')
  parser.add_argument('--fusion_off_flag', dest='fusion_off_flag', type=ast.literal_eval, help='fusion_off flag, default is False')
  parser.add_argument('--mixlist_file', default="ops_info.json", type=str,help='mixlist file name, default is ops_info.json')
  parser.add_argument('--fusion_off_file', default="fusion_switch.cfg", type=str,help='fusion_off file name, default is fusion_switch.cfg')
  parser.add_argument('--auto_tune', dest='auto_tune', type=ast.literal_eval,help='auto_tune flag, default is False')
```

## 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡，8卡网络训练。模型存储路径为${cur_path}/output/$ASCEND_DEVICE_ID，包括训练的log以及checkpoints文件。以8卡训练为例，loss信息在文件${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。
