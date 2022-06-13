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

**框架（Framework）：TensorFlow_2.6.2**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的BigTransport(Bit)图像分类网络训练代码**

<h2 id="概述.md">概述</h2>

- BigTransport（也称为**BiT**）是一种图像分类迁移学习方法。

- 参考论文：

  https://arxiv.org/abs/1912.11370

- 参考实现：

  https://github.com/keras-team/keras-io/blob/master/examples/vision/bit.py

- 适配昇腾 AI 处理器的实现：

  skip

- 通过Git获取对应commit\_id的代码方法如下：

        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换


## 默认配置<a name="section91661242121611"></a>
-   网络结构
    -   class MyBiTModel(keras.Model):
            def __init__(self, num_classes, module, **kwargs):
                super().__init__(**kwargs)

                self.num_classes = num_classes
                self.head = keras.layers.Dense(num_classes, kernel_initializer="zeros")
                self.bit_model = module

            def call(self, images):
                bit_embedding = self.bit_model(images)
                return self.head(bit_embedding)

-   训练超参（单卡）：
    -   Batch size: 64
    -   Train epoch: 15


## 支持特性<a name="section1899153513554"></a>

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 否    |
| 混合精度  | 否    |
| 数据并行  | 否    |


## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>
拉起脚本中，传入--precision_mode='allow_mix_precision'

```
parameter explain:
  '--log_steps', default=1, type=int, help='log frequency')
  '--data_dir', default="../bit_datasets/", help='directory to data')
  '--batch_size', default=64, type=int, help='batch size for 1p')
  '--epochs', default=30, type=int, help='train epochs')
  '--eval_static', dest="eval_static", type=ast.literal_eval, help='drop_reminder')
  '--precision_mode', default="allow_mix_precision", type=str,help='train model')
  '--over_dump', dest='over_dump', type=ast.literal_eval, help='if or not over detection, default is False')
  '--data_dump_flag', dest='data_dump_flag', type=ast.literal_eval, help='data dump flag, default is False')
  '--data_dump_step', default="10", help='data dump step, default is 10')
  '--profiling', dest='profiling', type=ast.literal_eval,help='if or not profiling for performance debug, default is False')
  '--profiling_dump_path', default="/home/data", type=str,help='the path to save profiling data')
  '--over_dump_path', default="/home/data", type=str,help='the path to save over dump data')
  '--data_dump_path', default="/home/data", type=str,help='the path to save dump data')
  '--use_mixlist', dest='use_mixlist', type=ast.literal_eval, help='use_mixlist flag, default is False')
  '--fusion_off_flag', dest='fusion_off_flag', type=ast.literal_eval, help='fusion_off flag, default is False')
  '--mixlist_file', default="ops_info.json", type=str,help='mixlist file name, default is ops_info.json')
  '--fusion_off_file', default="fusion_switch.cfg", type=str,help='fusion_off file name, default is fusion_switch.cfg')
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
    <td class="cellrowborder" valign="top" width="25.52%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001074498056_p1450714271532"><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><em id="i72359495125"><a name="i72359495125"></a><a name="i72359495125"></a>20.12</em></p>
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
data
├── tf_flowers
│   └── 3.0.1
│       ├── dataset_info.json
│       ├── features.json
│       ├── label.labels.txt
│       ├── tf_flowers-train.tfrecord-00000-of-00002
│       └── tf_flowers-train.tfrecord-00001-of-00002
└── ttst
    ├── assets
    ├── bit_m-r50x1_1.tar.gz
    ├── saved_model.pb
    └── variables
        ├── variables.data-00000-of-00001
        └── variables.index
```

## 模型训练<a name="section715881518135"></a>
- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练。

    1. 启动训练之前，首先要配置程序运行相关环境变量。

       环境变量配置信息参见：

          [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)


    2. 单卡训练

        2. 1单卡训练指令（脚本位于bit_ID2496_for_TensorFlow2.X/test/train_full_1p.sh）,请确保下面例子中的“--data_path”修改为用户的tfrecord的路径,这里选择将data文件夹放在home目录下。默认precision_mode='allow_mix_precision'

            bash train_full_1p.sh --data_path=/home/data

<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码<a name="section08421615141513"></a>

```
bit_ID2496_for_TensorFlow2.X
├── LICENSE
├── README.md
├── requirements.txt
├── run_1p.sh
├── train.py
└──test   # 训练脚本目录
    ├── train_full_1p.sh
    ├── train_performance_1p_dynamic_eval.sh
    └── train_performance_1p_static_eval.sh
```

## 脚本参数<a name="section6669162441511"></a>

```
batch_size=64
#训练epoch，可选
train_epochs=15

############维测参数##############
precision_mode="allow_mix_precision"
#维持参数，以下不需要修改
over_dump=False
if [[ $over_dump == True ]];then
    over_dump_path=$cur_path/test/overflow_dump #此处cur_path为代码根目录
    mkdir -p ${over_dump_path}
fi
data_dump_flag=False
data_dump_step="10"
profiling=False
use_mixlist=False
mixlist_file="${cur_path}/configs/ops_info.json"
fusion_off_flag=False
fusion_off_file="${cur_path}/configs/fusion_switch.cfg"
```

## 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡，8卡网络训练。模型存储路径为${cur_path}/output/$ASCEND_DEVICE_ID，包括训练的log以及checkpoints文件。以8卡训练为例，loss信息在文件${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。
