-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [快速上手](#快速上手.md)
-   [迁移学习指导](#迁移学习指导.md)
-   [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Image Classification**

**版本（Version）：1.1**

**修改时间（Modified） ：2021.12.28**

**大小（Size）：210KB**

**框架（Framework）：TensorFlow_2.4.1**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的TCC(Temporal Cycle-Consistency)训练代码**

<h2 id="概述.md">概述</h2>

## 简述

TCC(Temporal Cycle-Consistency)可用于视频的自监督表示学习，它被用于CVPR 2019论文Temporal Cycle-Consistency Learning中，其中的许多方法对其他的序列数据也非常有用。

- 论文路径

  https://arxiv.org/abs/1904.07846

- 开源代码路径

  https://github.com/google-research/google-research/tree/master/tcc

-   适配昇腾 AI 处理器的实现：
    
    https://gitee.com/ascend/modelzoo/tree/master/built-in/TensorFlow/Official/nlp/Transformer_ID0633_for_TensorFlow2.X

-   通过Git获取对应commit\_id的代码方法如下：
    
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    

## 默认配置<a name="section91661242121611"></a>

-   训练超参（单卡）：
    -   Batch size: 2
    -   Num frames: 20
    -   Max iters: 150000


## 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 否       |
| 混合精度   | 否       |
| 数据并行   | 否       |

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

1、用户自行准备好数据集，包括训练、验证数据集以及使用ImageNet预训练的ResNet-50模型，其中训练数据集和验证数据集均为Pouring数据集

2、数据集的处理及转换可以参考"简述->开源代码路径处理"

数据集目录参考如下：

```
├── tmp # 数据及预训练模型存放文件夹
	├── pouring_tfrecords #pouring数据集
	│	├──pouring_train-0-of-2.tfrecords
	│	├──pouring_train-1-of-2.tfrecords
	│	└──pouring_val-0-of-1.tfrecords
	└── resnet50v2_weight_tf_dim_ordering_tf_kernels_notop.h5 # ImageNet pre-trained ResNetV2-50
```



## 模型训练<a name="section715881518135"></a>
- 下载训练脚本。
- 开始训练。



    1. 启动训练之前，首先要配置程序运行相关环境变量。
    
    	环境变量配置信息参见章节：[训练环境准备]
    
    2. 单卡训练
        2. 1设置单卡训练参数（脚本位于TCC_ID0714_for_TensorFlow2.X/test/train_performance_1p.sh中），示例如下：
        	
        	nohup python3 $cur_path/../tcc/train.py \
            	--precision_mode=${precision_mode} \
            	--over_dump=${over_dump} \
            	--over_dump_path=${over_dump_path} \
            	--data_dump_flag=${data_dump_flag} \
            	--data_dump_step=${data_dump_step} \
            	--data_dump_path=${data_dump_path} \
    	      	--profiling=${profiling} \
    	      	--profiling_dump_path=${profiling_dump_path}\
    		    --alsologtostderr \
            	--force_train   > ${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log 2>&1 &
            	
        2. 2单卡训练指令（脚本位于TCC_ID0714_for_TensorFlow2.X/test/train_performance_1p.sh）,请确保下面例子中的“--data_path”修改为用户的数据路径，示例中将数据存放在/home/tcc中。
        
            bash train_performance_1p.sh --data_path=/home/tcc



<h2 id="迁移学习指导.md">高级参考</h2>

## 脚本和事例代码

```
TCC_ID0714_for_TensorFlow2.X
	├── LICENSE
	├── modelzoo_level.txt
	├── README.md
	├── requirements.txt  # 依赖
	├── tcc  #训练网络代码目录
	│   ├──config.py  #配置文件
	└── test  #训练脚本目录
    	├── train_performance_1p.sh
    	└── train_full_1p.sh  # 单p训练脚本
```

## 脚本参数<a name="section6669162441511"></a>

```
    --data_dir=${data_path} \      data path of training
    --batch_size=${batch_size} \   Total batch size for training,default:2
```

## 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡训练。模型存储路径为${cur_path}/output/$ASCEND_DEVICE_ID，包括训练的log以及checkpoints文件。以单卡训练为例，loss信息在文件${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。