-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [快速上手](#快速上手.md)
-   [迁移学习指导](#迁移学习指导.md)
-   [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：** Human Pose Forecast

**版本（Version）：1.1**

**修改时间（Modified） ：2021.08.27**

**大小（Size）：141M**

**框架（Framework）：TensorFlow 1.5.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Research**

**描述（Description）：基于TensorFlow框架的Triangular-Prism RNN人体姿态预测网络训练代码** 

<h2 id="概述.md">概述</h2>


TP-RNN是一个人体姿态预测网络,该模型模型通过对不同时间尺度的时间相关性进行编码，捕捉了嵌入在时间人体姿态序列中的潜在层次结构。它引入了LSTM序列的多阶段分层多尺度上层，以更好地学习一系列不同粒度中不同时间步长之间的长期时间关系。


- 参考论文：

    [Hsu-kuang Chiu, Ehsan Adeli, Borui Wang, De-An Huang, Juan Carlos Niebles. “Action-Agnostic Human Pose Forecasting.” arXiv:1810.09676v1](https://arxiv.org/pdf/1810.09676v1.pdf) 

- 开源代码：
 ```
https://github.com/eddyhkchiu/pose_forecast_wacv
 ```
-   适配昇腾 AI 处理器的实现：
```
https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/tprnn_ID1297_for_TensorFlow
```


-   通过Git获取对应commit\_id的代码方法如下：
```
git clone {repository_url}    # 克隆仓库的代码
cd {repository_name}    # 切换到模型的代码仓目录
git checkout  {branch}    # 切换到对应分支
git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
```


## 默认配置<a name="section91661242121611"></a>

- 网络结构
  - LSTM
  - M=2
  - K=2
  
- 训练超参

  - Batch size: 16
  - learning_rate_decay_factor: 0.95
  - learning_rate_step: 10000
  - Learning rate(LR): 0.005
  - Train epoch: 100000
  
## 支持特性 <a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 是       |
| 混合精度   | 是       |
| 数据并行   | 是       |

## 混合精度训练 <a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度 <a name="section20779114113713"></a>
在训练脚本中添加开启混合精度的代码。

相关代码示例。



```
config_proto = tf.ConfigProto()
##混合精度
custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("force_fp16")
```


<h2 id="训练环境准备.md">训练环境准备</h2>
1.  硬件环境准备请参见各硬件产品文档"[驱动和固件安装升级指南]( https://support.huawei.com/enterprise/zh/category/ai-computing-platform-pid-1557196528909)
    "。需要在硬件设备上安装与CANN版本配套的固件与驱动。
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

1. 模型训练使用H3.6M数据集
2. 数据集的处理可以参考"概述->开源代码"处理

   3.放入模型目录下，在训练脚本中指定数据集路径，可正常使用。

## 模型训练<a name="section715881518135"></a>

- 下载训练脚本。
  
- 开始训练。


1. 启动训练之前，首先要配置程序运行相关环境变量。

环境变量配置信息参见：

[Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)
    
2. 单卡训练

  单卡训练指令：python tprnn_train_human.py


## 脚本和示例代码<a name="section08421615141513"></a>

    ├── README.md                                //说明文档
    ├── requirements.txt						 //依赖
    ├── data
    │    ├──h3.6m 
    │        ├──dataset 
    ├── data_utils.py
    ├── load.py
    ├── tprnn_basic_model.py
    ├──	tprnn_generic_model.py
    ├──	tprnn_train_human.py
    ├──	tprnn_train_penn.py


## 脚本参数<a name="section6669162441511"></a>

```
--learning_rate                 learning rate， 默认：0.005
--learning_rate_decay_factor    learning_rate_decay_factor， default=0.95
--learning_rate_step            learning_rate_step， 默认：10000
--max_gradient_norm             max_gradient_norm， 默认：5
--batch_size                    每个NPU的batch size， 默认：16
--iterations                    trin ecoph， 默认：nt(1e5)
--rnn_size                      Size of each model layer，默认：1024
--seq_length_in                 Number of frames to feed into the encoder，默认：50
--seq_length_out                Number of frames that the decoder has to predict，默认：10
--omit_one_hot                  Whether to remove one-hot encoding from the data，默认：True
--action                        action，默认：all
--test_every                    每多少步进行一次测试，默认：1000
--save_every                    每多少步保存一次模型，默认：1000
--sample                        默认：False
--use_cpu                       默认：False
--load                          默认：0
--dataset                       数据类型，默认：human
--tprnn_scale                   默认：2
--tprnn_layers                  默认：2
--more                          默认：0
--dropout_keep                  默认：1.0
--model                         默认：basic'
```


## GPU 训练/评估结果<a name="section1589455252218"></a>
walking	        | 0.25	| 0.41	| 0.58 	| n/a	| 0.74	| 0.77|

eating	        | 0.20	| 0.33 	| 0.53	| n/a	| 0.84	| 1.14|

smoking	        | 0.26	| 0.48	| 0.88	| n/a	| 0.98	| 1.16|

discussion	| 0.30	| 0.66	| 0.98	| n/a	| 1.39	| 1.74|

directions	| 0.38	| 0.59	| 0.75	| 0.83	| 0.95	| 1.38|

greeting	| 0.51 	| 0.86	| 1.27	| 1.44 	| 1.72	| 1.81|

phoning	        | 0.57 	| 1.08	| 1.44	| 1.59	| 1.47	| 1.68|

posing	        | 0.42	| 0.76	| 1.29 	| 1.54	| 1.75	| 2.47|

purchases	| 0.59	| 0.82	| 1.12	| 1.18	| 1.52	| 2.28|

sitting	        | 0.41	| 0.66	| 1.07 	| 1.22	| 1.35	| 1.74|

sittingdown	| 0.41	| 0.79	| 1.13	| 1.27	| 1.47	| 1.93|

takingphoto	| 0.26	| 0.51	| 0.80	| 0.95	| 1.08	| 1.35|

waiting	        | 0.30	| 0.60	| 1.09	| 1.31	| 1.71	| 2.46|

walkingdog 	| 0.53	| 0.93	| 1.24	| 1.38	| 1.73	| 1.98|


## NPU 训练/评估结果<a name="section1589455252218"></a>


walking          | 0.391 | 0.667 | 0.956 | 1.109 |   n/a |   n/a |

eating           | 0.278 | 0.474 | 0.704 | 0.834 |   n/a |   n/a |

smoking          | 0.263 | 0.489 | 0.970 | 0.957 |   n/a |   n/a |

discussion       | 0.333 | 0.686 | 0.964 | 1.059 |   n/a |   n/a |

directions       | 0.398 | 0.599 | 0.786 | 0.884 |   n/a |   n/a |

greeting         | 0.561 | 0.921 | 1.342 | 1.522 |   n/a |   n/a |

phoning          | 0.632 | 1.210 | 1.634 | 1.806 |   n/a |   n/a |

posing           | 0.252 | 0.525 | 1.109 | 1.345 |   n/a |   n/a |

purchases        | 0.626 | 0.886 | 1.188 | 1.263 |   n/a |   n/a |

sitting          | 0.403 | 0.638 | 1.030 | 1.182 |   n/a |   n/a |

sittingdown      | 0.402 | 0.749 | 1.091 | 1.215 |   n/a |   n/a |

takingphoto      | 0.258 | 0.521 | 0.800 | 0.921 |   n/a |   n/a |

waiting          | 0.355 | 0.678 | 1.215 | 1.472 |   n/a |   n/a |

walkingdog       | 0.599 | 0.961 | 1.333 | 1.487 |   n/a |   n/a |

Step-Time:24.4185ms