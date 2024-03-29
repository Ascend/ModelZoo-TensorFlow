-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [快速上手](#快速上手.md)
-   [迁移学习指导](#迁移学习指导.md)
-   [高级参考](#高级参考.md)
## 基本信息

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Object Detection**

**版本（Version）：1.1**

**修改时间（Modified） ：2021.09.15**

**大小（Size）：1.6M**

**框架（Framework）：TensorFlow_1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的EAST_v2检测网络训练代码**

## 概述

EAST是一种高效而准确的场景文本检测网络，EAST_v2是从EAST修改而来，包括数据增强，包括水平翻转、随机旋转、静音刻度和换位等。在后期处理中，实现了不同于 LANMS 的修改处理，效果更好，后期处理时间较短。     

-   参考论文：

    https://arxiv.org/pdf/1704.03155v2.pdf
    
-   参考实现：

    https://github.com/yangxuehang/east_v2
    
-   适配昇腾 AI 处理器的实现：
    
    
    https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/built-in/cv/detection/EastV2_ID1404_for_TensorFlow
        

-   通过Git获取对应commit\_id的代码方法如下：
    
    
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    

#### 默认配置<a name="section91661242121611"></a>

-   网络结构
    
    - resnet_v1_50, 
    
-   训练超参(单卡)：
    
    - Batch size: 16
    - Learning rate(LR): 0.001
    - moving_average_decay: 0.997
    - max_steps:1000
    
    ​	

#### 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 否       |
| 混合精度   | 是       |
| 数据并行   | 否       |


#### 混合精度训练<a name="section168064817164"></a>

 混合精度训练昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

#### 开启混合精度<a name="section20779114113713"></a>

    with tf.Session(config=npu_config_proto(config_proto=tf.ConfigProto(allow_soft_placement=True))) as sess:

## 训练环境准备

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
    <td class="cellrowborder" valign="top" width="25.52%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001074498056_p1450714271532"><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><em id="i72359495125"><a name="i72359495125"></a><a name="i72359495125"></a>21.0.2</em></p>
    </td>
    <td class="cellrowborder" valign="top" width="27.16%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001074498056_p18244640152312"><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><em id="i162363492129"><a name="i162363492129"></a><a name="i162363492129"></a><a href="https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software" target="_blank" rel="noopener noreferrer">5.0.2</a></em></p>
    </td>
    </tr>
    </tbody>
    </table>


## 快速上手

#### 数据集准备<a name="section361114841316"></a>

- 用户需自行准备训练数据集，例如icdar2015数据集

#### 预训练模型准备<a name="section361114841316"></a>

1. resnet_v1_50.ckpt，请参考github实现获取

#### 模型训练<a name="section715881518135"></a>

-  单击“立即下载”，并选择合适的下载方式下载源码包。
-  开始训练    
   
    1. 启动训练之前，首先要配置程序运行相关环境变量。

       环境变量配置信息参见：

          [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)
    
    2. 单卡训练 

        2.1 配置train_full_1p.sh脚本中`data_path,ckpt_path`（脚本路径EastV2_ID1404_for_TensorFlow/test/train_full_1p.sh）,请用户根据实际路径配置，数据集和预训练模型参数如下所示：

            --data_path=/icdar2015
            --ckpt_path=/pretrained_model

        2.2 单p指令如下:

            bash train_full_1p.sh --data_path=/icdar2015 --ckpt_path=/pretrained_model

-  验证。

    1. 执行完训练之后执行：
    
       ```
        python3 predict.py
       ```

## 迁移学习指导

#### 数据集准备。

获取数据。
请参见“快速上手”中的数据集准备。

#### 模型训练。

参考“模型训练”中训练步骤。

#### 模型评估。

参考“模型训练”中验证步骤。

## 高级参考

#### 脚本和示例代码

```
.
├── lanms
├── nets
├── post_processing
├── test
│   ├── env.sh
│   ├── train_full_1p.sh	// 训练脚本
│   └── train_performance_1p.sh // 训练脚本
├── utils
├── __init__.py
├── augment.py
├── config.py
├── data_utils.tpy										
├── data_utils_mt.py
├── dataset.py
├── LICENSE
├── locality_aware_nms.py
├── loss.py
├── model.py
├── modelzoo_level.txt
├── multigpu_train.py
├── predict.py	
├── README.md	
├── requirements.txt					
└── timers.py

```


#### 脚本参数<a name="section6669162441511"></a>

```
--data_path                     train data dir
--ckpt_path                     pretrained_model dir
--train_steps                   train step number
```

#### 训练过程<a name="section1589455252218"></a>

1. 通过“模型训练”中的训练指令启动单卡训练。
2. 将训练脚本（train_full_1p.sh）中的data_path设置为训练数据集的路径,ckpt_apth设置为预训练模型的路径。具体的流程参见“模型训练”的示例。
3. 模型存储路径为“curpath/output/ASCEND_DEVICE_ID”，包括训练的log以及checkpoints文件。
4. 以单卡训练为例，loss信息在文件curpath/output/{ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。

#### 推理/验证过程<a name="section1465595372416"></a>

1. 在1000个 step训练执行完成后,执行： 
   

```
python3 predict.py
```
