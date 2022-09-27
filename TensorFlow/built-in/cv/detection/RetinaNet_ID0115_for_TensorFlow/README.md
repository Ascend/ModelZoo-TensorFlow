# RetinaNet_for_TensorFlow
## 目录
* [基本信息](#基本信息)
* [概述](#概述)
* [训练环境准备](#训练环境准备)
* [快速上手](#快速上手)
* [高级参考](#高级参考)


## 基本信息

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Object Detection**

**版本（Version）：1.1**

**修改时间（Modified） ：2021.04.29**

**大小（Size）：270M**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的RetinaNet代码**

## 概述

  　RetinaNet算法源自2018年Facebook AI Research的论文Focal Loss for Dense Object Detection。该论文最大的共享在于提出了Focal Loss用于解决类别不均衡问题，从而创造了RetinaNet（One Stage目标检测算法）这个精度超越景点Two Stage的Faster-RCNN的目标检测网络。

- 参考论文：
  

　　Focal loss for dense object detection

　　https://arxiv.org/pdf/1708.02002.pdf
    
- 参考实现：

  https://github.com/MingtaoGuo/RetinaNet_TensorFlow

- 适配昇腾 AI 处理器的实现：    
  
    https://gitee.com/ascend/modelzoo/tree/master/built-in/TensorFlow/Research/cv/detection/RetinaNet_ID0115_for_TensorFlow

- 通过Git获取对应commit\_id的代码方法如下：
  
    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
        
    ```


### 默认配置

-   网络结构
    - resnet_v2_50

-   训练数据集预处理

    - NA

-   训练超参（单卡）：
    -  BATCH_SIZE = 2
    -  IMG_H = 512
    -  IMG_W = 512
    -  WEIGHT_DECAY = 0.0001
    -  LEARNING_RATE = 0.001 


### 支持特性

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 是    |
| 混合精度  | 是|
| 数据并行  | 是    |

### 混合精度训练

```
　　昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。
```

### 开启混合精度
   相关代码示例。　
```
   flags.DEFINE_string(name='precision_mode', default= 'allow_fp32_to_fp16',
                    help='allow_fp32_to_fp16/force_fp16/force_fp32/ ' 
                    'must_keep_origin_dtype/allow_mix_precision.')
   
   custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes(FLAGS.precision_mode)
```


## 训练环境准备

1.  硬件环境准备请参见各硬件产品文档"[驱动和固件安装升级指南]( https://support.huawei.com/enterprise/zh/category/ai-computing-platform-pid-1557196528909)"。需要在硬件设备上安装与CANN版本配套的固件与驱动。_
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
### 数据集准备<a name="section361114841316"></a>

1. 训练使用数据集 [VOC2007]，参考github源代码提供的路径下载。
2. 训练使用预训练模型 [resnet_v2_50]，参考github源代码提供的路径下载。

### 模型训练<a name="section715881518135"></a>

- 下载训练脚本（单击“立即下载”，并选择合适的下载方式下载源码包。）

- 开始训练

    1. 启动训练之前，首先要配置程序运行相关环境变量。

       环境变量配置信息参见：

          [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)
    
    2. 配置train_full_1p.sh中data_path和ckpt_path训练参数（脚本位于RetinaNet_for_TensorFlow/test目录下），请用户根据实际路径进行配置，确保data_path和ckpt_path下存在已准备的数据集和预训练模型，如下所示（仅供参考）：
               
        ```
                --data_path=./VOCdevkit
                --ckpt_path=./resnet_ckpt
        ```

    3.  单卡训练。
    
    单卡训练指令如下（脚本位于RetinaNet_for_TensorFlow/test目录下）：
    
    ```
            bash train_full_1p.sh --data_path="数据集路径" --ckpt_path="预训练模型路径"
    ```


    4.  8卡训练。
        
        8卡训练指令如下（脚本位于RetinaNet_for_TensorFlow/test目录下）：
        ```
            bash train_full_8p.sh --data_path="数据集路径" --ckpt_path="预训练模型路径"
        ```


- 开始推理。

```
　　NA
```

## 高级参考
### 脚本和示例代码<a name="section08421615141513"></a>

```

└─RetinaNet_for_TensorFlow
    ├─IMGS
    ├─npu_configs
    |   ├─rank_table_1p.json
    |   └─rank_table_8p.json
    ├─test├─env.sh
    |     ├─train_full_1p.sh
    |     ├─train_full_8p.sh
    |     ├─train_performance_1p.sh
    |     └─train_performance_8p.sh
    |  
    ├─LICENSE
    ├─README.md 
    ├─config.py
    ├─networks.py
    ├─ops.py
    ├─resnet.py
    ├─test.py
    ├─train.py
    └─utils.py
```


### 脚本参数<a name="section6669162441511"></a>

```
#Training 
--data_path                    the path of train data. 
--ckpt_path                    the resnet_ckpt for train. 
--model_dir                    the path of save ckpt. 
--precision_mode               allow_fp32_to_fp16/force_fp16/. 
--over_dump                    if or not over detection, default is False. 
--data_dump_flag               data dump flag, default is False. 
--data_dump_step               data dump step, default is 10. 
--profiling                    if or not profiling for performance debug, default is False. 
--profiling_dump_path          the path to save profiling data. 
--over_dump_path               the path to save over dump data. 
--data_dump_path               the path to save dump data.
--autotune                     whether to enable autotune, default is False.
```

### 训练过程<a name="section1589455252218"></a>

```
　　NA
```

### 推理/验证过程<a name="section1465595372416"></a>

```
　　NA
```
