# XLNet_ID0025_for_TensorFlow

## 目录
* [基本信息](#基本信息)
* [概述](#概述)
* [训练环境准备](#训练环境准备)
* [快速上手](#快速上手)
* [高级参考](#高级参考)

## 基本信息

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Natural Language Processing**

**版本（Version）：1.1**

**修改时间（Modified） ：2021.04.06**

**大小（Size）：415KB**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的XLNET网络训练代码**

## 概述

   XLNet是一种基于新的广义置换语言建模目标的新的无监督语言表示学习方法。此外，XLNet使用[Transformer-XL](https://arxiv.org/abs/1901.02860) 作为主干模型，对涉及长上下文的语言任务表现出出色的性能。总体而言，XLNet在各种下游语言任务上实现了最先进的（SOTA）结果，包括问题回答、自然语言推理、情绪分析和文档排名。

-   参考论文：

        https://arxiv.org/abs/1906.08237

-   参考实现：
        
        ```
        https://github.com/zihangdai/xlnet
        ```
    
-   适配昇腾 AI 处理器的实现：
    
        ```
        https://gitee.com/ascend/modelzoo/tree/master/built-in/TensorFlow/Research/nlp/XLNet_ID0025_for_TensorFlow
        ```


-   通过Git获取对应commit\_id的代码方法如下：
    
        ```
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
        ```

### 默认配置<a name="section91661242121611"></a>

  -   初始学习率为5e-5 learning rate
  -   优化器：Adam
  -   单卡batchsize：8
  -   4卡batchsize：4*8
  -   steps数设置为1200
  -   Weight decay为0.0000


### 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 是       |
| 混合精度   | 是       |
| 数据并行   | 是       |


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

## 安装依赖
参照：requirements.txt

## 快速上手

### 数据集准备<a name="section361114841316"></a>

1、模型使用GLUE data数据集，请用户自行下载。
2、模型使用XLNet-Large预训练模型，请用户自行下载。

### 模型训练<a name="section715881518135"></a>
```
 单P训练，以数据目录为./glue_data/STS-B/、checkpoints目录为 ./LARGE_dir/xlnet_cased_L-24_H-1024_A-16 为例:
  cd test
  source ./env.sh
  bash train_full_1p.sh （全量）
  bash train_performance_1p.sh （功能测试）
  
4p训练,以数据目录为./glue_data/STS-B/、checkpoints目录为 ./LARGE_dir/xlnet_cased_L-24_H-1024_A-16 为例:
  cd test
  source ./env.sh
  bash train_full_4p.sh
```

### 验证
```
基于ckpt做eval，以数据目录为./glue_data/STS-B/、checkpoints目录为 ./test/output/ckpt_npu 为例:
  cd test
  source ./env.sh
  bash eval.sh
```

## 高级参考

### 脚本和示例代码

```
.
├── configs
│   ├── 1p.json
│   ├── 4p.json	
├── scripts
│   ├── gpu_squad_base.sh
│   ├── tpu_race_large_bsz32.sh
│   ├── tpu_race_large_bsz8.sh
│   └── tpu_squad_large.sh
├── test
│   ├── env.sh
│   ├── train_full_1p.sh	
│   ├── train_full_4p.sh	
│   ├── train_performance_1p.sh	
│   └── train_performance_4p.sh 
├── LICENSE
├── README.md
├── __init__.py
├── classifier_utils.py
├── data_utils.py
├── function_builder.py
├── gpu_utils.py
├── model_utils.py
├── modeling.py
├── modelzoo_level.txt
├── prepro_utils.py
├── requirements.txt
├── run_classifier.py
├── run_race.py
├── run_squad.py
├── squad_utils.py
├── tpu_estimator.py
├── train.py	
├── train_gpu.py						
└── xlnet.py

```

### 脚本参数<a name="section6669162441511"></a>

```
    --data_path                       train data dir, default : path/to/data
    --precision_mode                  precision mode,default:allow_mix_precision
    --over_dump                       overflow dump flag,default:false
    --data_dump_flag                  data dump flag, default:false
    --data_dump_step                  dump when step is equal to data_dump_step
    --profiling                       profiling flag,default:false
    --profiling		              if or not profiling for performance debug, default is False
    --h/--help                        show help message 
```


