## 目录
[TOC]

## 基本信息

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：** Reinforcement learning

**修改时间（Modified） ：2022.05.09**

**大小（Size）：33M**

**框架（Framework）：TensorFlow 1.15.2**

**模型格式（Model Format）：h5**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架实现的棋类对弈Alpha Zero强化学习代码** 

## 概述

Alpha Zero是一个经典的强化学习的实例，主要特点是减少先验知识输入，主要通过自训练的方式进行学习。采用8个Conv-BN-Relu-Conv-BN-Add-Relu的残差块构建bottleneck layer，通过1\*1卷积网络学习得到策略头和价值头。 

- 参考论文：

    [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm]([1712.01815.pdf (arxiv.org)](https://arxiv.org/pdf/1712.01815.pdf))

    [Mastering the Game of Go without Human Knowledge](https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ)

- 参考实现：

    https://github.com/Zeta36/chess-alpha-zero

- 适配昇腾 AI 处理器的实现：
  
  https://gitee.com/ascend/modelzoo/tree/master/contrib/TensorFlow/Research/reinforcement_learning/chess-alpha-zero_for_Tensorflow

## 默认配置

- 初始数据集获取：

  算法需要国际象棋的相关棋谱记录文件（.pgn文件）进行，相关记录文件可以在[FICS](http://ficsgames.org/download.html)下载获得，将文件存放到`chess-alpha-zero/data/play_data`文件夹下，在chess-alpha-zero文件夹下执行命令

  ``````bash
  python3 src/chess_zero/run.py --cmd sl
  ``````

  将在`chess-alpha-zero/data/play_data`生成所需要的json文件

- 训练超参

  模型超参在`chess-alpha-zero/src/chess_zero/configs/mini.py`，`chess-alpha-zero/src/chess_zero/configs/normal.py`，`chess-alpha-zero/src/chess_zero/configs/distributed.py`文件中进行了定义，默认选择mini.py配置文件。
  
  以下数据来自`chess-alpha-zero/src/chess_zero/configs/mini.py`文件
  
  - Batch size: 384
  - Learning rate(LR): 0.001
  - Optimizer: AdamOptimizer（learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08）
  - Weight decay: 0.0001


## 支持特性

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 是       |
| 并行数据   | 是       |

## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

脚本已默认开启混合精度

## 训练环境准备

1.  硬件环境准备请参见各硬件产品文档"[驱动和固件安装升级指南]( https://support.huawei.com/enterprise/zh/category/ai-computing-platform-pid-1557196528909)"。需要在硬件设备上安装与CANN版本配套的固件与驱动。
2.  宿主机上需要安装Docker并登录[Ascend Hub中心](https://ascendhub.huawei.com/#/detail?name=ascend-tensorflow-arm)获取镜像。

    当前模型支持的镜像列表如[表1](#zh-cn_topic_0000001074498056_table1519011227314)所示。

    **表 1** 镜像列表

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


## 快速上手

- 数据集准备

  算法需要国际象棋的相关棋谱记录文件（.pgn文件）进行，相关记录文件可以在[FICS](http://ficsgames.org/download.html)下载获得，将文件存放到`chess-alpha-zero/data/play_data`文件夹下。

  通过obs可以获得已经经过预处理的数据（obs://zjy-lenet/work/data/）,获得data后，放置在chess-alpha-zero文件夹下亦可。

- 进行训练

  根据下面的模型训练部分的训练脚本

## 模型训练

- 启动训练之前，首先要配置程序运行相关环境变量。

  代码可以在cpu，gpu，npu环境下运行。

  环境变量配置信息参见：

     [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

- 环境配置
  
  安装代码所需要的python包

  ```bash
  pip install -r requirements.txt
  ```

- 以下步骤可以选择性进行执行

  - 自动数据预处理

    ```bash
    python3 src/chess_zero/run.py --cmd sl
    ```

  - NPU单卡精度训练

    ```bash
    python3 src/chess_zero/run.py --cmd opt --npu
    ```

  - NPU单卡性能训练

    ```bash
    python3 src/chess_zero/run.py --cmd opt --epochs 10 --npu
    ```

## 脚本和示例代码

```
├── requirements.txt                    //模型依赖文件
├── README.md                           //代码说明文档
├── LICENSE.txt                         //license文件
├── modelzoo_level.txt
├── data
│    ├──model                           //已经存在的模型
│    |    ├──model_best_config.json     //最优模型结构记录
│    |    ├──model_best_weight.h5       //最优模型权重记录
│    |    ├──next_generation            //所有训练的模型记录（自动生成）
├── src/chess_zero
│    ├──run.py                          //启动文件
│    ├──manager.py                      //命令解析
│    ├──config.py                       //配置定义
│    ├──agent                           
│    |    ├──api_chess.py               //使用中的管道定义等
│    |    ├──model_chess.py             //训练模型定义
│    |    ├──player_chess.py            //对战过程的对象和方法定义
│    ├──configs                         
│    |    ├──distributed.py             //分布式配置文件
│    |    ├──mini.py                    //最基本配置文件
│    |    ├──normal.py                  //一般配置文件
│    ├──env                             
│    |    ├──chess_env.py               //对战环境定义
│    ├──lib                             
│    |    ├──data_helper.py             //数据记录工具类
│    |    ├──logger.py                  //日志记录工具类
│    |    ├──model_helper.py            //模型记录工具类
│    |    ├──tf_util.py                 //tf工具类
│    ├──play_game                       
│    |    ├──uci.py                     //人机对战记录类
│    ├──worker  
│    |    ├──evaluate.py                //评估过程的脚本
│    |    ├──optimize.py                //训练优化脚本
│    |    ├──self_play.py               //自我对弈过程脚本
│    |    ├──sl.py                      //监督学习脚本
```

## 训练过程和结果

1. 代码在GPU的运行结果

    运行环境：modelarts的notebook开发平台

    运行代码：obs://zjy-lenet/GPU-work/src/

    运行日志：obs://zjy-lenet/GPU-work/logs/gpu.log

2. 代码在NPU的运行结果
   
   运行环境：modelarts的pycharm插件进行训练作业的训练

   运行代码和日志：obs://zjy-lenet/MA-new-7-05-09-12-00/

3. 总结结果

    因为modelarts平台的notebook，kernel状态不稳定，会提前中断结果。
    
    又因为强化学习，我们将会比较相同的epoch数量下，GPU和NPU的精度（以loss记）和性能结果
   
   |平台|精度(强化学习，此处分析loss值)|性能(ms/step)|
   |----|----|----|
   |GPU-1p|1.6061|542.9|
   |NPU-1p|0.6934|322.6|