## 目录
[TOC]

## 基本信息

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：** Reinforcement learning

**版本（Version）：1.1**

**修改时间（Modified） ：2021.10.23**

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

  模型超参在`chess-alpha-zero/src/chess_zero/configs/mini.py`，`chess-alpha-zero/src/chess_zero/configs/normal.py`，`chess-alpha-zero/src/chess_zero/configs/distributed.py`文件中进行了定义，默认选择mini.py
  
  配置文件。
  
  以下数据来自mini.py文件
  
  - Batch size: 384
  - Learning rate(LR): 0.001
  - Optimizer: AdamOptimizer（learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08）
  - Weight decay: 0.0001


## 支持特性

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 是       |
| 并行数据   | 是       |


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
  通过百度网盘（[link](https://pan.baidu.com/s/1HmlR20U5ubhR5KiRz4CJSA) ,提取码：jahh）获得data后，放置在chess-alpha-zero文件夹下亦可。

- 进行训练

  根据下面的模型训练部分的训练脚本

## 模型训练

- 单击“立即下载”，并选择合适的下载方式下载源码包。

- 启动训练之前，首先要配置程序运行相关环境变量。

  代码可以在cpu，gpu，npu环境下运行。

  环境变量配置信息参见：

     [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

- 以下步骤可以选择性进行执行

  - 自动数据预处理和监督学习

    ```bash
    python3 src/chess_zero/run.py --cmd sl
    ```

  - 以当前最优或最新模型训练

    ```bash
    python3 src/chess_zero/run.py --cmd self
    ```

  - 模型训练

    ```bash
    python3 src/chess_zero/run.py --cmd opt
    ```

  - 模型评估

    ```bash
    python3 src/chess_zero/run.py --cmd eval
    ```

  - 人机对战

    ```
    python3 src/chess_zero/run.py --cmd uci
    ```

## 脚本和示例代码

```
├── requirements.txt                    //模型依赖文件
├── README.md                           //代码说明文档
├── LICENSE.txt                         //license文件
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

## 训练过程

2.  参考脚本的模型存储路径为`chess-alpha-zero/logs`，训练脚本log中包括如下信息。

```
2021-10-23 10:44:31.409927: W tensorflow/core/platform/profile_utils/cpu_utils.cc:98] Failed to find bogomips in /proc/cpuinfo; cannot determine CPU frequency
2021-10-23 10:44:31.417572: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0xaaab1b2afba0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2021-10-23 10:44:31.417618: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2021-10-23 10:44:35,610@chess_zero.agent.model_chess DEBUG # loaded model digest = abd8b08a5cf7a3587132bf545e9e48ccddb9bf80d6b7f86f1551a3bb48bb2d20
2021-10-23 10:44:35,758@chess_zero.worker.optimize DEBUG # loading data from /home/ma-user/modelarts/user-job-dir/chess-alpha-zero/data/play_data/play_20211019-203107.118167.json
2021-10-23 10:44:35,879@chess_zero.worker.optimize DEBUG # loading data from /home/ma-user/modelarts/user-job-dir/chess-alpha-zero/data/play_data/play_20211019-202942.541856.json
2021-10-23 10:44:35,881@chess_zero.worker.optimize DEBUG # loading data from /home/ma-user/modelarts/user-job-dir/chess-alpha-zero/data/play_data/play_20211019-203338.612125.json
2021-10-23 10:44:35,881@chess_zero.worker.optimize DEBUG # loading data from /home/ma-user/modelarts/user-job-dir/chess-alpha-zero/data/play_data/play_20211019-203019.062189.json
2021-10-23 10:44:35,882@chess_zero.worker.optimize DEBUG # loading data from /home/ma-user/modelarts/user-job-dir/chess-alpha-zero/data/play_data/play_20211019-203830.382636.json
Using TensorFlow backend.
Using TensorFlow backend.
Using TensorFlow backend.
Using TensorFlow backend.
Using TensorFlow backend.
2021-10-23 10:44:59,993@chess_zero.worker.optimize DEBUG # loading data from /home/ma-user/modelarts/user-job-dir/chess-alpha-zero/data/play_data/play_20211019-202909.158720.json
2021-10-23 10:45:00,005@chess_zero.worker.optimize DEBUG # loading data from /home/ma-user/modelarts/user-job-dir/chess-alpha-zero/data/play_data/play_20211019-203708.229829.json
2021-10-23 10:45:02,800@chess_zero.worker.optimize DEBUG # loading data from /home/ma-user/modelarts/user-job-dir/chess-alpha-zero/data/play_data/play_20211019-203242.718128.json
2021-10-23 10:45:02,812@chess_zero.worker.optimize DEBUG # loading data from /home/ma-user/modelarts/user-job-dir/chess-alpha-zero/data/play_data/play_20211019-203549.557850.json
2021-10-23 10:45:02,823@chess_zero.worker.optimize DEBUG # loading data from /home/ma-user/modelarts/user-job-dir/chess-alpha-zero/data/play_data/play_20211019-203953.598073.json
2021-10-23 10:45:19,187@chess_zero.worker.optimize DEBUG # loading data from /home/ma-user/modelarts/user-job-dir/chess-alpha-zero/data/play_data/play_20211019-204138.046049.json
Train on 113192 samples, validate on 2311 samples
```

## 推理/验证过程

1.  通过“模型训练”中的测试指令启动测试。

2.  当前只能针对该工程训练出的checkpoint进行推理测试。


```
Epoch 1/1

   384/113192 [..............................] - ETA: 1:46:34 - loss: 5.0509 - policy_out_loss: 3.0319 - value_out_loss: 0.6021
   768/113192 [..............................] - ETA: 1:31:39 - loss: 4.8202 - policy_out_loss: 2.8628 - value_out_loss: 0.5829
  1152/113192 [..............................] - ETA: 1:24:34 - loss: 4.7062 - policy_out_loss: 2.8041 - value_out_loss: 0.5422
  1536/113192 [..............................] - ETA: 1:20:46 - loss: 4.6641 - policy_out_loss: 2.7708 - value_out_loss: 0.5415
  1920/113192 [..............................] - ETA: 1:19:45 - loss: 4.6732 - policy_out_loss: 2.7797 - value_out_loss: 0.5393
  2304/113192 [..............................] - ETA: 1:17:56 - loss: 4.7085 - policy_out_loss: 2.8048 - value_out_loss: 0.5430
  2688/113192 [..............................] - ETA: 1:15:41 - loss: 4.7036 - policy_out_loss: 2.8043 - value_out_loss: 0.5385
  3072/113192 [..............................] - ETA: 1:13:35 - loss: 4.7175 - policy_out_loss: 2.8193 - value_out_loss: 0.5334
  3456/113192 [..............................] - ETA: 1:12:22 - loss: 4.7451 - policy_out_loss: 2.8440 - value_out_loss: 0.5297
  3840/113192 [>.............................] - ETA: 1:11:15 - loss: 4.7728 - policy_out_loss: 2.8726 - value_out_loss: 0.5214
  4224/113192 [>.............................] - ETA: 1:10:24 - loss: 4.7734 - policy_out_loss: 2.8765 - value_out_loss: 0.5168
  4608/113192 [>.............................] - ETA: 1:09:30 - loss: 4.7658 - policy_out_loss: 2.8731 - value_out_loss: 0.5131
  4992/113192 [>.............................] - ETA: 1:08:30 - loss: 4.7589 - policy_out_loss: 2.8682 - value_out_loss: 0.5120
  5376/113192 [>.............................] - ETA: 1:07:44 - loss: 4.7482 - policy_out_loss: 2.8616 - value_out_loss: 0.5092
  5760/113192 [>.............................] - ETA: 1:07:00 - loss: 4.7077 - policy_out_loss: 2.8324 - value_out_loss: 0.5050
  6144/113192 [>.............................] - ETA: 1:06:18 - loss: 4.6858 - policy_out_loss: 2.8149 - value_out_loss: 0.5045
  6528/113192 [>.............................] - ETA: 1:05:47 - loss: 4.6624 - policy_out_loss: 2.7976 - value_out_loss: 0.5025
  6912/113192 [>.............................] - ETA: 1:05:12 - loss: 4.6473 - policy_out_loss: 2.7882 - value_out_loss: 0.4987
  7296/113192 [>.............................] - ETA: 1:04:42 - loss: 4.6291 - policy_out_loss: 2.7749 - value_out_loss: 0.4970
  7680/113192 [=>............................] - ETA: 1:04:14 - loss: 4.6271 - policy_out_loss: 2.7740 - value_out_loss: 0.4957
  8064/113192 [=>............................] - ETA: 1:03:41 - loss: 4.6206 - policy_out_loss: 2.7687 - value_out_loss: 0.4956
  8448/113192 [=>............................] - ETA: 1:03:07 - loss: 4.6143 - policy_out_loss: 2.7628 - value_out_loss: 0.4963
  8832/113192 [=>............................] - ETA: 1:02:41 - loss: 4.5958 - policy_out_loss: 2.7485 - value_out_loss: 0.4953
  9216/113192 [=>............................] - ETA: 1:02:19 - loss: 4.5853 - policy_out_loss: 2.7402 - value_out_loss: 0.4951
```