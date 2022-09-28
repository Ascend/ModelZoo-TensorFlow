- [基本信息](#基本信息.md)
- [概述](#概述.md)
- [训练环境准备](#训练环境准备.md)
- [快速上手](#快速上手.md)
- [迁移学习指导](#迁移学习指导.md)
- [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Reinforcement Learning**

**版本（Version）：1.1**

**修改时间（Modified） ：2022.8.23**

**大小（Size）：33MB**

**框架（Framework）：TensorFlow1.15.0**

**模型格式（Model Format）：h5**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架实现的棋类对弈Alpha Zero强化学习代码**

<h2 id="概述.md">概述</h2>

## 简述<a name="section194554031510"></a>

Alpha Zero是一个经典的强化学习的实例，主要特点是减少先验知识输入，主要通过自训练的方式进行学习。采用8个Conv-BN-Relu-Conv-BN-Add-Relu的残差块构建bottleneck layer，通过1\*1卷积网络学习得到策略头和价值头。

- 参考论文：

  [https://arxiv.org/pdf/1712.01815.pdf](Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm)

- 参考实现：

  https://github.com/Zeta36/chess-alpha-zero

- 适配昇腾 AI 处理器的实现：
  
  https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/reinforcement-learning/Chess-Alpha-Zero_ID2100_for_TensorFlow

- 通过Git获取对应commit\_id的代码方法如下：
  
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    

## 默认配置<a name="section91661242121611"></a>

-   训练超参（单卡）：
    - cmd
    - new
    - type
    - total-step
    - epochs
    - npu


## 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 是       |
| 混合精度   | 是       |
| 数据并行   | 是       |


## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

拉起脚本中，传入--precision_mode='allow_mix_precision'

```
 ./train_performance_1p_16bs.sh --help

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
custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
```

<h2 id="训练环境准备.md">训练环境准备</h2>

-  硬件环境和运行环境准备请参见《[CANN软件安装指南](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=installation-update)》
-  运行以下命令安装依赖。
```
pip3 install requirements.txt
```
说明：依赖配置文件requirements.txt文件位于模型的根目录

<h2 id="快速上手.md">快速上手</h2>

## 数据集准备<a name="section361114841316"></a>

1、用户自行准备好数据集（方法见https://github.com/Zeta36/chess-alpha-zero）

2、Chess-Alpha-Zero训练的模型及数据集可以参考"简述 -> 参考实现"


## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练。

    - 启动训练之前，首先要配置程序运行相关环境变量。

      环境变量配置信息参见：

      [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)

    - 单卡训练

         1. 训练需要传入的参数，根据训练自行配置

              ```
              --cmd opt --npu --epochs 10
              ```

              

         2. 训练执行的脚本

             ```
             bash train_full_1p.sh
             ```



         3. 精度训练执行结果

           |平台|精度(强化学习，此处分析loss值)|性能(ms/step)|
           |----|----|----|
           |GPU-1p|1.6061|542.9|
           |NPU-1p|0.6934|322.6|
                  


<h2 id="高级参考.md">高级参考</h2>

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

## 脚本参数<a name="section6669162441511"></a>

```
--cmd
--new
--type
--total-step
--epochs
--npu
```

## 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡，8卡网络训练。模型存储路径为${cur_path}/output/$ASCEND_DEVICE_ID，包括训练的log以及checkpoints文件。以8卡训练为例，loss信息在文件${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。