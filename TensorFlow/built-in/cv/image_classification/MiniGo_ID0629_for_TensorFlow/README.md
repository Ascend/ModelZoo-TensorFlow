- [基本信息](#基本信息.md)
- [概述](#概述.md)
- [训练环境准备](#训练环境准备.md)
- [快速上手](#快速上手.md)
- [迁移学习指导](#迁移学习指导.md)
- [高级参考](#高级参考.md)

## 基本信息

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Image Classification**

**版本（Version）：1.1**

**修改时间（Modified） ：2021.07.16**

**大小（Size）**_**：5M**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架，以Mupha为基础，以AlphaGo Zero为模型的极简Go引擎**

## 概述

-    Minigo 基于 Brian Lee 的 “ [MuGo](https://github.com/brilee/MuGo) ”（纯[自然的](https://github.com/brilee/MuGo) Python实现），这是 AlphaGo 发表于《 *自然*[》](https://www.nature.com/articles/nature16961)的第一篇论文[“用深度神经网络和树搜索掌握围棋游戏](https://www.nature.com/articles/nature16961)”的纯Python实现 。此实现增加了最新的 AlphaGo Zero 论文[“精通无人类知识的游戏”中](https://www.nature.com/articles/nature24270) 存在的功能和体系结构更改。最近，在[“使用通用强化学习算法通过自学掌握象棋和将棋”中，](https://arxiv.org/abs/1712.01815) 为 Chess 和 Shogi 扩展了此体系结构。这些论文通常会在 Minigo 文档中被删节为*AG*（对于AlphaGo），*AGZ*（对于AlphaGo Zero）和*AZ* （对于AlphaZero）。

- 参考论文：

    [https://github.com/brilee/MuGo](https://github.com/brilee/MuGo)

- 参考实现：

    [https://github.com/tensorflow/minigo](https://github.com/tensorflow/minigo)

- 适配昇腾 AI 处理器的实现：
  
    [https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/built-in/cv/image_classification/MiniGo_ID0629_for_TensorFlow](https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/built-in/cv/image_classification/MiniGo_ID0629_for_TensorFlow)

- 通过Git获取对应commit\_id的代码方法如下：

    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    ```

#### 默认配置<a name="section91661242121611"></a>
-   训练超参（单卡）：
  - Batch size: 128
  - Momentum: 0.9
  - LR scheduler: cosine
  - Learning rate(LR): [0.01, 0.001, 0.0001]
  - Optimizer: Momentum
  - Weight decay: 0.0001
  - Label smoothing: 0.1
  - Train_steps: 58500


#### 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 否       |
| 混合精度   | 是       |
| 数据并行   | 否       |


#### 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

#### 开启混合精度<a name="section20779114113713"></a>
相关代码示例。

```
config_proto = tf.ConfigProto(allow_soft_placement=True)
  custom_op = config_proto.graph_options.rewrite_options.custom_optimizers.add()
  custom_op.name = 'NpuOptimizer'
  custom_op.parameter_map["use_off_line"].b = True
  custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
  config_proto.graph_options.rewrite_options.remapping = RewriterConfig.OFF
  session_config = npu_config_proto(config_proto=config_proto)
```

## 训练环境准备

-  硬件环境和运行环境准备请参见《[CANN软件安装指南](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=installation-update)》
-  运行以下命令安装依赖。
```
pip3 install requirements.txt
```
说明：依赖配置文件requirements.txt文件位于模型的根目录


## 快速上手

#### 数据集准备<a name="section361114841316"></a>

- 模型训练使用minigo网络自对弈(selfplay)生成数据集，数据集制作方法如下：


```python
 #（Step1）初始化随机模型，生成预训练ckpt
 # 脚本位于  MiniGo_ID0629_for_TensorFlow/bootstrap.py  ，示例如下：
 python3 bootstrap.py --work_dir=estimator_working_dir --export_path=outputs/models/000000-bootstrap
 # 该步骤初始化随机模型，将 ckpt 保存在 --work_dir ，同时选择最后一个 ckpt 存入 --export_path 作为最新模型，名为 000000-bootstrap ，以便后续 selfplay 可以使用此随机模型。

 #（Step2）使用随机模型自我对弈，生成训练数据集
 # 脚本位于  MiniGo_ID0629_for_TensorFlow/selfplay.py  ，示例如下：
 python3 selfplay.py --load_file=outputs/models/000000-bootstrap --num_readouts 10 --verbose 3 --selfplay_dir=outputs/data/selfplay --holdout_dir=outputs/data/holdout --sgf_dir=outputs/sgf
 # 该步骤使用最新随机模型 000000-bootstrap 自我对弈，生成训练数据存入 --selfplay_dir ，SGF目录存入 --sgf_dir。

 # 说明：
 # 1. 该步骤运行一次，只可生成一个数据样本。若要生成多个数据，循环调用该步骤即可。耗时较长，请耐心等待。
 # 2. 基于minigo网络原理，建议一次生成最多2000个数据，投入训练后得到最新模型，然后使用该最新模型替换 000000-bootstrap 继续自对弈生成训练数据，再投入训练。。。如此循环往复，可有效提升模型训练效果。
 # 3. 参数说明：
 --load_file：指定一个模型
 --num_readouts：每次移动要进行多少次搜索
 --verbose：每次移动会打印计时信息和统计数据。如果 >= 3 ，将在每次移动时打印一个 board
```

- 生成数据集后，在训练脚本中指定数据集路径，即可正常使用。

#### 模型训练<a name="section715881518135"></a>
- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练。
  
    1. 启动训练之前，首先要配置程序运行相关环境变量。

       环境变量配置信息参见：

          [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)

    2. 单卡训练（仅测试性能&功能）
       
        2.1 设置单卡训练参数（脚本位于./MiniGo_ID0629_for_TensorFlow/test/train_performance_1p.sh），示例如下。
            
        
        ```python
        # 训练steps
        train_steps=500
        # 训练batch_size
        batch_size=128
        ```
        

        2.2 单卡训练指令（脚本位于./MiniGo_ID0629_for_TensorFlow/test/train_performance_1p.sh） 

        ```
        于终端中运行export ASCEND_DEVICE_ID=0 (0~7)以指定单卡训练时使用的卡
        bash train_performance_1p.sh --data_path=xx
        数据集路径默认为 MiniGo_ID0629_for_TensorFlow/outputs/data/selfplay（即Step2自对弈的生成路径，不建议改动）
        数据集应有如下结构（数据切分可能不同），配置data_path时需指定为selfplay这一层，例：--data_path=./outputs/data/selfplay
        ├─selfplay
           ├─...
           ├─...
        ```

    3. 单卡训练
       
        3.1 设置单卡训练参数（脚本位于./MiniGo_ID0629_for_TensorFlow/test/train_full_1p.sh），示例如下。
            
        
        ```python
        # 训练steps
        train_steps=80000
        # 训练batch_size
        batch_size=128
        ```
        

        3.2 单卡训练指令（脚本位于./MiniGo_ID0629_for_TensorFlow/test/train_full_1p.sh） 

        ```
        于终端中运行export ASCEND_DEVICE_ID=0 (0~7)以指定单卡训练时使用的卡
        bash train_full_1p.sh --data_path=xx
        数据集路径默认为 MiniGo_ID0629_for_TensorFlow/outputs/data/selfplay（即Step2自对弈的生成路径，不建议改动）
        数据集应有如下结构（数据切分可能不同），配置data_path时需指定为selfplay这一层，例：--data_path=./outputs/data/selfplay
        ├─selfplay
           ├─...
           ├─...
        ```

    4. 8卡训练
       
        4.1 首先检查 MiniGo_ID0629_for_TensorFlow/test 目录下是否有存在8卡IP的json配置文件 "8p.json"

​              4.2 设置单卡训练参数（脚本位于./MiniGo_ID0629_for_TensorFlow/test/train_full_8p.sh），示例如下。

        python
        # 训练steps
        train_steps=80000
        # 训练batch_size
        batch_size=128

​              4.3 单卡训练指令（脚本位于./MiniGo_ID0629_for_TensorFlow/test/train_full_8p.sh） 

    ```
    bash train_full_8p.sh --data_path=xx
    数据集路径默认为 MiniGo_ID0629_for_TensorFlow/outputs/data/selfplay（即Step2自对弈的生成路径，不建议改动）
    数据集应有如下结构（数据切分可能不同），配置data_path时需指定为selfplay这一层，例：--data_path=./outputs/data/selfplay
    ├─selfplay
       ├─...
       ├─...
    ```

5. 模型评估

    5.1 单卡评估指令（脚本位于./MiniGo_ID0629_for_TensorFlow/evaluate.py） 

    ```
    python3 evaluate.py --eval_sgf_dir=outputs/evals --num_evaluation_games=3 black_model_file white_model_file
    # 该步骤使用训练后的模型进行对弈(可视化)，结果储存为 .sgf  文件。
    # 参数说明：
    --eval_sgf_dir：结果保存路径
    --num_evaluation_games：游戏局数
    black_model_file, white_model_file：黑白方所用模型文件
    ```

6. 交互式人机对弈

    6.1 单卡交互式人机对弈指令（脚本位于./MiniGo_ID0629_for_TensorFlow/gtp.py） 

    ```
    python3 gtp.py --load_file=outputs/models/000001-first_generation --num_readouts=400 --verbose=3
    # 该步骤借助GTP平台使用训练好的模型进行交互式人机对弈    
    # 在加载一些消息后，会显示 “GTP engine ready” , 此时它可以接收命令：
    # (Step1) 打印棋盘
    showboard
    #  (Step2) 黑方先行(我方) play 颜色 位置
    play black K11
    # (Step3) 将走棋权交给白方(电脑) genmove 颜色
    genmove white
    ```


## 迁移学习指导

#### 数据集准备。

1.  获取数据。

    1.1 方案一：自对弈(selfplay)生成数据集

    ```
    参考 “快速上手”  - 数据集准备。
    ```

    1.2 方案二：使用.sgf格式围棋棋谱生成数据集

    ```
    基于minigo网络原理，方案一 为首选方案，但 方案二 使用高质量围棋棋谱制作数据集，可有效提升模型训练效果。
    具体教程参考： MiniGo_ID0629_for_TensorFlow/sgf_DIY_Dataset.md
    ```
    
    1.3 数据集存放路径

    ```
    数据集生成后，建议将生成的数据集放入 MiniGo_ID0629_for_TensorFlow/outputs/data/selfplay
    也可自定义路径，但注意训练时 --data_path 需作相应修改。
    ```

#### 模型训练

请参考“快速上手”章节

## 高级参考

#### 脚本和示例代码<a name="section08421615141513"></a>

    ├── README.md                            //代码说明文档
    ├── bootstrap.py                         //初始化随机模型
    ├── selfplay.py                          //自对弈
    ├── train.py                             //网络训练
    ├── evaluate.py                          //模型评估
    ├── gtp.py                               //GTP平台交互式人机对弈
    ├── sgf_DIY_Dataset.md                   //自制数据集说明文档
    ├── sgf_file_check.py                    //检查 sgf 文件内容
    ├── sgf_to_tfrecord.py                   //制作数据集
    ├── test
    │    ├──train_full_1p.sh                 //单卡运行启动脚本(train_steps=80000)
    │    ├──train_full_8p.sh                 //8卡执行脚本(train_steps=80000)
    │    ├──train_performance_1p.sh          //单卡运行启动脚本(train_steps=500)
    │    ├──train_performance_8p.sh          //8卡执行脚本(train_steps=500)
    │    ├──env.sh                           //环境变量配置文件
    │    ├──8p.json                          //8p配置文件  


#### 脚本参数<a name="section6669162441511"></a>

```
--work_dir                    模型ckpt保存路径
--export_path                 模型导出路径
--load_file                   指定一个模型
--num_readouts                每次移动要进行多少次搜索
--verbose                     每次移动会打印计时信息和统计数据。如果 >= 3 ，将在每次移动时打印一个 board
--batch_size                  训练的batch size
--train_steps                 训练的steps
--data_path                   数据集路径
--eval_sgf_dir                评估结果保存路径
--num_evaluation_games        评估游戏局数
--lr                          学习率，默认[0.01, 0.001, 0.0001]
--weight_decay                权重衰减，默认: 1e-4
--momentum                    动量，默认：0.9
--rank_size                   使用NPU卡数量，默认：1
```

#### 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡训练。
将训练脚本（train_full_1p.sh）中的data_path设置为训练数据集的路径。具体的流程参见“模型训练”的示例。