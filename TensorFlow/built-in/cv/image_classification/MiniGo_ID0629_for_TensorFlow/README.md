# MiniGo_for_TensorFlow
## 目录
* [基本信息](#基本信息)
* [概述](#概述)
* [训练环境准备](#训练环境准备)
* [迁移学习指导](#迁移学习指导)
* [快速上手](#快速上手)
* [高级参考](#高级参考)


## 基本信息

**发布者（Publisher）：Huawei**
**应用领域（Application Domain）：Image Classification
**版本（Version）：1.1
**修改时间（Modified） ：2021.07.16
**大小（Size）：5M
**框架（Framework）：TensorFlow 1.15.0
**模型格式（Model Format）：ckpt
**精度（Precision）：Mixed
**处理器（Processor）：昇腾910
**应用级别（Categories）：Official
**描述（Description）：基于TensorFlow框架，以Mupha为基础，以AlphaGo Zero为模型的极简Go引擎

## 概述


Minigo基于Brian Lee的“ [MuGo](https://github.com/brilee/MuGo) ”（纯[自然的](https://github.com/brilee/MuGo)Python实现），这是AlphaGo发表于《 *自然*[》](https://www.nature.com/articles/nature16961)的第一篇论文[“用深度神经网络和树搜索掌握围棋游戏](https://www.nature.com/articles/nature16961)”的纯Python实现 。此实现增加了最新的AlphaGo Zero论文[“精通无人类知识的游戏”中](https://www.nature.com/articles/nature24270)存在的功能和体系结构更改。最近，在[“使用通用强化学习算法通过自学掌握象棋和将棋”中，](https://arxiv.org/abs/1712.01815)为Chess和Shogi扩展了此体系结构。这些论文通常会在Minigo文档中被删节为*AG*（对于AlphaGo），*AGZ*（对于AlphaGo Zero）和*AZ* （对于AlphaZero）。

- 参考论文：

    https://github.com/brilee/MuGo

- 参考实现：
      
    https://github.com/tensorflow/minigo

- 适配昇腾 AI 处理器的实现：
    
    https://gitee.com/ascend/modelzoo/tree/master/built-in/TensorFlow/Research/cv/image_classification/MiniGo_ID0629_for_TensorFlow


- 通过Git获取对应commit\_id的代码方法如下：
    
    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    ```

### 默认配置<a name="section91661242121611"></a>

- 训练超参（单卡）

  - Batch size: 128
  - Momentum: 0.9
  - LR scheduler: cosine
  - Learning rate(LR): [0.01, 0.001, 0.0001]
  - Optimizer: Momentum
  - Weight decay: 0.0001
  - Label smoothing: 0.1
  - Train_steps: 58500

### 支持特性<a name="section1899153513554"></a>

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 是    |
| 混合精度  | 是    |
| 并行数据  | 是    |

### 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

### 开启混合精度<a name="section20779114113713"></a>

脚本已默认开启混合精度，设置precision_mode参数的脚本参考如下。

  ```
  run_config = NPURunConfig(        
  		model_dir=flags_obj.model_dir,        
  		session_config=session_config,        
  		keep_checkpoint_max=5,        
  		save_checkpoints_steps=5000,        
  		enable_data_pre_proc=True,        
  		iterations_per_loop=iterations_per_loop,        			
  		log_step_count_steps=iterations_per_loop,        
  		precision_mode='allow_mix_precision',        
  		hcom_parallel=True      
        )
  ```


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

### 数据集准备<a name="section361114841316"></a>

模型训练使用minigo网络自对弈(selfplay)生成数据集。

- （Step1）初始化随机模型，生成预训练ckpt

脚本位于  MiniGo_ID0629_for_TensorFlow/bootstrap.py  ，示例如下：

python3 bootstrap.py --work_dir=estimator_working_dir --export_path=outputs/models/000000-bootstrap

该步骤初始化随机模型，将 ckpt 保存在 --work_dir ，同时选择最后一个 ckpt 存入 --export_path 作为最新模型，名为 000000-bootstrap ，以便后续 selfplay 可以使用此随机模型。


- （Step2）使用随机模型自我对弈，生成训练数据集

脚本位于  MiniGo_ID0629_for_TensorFlow/selfplay.py  ，示例如下：

python3 selfplay.py --load_file=outputs/models/000000-bootstrap --num_readouts 10 --verbose 3 --selfplay_dir=outputs/data/selfplay --holdout_dir=outputs/data/holdout --sgf_dir=outputs/sgf

该步骤使用最新随机模型 000000-bootstrap 自我对弈，生成训练数据存入 --selfplay_dir ，SGF目录存入 --sgf_dir。

**注意：**

**1. 该步骤运行一次，只可生成一个数据样本。若要生成多个数据，循环调用该步骤即可。耗时较长，请耐心等待。**

**2. 另外基于minigo网络原理，建议一次生成最多2000个数据，投入训练后得到最新模型，然后使用该最新模型替换000000-bootstrap继续自对弈生成训练数据，再投入训练。。。如此循环往复，可有效提升模型训练效果。**

**3. 参数说明：**
    
    --load_file：指定一个模型
    
    --num_readouts：每次移动要进行多少次搜索
    
    --verbose：每次移动会打印计时信息和统计数据。如果 >= 3 ，将在每次移动时打印一个 board


### 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。

- 启动训练之前，首先要配置程序运行相关环境变量。

  环境变量配置信息参见：

     [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

- 单卡训练 

  1. 配置训练参数。

    脚本位于  MiniGo_ID0629_for_TensorFlow/test/train_full_1p.sh  ，可配置的参数如下：
    
    --batch_size=128   // 训练的batch size
    
    --train_steps=80000    //训练的steps，实际训练steps取决于 数据集的样本个数 和 train_steps 的较小者
    
    --data_path="$./outputs/data/selfplay"    //数据集路径默认为 MiniGo_ID0629_for_TensorFlow/outputs/data/selfplay(即Step2自对弈的生成路径)，不建议改动


  2. 启动训练。

    脚本位于  MiniGo_ID0629_for_TensorFlow/test/train_full_1p.sh  ，示例如下：
    
    bash train_full_1p.sh


- 8卡训练

  1. 配置训练参数。

    首先检查 MiniGo_ID0629_for_TensorFlow/test 目录下是否有存在8卡IP的json配置文件“8p.json”

    脚本位于  MiniGo_ID0629_for_TensorFlow/test/train_full_8p.sh  ，可配置的参数如下：
    
    --batch_size=128   // 训练的batch size
    
    --train_steps=80000    //训练的steps，实际训练steps取决于 数据集的样本个数 和 train_steps 的较小者
    
    --data_path="$./outputs/data/selfplay"    //数据集路径默认为 MiniGo_ID0629_for_TensorFlow/outputs/data/selfplay(即Step2自对弈的生成路径)，不建议改动


  2. 启动训练。

    脚本位于  MiniGo_ID0629_for_TensorFlow/test/train_full_8p.sh  ，示例如下：
    
    bash train_full_8p.sh


- 模型评估

    脚本位于  MiniGo_ID0629_for_TensorFlow/evaluate.py  ，示例如下：

    python3 evaluate.py --eval_sgf_dir=outputs/evals --num_evaluation_games=3 black_model white_model

    该步骤使用训练后的模型进行对弈(可视化)，结果储存为 .sgf  文件。

    参数说明：
    
    --eval_sgf_dir：结果保存路径
    
    --num_evaluation_games：游戏局数
    
    black_model white_model：黑白方所用模型


- 交互式人机对弈

    该步骤借助GTP平台使用训练好的模型进行交互式人机对弈

    脚本位于  MiniGo_ID0629_for_TensorFlow/gtp.py  ，示例如下：
    
    python3 gtp.py --load_file=outputs/models/000001-first_generation --num_readouts=400 --verbose=3

    在加载一些消息后，会显示 “GTP engine ready” , 此时它可以接收命令：

    (Step1) 打印棋盘
    
    showboard
    
    (Step2) 黑方先行(我方) play 颜色 位置
    
    play black K11
    
    (Step3) 将走棋权交给白方(电脑) genmove 颜色
    
    genmove white


## 迁移学习指导

### 数据集准备<a name="section361114841316"></a>

数据集生成后，建议将生成的数据集放入 MiniGo_ID0629_for_TensorFlow/outputs/data/selfplay/ ，也可自定义路径，但注意训练时 --data_path 需作相应修改。

- 方案一：自对弈(selfplay)生成数据集

    参考 “快速上手”  - 数据集准备。


- 方案二：使用.sgf格式围棋棋谱生成数据集 
    
    基于minigo网络原理，方案一为首选方案，但方案二使用高质量围棋棋谱制作数据集，可有效提升模型训练效果。
    
    具体教程参考： MiniGo_ID0629_for_TensorFlow/sgf_DIY_Dataset.md


### 模型训练<a name="section715881518135"></a>

    参考 “快速上手”  - 模型训练。

### 模型评估<a name="section715881518135"></a>
    
    参考 “快速上手”  - 模型评估。

### 交互式人机对弈<a name="section715881518135"></a>
    
    参考 “快速上手”  - 交互式人机对弈


## 高级参考

### 核心脚本和示例代码<a name="section08421615141513"></a>

```
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
```

### 脚本参数<a name="section6669162441511"></a>

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

### 训练过程<a name="section1589455252218"></a>

1.  通过 “模型训练” 中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡、8卡网络训练。

2.  训练脚本log中包括如下信息（仅作参考）：

```
INFO:tensorflow:loss = 3.6029148, step = 56401 (13.235 sec)
INFO:tensorflow:global_step/sec: 7.58255
INFO:tensorflow:loss = 3.2465415, step = 56501 (13.189 sec)
INFO:tensorflow:global_step/sec: 7.58776
INFO:tensorflow:loss = 3.657736, step = 56601 (13.179 sec)
INFO:tensorflow:global_step/sec: 7.60866
INFO:tensorflow:loss = 3.5327232, step = 56701 (13.143 sec)
INFO:tensorflow:global_step/sec: 7.54431
INFO:tensorflow:loss = 3.738246, step = 56801 (13.255 sec)
INFO:tensorflow:global_step/sec: 7.61954
INFO:tensorflow:loss = 3.2404513, step = 56901 (13.124 sec)
INFO:tensorflow:global_step/sec: 7.39826
INFO:tensorflow:loss = 3.4218066, step = 57001 (13.517 sec)
INFO:tensorflow:global_step/sec: 7.51395
INFO:tensorflow:loss = 3.3448925, step = 57101 (13.308 sec)
INFO:tensorflow:global_step/sec: 7.54161
INFO:tensorflow:loss = 3.2767594, step = 57201 (13.260 sec)
INFO:tensorflow:global_step/sec: 4.71074
INFO:tensorflow:loss = 3.64554, step = 57301 (21.228 sec)
INFO:tensorflow:global_step/sec: 7.61293
INFO:tensorflow:loss = 3.4691107, step = 57401 (13.136 sec)
INFO:tensorflow:global_step/sec: 7.52592
INFO:tensorflow:loss = 3.8170938, step = 57501 (13.287 sec)
INFO:tensorflow:global_step/sec: 7.58268
INFO:tensorflow:loss = 3.5368466, step = 57601 (13.188 sec)
INFO:tensorflow:global_step/sec: 7.5716
INFO:tensorflow:loss = 3.1940696, step = 57701 (13.207 sec)
INFO:tensorflow:global_step/sec: 7.59141
INFO:tensorflow:loss = 3.6414692, step = 57801 (13.173 sec)
INFO:tensorflow:global_step/sec: 7.5387
INFO:tensorflow:loss = 3.464244, step = 57901 (13.265 sec)
INFO:tensorflow:global_step/sec: 7.3802
INFO:tensorflow:loss = 3.649315, step = 58001 (13.550 sec)
INFO:tensorflow:global_step/sec: 7.57948
INFO:tensorflow:loss = 3.1356702, step = 58101 (13.193 sec)
INFO:tensorflow:global_step/sec: 7.57194
INFO:tensorflow:loss = 3.6425178, step = 58201 (13.207 sec)
INFO:tensorflow:global_step/sec: 7.54483
INFO:tensorflow:loss = 3.3034708, step = 58301 (13.254 sec)
INFO:tensorflow:global_step/sec: 7.54207
INFO:tensorflow:loss = 3.5668197, step = 58401 (13.259 sec)
INFO:tensorflow:global_step/sec: 7.54808
INFO:tensorflow:loss = 3.4597926, step = 58501 (13.248 sec)
INFO:tensorflow:Saving checkpoints for 58545 into /npu/debug00274026/MiniGo_ID0629_for_TensorFlow/test/..//estimator_working_dir/model.ckpt.
I0706 17:13:54.296661 281473570541584 basic_session_run_hooks.py:606] Saving checkpoints for 58545 into /npu/debug00274026/MiniGo_ID0629_for_TensorFlow/test/..//estimator_working_dir/model.ckpt.
INFO:tensorflow:Loss for final step: 3.4745815.
I0706 17:14:06.680798 281473570541584 utils.py:113] Training: 8057.882 seconds
Copying /npu/debug00274026/MiniGo_ID0629_for_TensorFlow/test/..//estimator_working_dir/model.ckpt-58545.index to /npu/debug00274026/MiniGo_ID0629_for_TensorFlow/test/..//outputs/models/000001-first_generation.index
Copying /npu/debug00274026/MiniGo_ID0629_for_TensorFlow/test/..//estimator_working_dir/model.ckpt-58545.meta to /npu/debug00274026/MiniGo_ID0629_for_TensorFlow/test/..//outputs/models/000001-first_generation.meta
Copying /npu/debug00274026/MiniGo_ID0629_for_TensorFlow/test/..//estimator_working_dir/model.ckpt-58545.data-00000-of-00001 to /npu/debug00274026/MiniGo_ID0629_for_TensorFlow/test/..//outputs/models/000001-first_generation.data-00000-of-00001
```

### 推理/验证过程<a name="section1465595372416"></a>

1.  通过 “模型训练” 中的指令启动测试。

2.  当前只能针对该工程训练出的checkpoint进行推理测试。

```
# 详细过程暂不提供 #
```