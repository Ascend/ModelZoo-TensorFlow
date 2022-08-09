- [基本信息](#基本信息.md)
- [概述](#概述.md)
- [训练环境准备](#训练环境准备.md)
- [快速上手](#快速上手.md)
- [迁移学习指导](#迁移学习指导.md)
- [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Object Detection**

**版本（Version）：1.1**

**修改时间（Modified） ：2022.8.8**

**大小（Size）：12KB**

**框架（Framework）：TensorFlow_1.15**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的BISIMULATION双模拟处理网络训练代码**

<h2 id="概述.md">概述</h2>

## 简述<a name="section194554031510"></a>

BISIMULATION用于计算确定性马尔可夫决策过程中的状态相似性的可扩展方法,将加载经过训练的双模拟度量近似值，开始评估运行，并报告从指定起始帧到剧集中每个其他帧的双模拟距离，并在此过程中生成一组 .png 和 .pdf 文件。它还将尝试生成编译所有 .png 文件的视频。

- 参考论文：
  
  [http://arxiv.org/abs/1911.09291](Scalable methods for computing state similarity in deterministic Markov Decision Processes)

- 参考实现：

  https://github.com/google-research/google-research/tree/master/bisimulation_aaai2020

- 适配昇腾 AI 处理器的实现：
  
  https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/BISIMULATION_ID2029_for_Tensorflow

- 通过Git获取对应commit\_id的代码方法如下：
  
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    

## 默认配置<a name="section91661242121611"></a>

-   训练超参（单卡）：
    - Batch size: 64
    - num_iterations：10000
    - data_url: ./dataset
    - train_url: ./output
    - base_dir: ./base_dir
    - grid_file: ./grid_file
    - gin_files: ./gin_files
    - starting_learning_rate：0.01
    - learning_rate_decay: 0.96


## 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 是       |
| 混合精度   | 是       |
| 数据并行   | 是       |


## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

拉起脚本中，

```
 ./train_full_1p.sh --help

parameter explain:
    --data_path              # dataset of training
    --output_path            # output of training
    --train_steps            # max_step for training
    --train_epochs           # max_epoch for training
    --batch_size             # batch size
    -h/--help                show help message
```

混合精度相关脚本grid_world.py代码示例:

 ```
    config_proto = tf.ConfigProto()
    custom_op = config_proto.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = 'NpuOptimizer'
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
    custom_op.parameter_map["customize_dtypes"].s = tf.compat.as_bytes("./matmul_setting.cfg")
    config = npu_config_proto(config_proto=config_proto)

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

1、数据集路径configs/mirrored_rooms.grid 、configs/mirrored_rooms.gin

2、BISIMULATION训练的模型及数据集可以参考"简述 -> 参考实现"


## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练。

    - 启动训练之前，首先要配置程序运行相关环境变量。

      环境变量配置信息参见：

      [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

    - 单卡训练

        1.首先在脚本test/train_full_1p.sh中，配置训练数据集路径，请用户根据实际路径配置，数据集参数如下所示：

             ```

             --base_dir=/tmp/grid_world --grid_file=configs/mirrored_rooms.grid --gin_files=configs/mirrored_rooms.gin

             ```

        2.启动训练
        
             启动单卡训练 （脚本为compute_metric.py） 
        
             ```
             python3 compute_metric.py

             ```
           


<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码

```
|--LICENSE
|--README.md                                                      #说明文档									
|--compute_metric.py                                              #训练代码
|--config.cfg
|--global_var.py
|--grid_world.py 									
|--requirements.txt                                               #所需依赖
|--matmul_setting.cfg
|--modelarts_entry_acc.py
|--modelarts_entry_perf.py		   						
|--configs                                                        #训练需要的数据集
|       |--mirrored_rooms.gin
|       |--mirrored_rooms.grid
|       |--mirrored_rooms_policy.grid
|       |--mirrored_rooms_profiling.gin
|--test			           	                          #训练脚本目录
|	|--train_full_1p.sh
|	|--train_performance_1p.sh
```

## 脚本参数<a name="section6669162441511"></a>

```
--data_url					
--train_url					
--base_dir				       
--batch_size					
--grid_file					
--gin_files
--base_dir,
--wall_length=2,
--grid_file=None
--gamma=0.99
--representation_dimension=64
--batch_size=64
--target_update_period=100
--num_iterations=10000
--starting_learning_rate=0.01
--use_decayed_learning_rate=False
--learning_rate_decay=0.96
--epsilon=1e-8
--staircase=False
--add_noise=True
--bisim_horizon_discount=0.9
--double_period_halfway=True
--total_final_samples=1000
--debug=False
```

## 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡，8卡网络训练。模型存储路径为${cur_path}/output/$ASCEND_DEVICE_ID，包括训练的log以及checkpoints文件。以8卡训练为例，loss信息在文件${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。