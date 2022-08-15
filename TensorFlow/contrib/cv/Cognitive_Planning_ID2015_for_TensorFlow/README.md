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

**修改时间（Modified） ：2022.8.15**

**大小（Size）：79464232KB**

**框架（Framework）：TensorFlow_1.15**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的cognitive_planning语义目标驱动导航网络训练代码**

<h2 id="概述.md">概述</h2>

## 简述<a name="section194554031510"></a>

本模型利用深度网络来学习基于语义表示的目标驱动路径规划导航策略，其主要特点是以图像为输入，利用预训练的Faster R-CNN提取目标语义信息，用LSTM学习当前状态下采取行动a的代价，最终根据最低代价生成一系列移动，指导机器人从起始位置移动到目标位置。

- 参考论文：
  
  [https://arxiv.org/abs/1805.06066][Mousavian, Arsalan, Alexander Toshev, Marek Fišer, Jana Košecká, Ayzaan Wahid, and James Davidson. "Visual representations for semantic target driven navigation." In 2019 International Conference on Robotics and Automation (ICRA), pp. 8846-8852. IEEE, 2019.]

- 参考实现：

  https://github.com/tensorflow/models/tree/master/research/cognitive_planning

- 适配昇腾 AI 处理器的实现：
  
  https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/ Cognitive_Planning_ID2015_for_TensorFlow

- 通过Git获取对应commit\_id的代码方法如下：
  
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    

## 默认配置<a name="section91661242121611"></a>

-   训练超参（单卡）：
  - Batch size: 8
  - Momentum: 0.9
  - LR scheduler: exponential, with decay rate of 0.98 at every 1000 steps.
  - Learning rate(LR): 0.0001
  - Optimizer: AlexOptimizer
  - train_iters: 69000
  - lstm_cell_size: 2048 
  - policy_fc_size: 2048
  - sequence_length: 20
  - max_eval_episode_length: 100


## 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 是       |
| 混合精度   | 是      |
| 数据并行   | 否      |


## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

拉起脚本中，

```
 ./train_full_1p.sh --help

parameter explain:
    --precision_mode         #precision mode(allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision)
    --data_path              # dataset of training
    --output_path            # output of training
    --train_steps            # max_step for training
    --train_epochs           # max_epoch for training
    --batch_size             # batch size
    -h/--help                show help message
```

混合精度相关代码示例:

 ```
    precision_mode="allow_mix_precision"

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

1、用户需自行下载ActiveVision Dataset数据集。

2、Cognitive_Planning训练的模型及数据集可以参考"简述 -> 参考实现"


## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练。

    - 启动训练之前，首先要配置程序运行相关环境变量。

      环境变量配置信息参见：

      [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

    - 单卡训练


        1. 配置训练参数。
        
        训练参数已经默认在脚本中设置，需要在启动训练时指定数据集路径和输出路径
        
        ```
          --mode='train'   
          --logdir=${output_path}/checkpoint  
          --modality_types='det'   
          --batch_size=8   
          --train_iters=200000    
          --lstm_cell_size=2048   
          --policy_fc_size=2048   
          --sequence_length=20   
          --max_eval_episode_length=100     
          --test_iters=194   
          --gin_config=envs/configs/active_vision_config.gin   
          --gin_params="ActiveVisionDatasetEnv.dataset_root='${data_path}'"   
          --logtostderr
        ```
        
      2. 启动训练。

         脚本为Cognitive_Planning_ID2015_for_TensorFlow/test/train_full_1p.sh
    
         ```
         bash train_full_1p.sh
         ```
        


<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码

```
├── train_supervised_active_vision.py         //网络训练与测试代码
├── README.md                                 //代码说明文档
├── envs                                      //初始化数据集，建立环境
│    ├──active_vision_dataset_env.py         
│    ├──task_env.py                 
│    ├──util.py
│    ├──configs
|    │    ├──active_vision_config.gin
├── preprocessing                             //网络预处理
├── embedders.py
├── modelzoo_level.txt
├── label_map_util.py                       
├── test                          
│    ├──train_full_1p.sh                //训练验证full脚本
│    ├──train_performance_1p.sh              //训练验证perf性能脚本
├──requirements.txt
├──modelarts_entry_acc.py
├──modelarts_entry_perf.py
├──standard_fields.py
├──string_int_label_map_pb2.py
├──tasks.py
├──train_supervised_active_vision.py
├──visualization_utils.py
├──viz_active_vision_dataset_main.py
```

## 脚本参数<a name="section6669162441511"></a>

```
--mode                   运行模式（train_and_evaluate）；可选：train，eval
--logdir                 ckpt文件存放路径
--modality_types=det
--batch_size=8
--train_iters=200000
--lstm_cell_size=2048
--policy_fc_size=2048
--sequence_length=20
--max_eval_episode_length=100
--test_iters=194
--gin_config=envs/configs/active_vision_config.gin
--gin_params="ActiveVisionDatasetEnv.dataset_root='C:/Users/10901/Desktop/cognitive_models/research/cognitive_planning/ActiveVisionDataset'"
--logtostderr
```

## 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡。模型存储路径为${cur_path}/output/$ASCEND_DEVICE_ID，包括训练的log以及checkpoints文件。以1卡训练为例，loss信息在文件${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。