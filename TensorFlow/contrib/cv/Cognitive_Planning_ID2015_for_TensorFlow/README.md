-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [快速上手](#快速上手.md)
-   [迁移学习指导](#迁移学习指导.md)
-   [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）： Computer Version**

**版本（Version）：1.1**

**修改时间（Modified） ：2022.1.3**

**大小（Size）：74M**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Research**

**描述（Description）：基于TensorFlow框架的cognitive_planning语义目标驱动导航网络训练代码** 

<h2 id="概述.md">概述</h2>

本模型利用深度网络来学习基于语义表示的目标驱动路径规划导航策略，其主要特点是以图像为输入，利用预训练的Faster R-CNN提取目标语义信息，用LSTM学习当前状态下采取行动a的代价，最终根据最低代价生成一系列移动，指导机器人从起始位置移动到目标位置。

- 参考论文：

    [Mousavian, Arsalan, Alexander Toshev, Marek Fišer, Jana Košecká, Ayzaan Wahid, and James Davidson. "Visual representations for semantic target driven navigation." In 2019 International Conference on Robotics and Automation (ICRA), pp. 8846-8852. IEEE, 2019.](https://arxiv.org/abs/1805.06066) 

- 参考实现：https://github.com/tensorflow/models/tree/master/research/cognitive_planning
- 适配昇腾 AI 处理器的实现：
  https://gitee.com/juyierchun/modelzoo/edit/master/contrib/TensorFlow/Research/cv/Cognitive_Planning_ID2015_for_TensorFlow
  
  
## 默认配置<a name="section91661242121611"></a>

- 网络结构
  - CNN+LSTM
  
- 训练超参

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

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 是    |
| 混合精度  | 是    |
| 并行数据  | 否    |


## 训练环境准备
1. NPU环境  
硬件环境：
    ```
    NPU: 1*Ascend 910   
    CPU: 24*vCPUs 96GB  
    ```
    运行环境： 
    ```
    ascend-share/5.0.4.alpha005_tensorflow-ascend910-cp37-euleros2.8-aarch64-training:1.15.0-21.0.2_1230
    ```
2. 第三方依赖 
    ```
    networkx
    gin-config
    ```

# 快速上手
## 数据集准备
- 用户需自行下载ActiveVision Dataset数据集，已上传至obs中。obs路径如下：obs://cognitive-planning/dataset/AVD_Minimal/。
## 模型训练
* 单击“立即下载”，并选择合适的下载方式下载源码包。
* 开始训练
   *  启动训练之前，首先要配置程序运行相关环境变量。  环境变量配置信息参见：  
[Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)
 1. 配置训练参数。

     首先在脚本train_full_1p.sh中，配置参数以及训练数据集路径，请用户根据实际路径配置，如下所示：

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


## 验证

1. 测试的时候，需要修改脚本启动参数（Cognitive_Planning_ID2015_for_TensorFlow/test/train_full_1p.sh），配置mode为eval。

    ```
     --mode='eval' 
    ```

  2. 测试指令（脚本位于Cognitive_Planning_ID2015_for_TensorFlow/test/train_full_1p.sh）

      ```
      bash train_full_1p.sh
      ```

# 训练过程及结果
1. 执行train_supervised_active_vision.py文件。
2. 将训练得到的checkpoint文件放入checkpoint文件夹。
3. 在GPU复现中，由于自行编写的脚本与原论文中不同，评估结果也有一定差异。
以下为79000次迭代后的loss下降结果：

 |         | loss   |
 | --------   | :-----:  |
 | GPU复现    | 0.0553 |
 | NPU复现    | 0.0609 |





<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码<a name="section08421615141513"></a>

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

