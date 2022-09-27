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

**大小（Size）：150649408KB**

**框架（Framework）：TensorFlow_1.15**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的CondConv缩放网络训练代码**

<h2 id="概述.md">概述</h2>

## 简述<a name="section194554031510"></a>

抽象的： 卷积层是现代深度神经网络的基本组成部分之一。一个基本假设是，数据集中的所有示例都应该共享卷积核。我们提出了条件参数化卷积（CondConv），它为每个示例学习专门的卷积核。用 CondConv 替换普通卷积使我们能够增加网络的大小和容量，同时保持有效的推理。我们证明了使用 CondConv 缩放网络可以提高几种现有卷积神经网络架构在分类和检测任务上的性能和推理成本权衡。在 ImageNet 分类中，我们应用于 EfficientNet-B0 的 CondConv 方法实现了 78.3% 准确率的最先进性能，且仅 413M 乘加。

- 参考论文：
  
  [https://arxiv.org/abs/1904.04971](CondConv: Conditionally Parameterized Convolutions for Efficient Inference)

- 参考实现：

  https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/condconv

- 适配昇腾 AI 处理器的实现：
  
  https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/efficientnet-condconv_ID2074_for_TensorFlow

- 通过Git获取对应commit\_id的代码方法如下：
  
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    

## 默认配置<a name="section91661242121611"></a>

-   训练超参（单卡）：
    - data_dir
    - model_dir
    - model_name 
    - npu_dump_data 
    - npu_profiling 
    - npu_dump_graph
    - log_step_count_steps
    - npu_auto_tune 
    - batch_size
    - train_steps
    - num_train_images

## 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 是       |
| 混合精度   | 是      |
| 数据并行   | 是       |


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

1、模型训练使用ImageNet2012数据集，数据集请用户自行获取

2、efficientnet-condconv训练的模型及数据集可以参考"简述 -> 参考实现"


## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练。

    - 启动训练之前，首先要配置程序运行相关环境变量。

      环境变量配置信息参见：

      [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)

    - 单卡训练


        1. 配置训练参数。
        
        训练参数已经默认在脚本中设置，需要在启动训练时指定数据集路径和输出路径
        
        ```
        --data_dir=$data_path 
        --model_dir=./result 
        --model_name='efficientnet-condconv-b0-8e' 
        --batch_size=32 
        --obs_dir=${obs_url} 
        --npu_dump_data=True 
        --npu_profiling=True 
        --npu_dump_graph=False 
        --use_async_checkpointing=False 
        --mode=train 
	--log_step_count_steps=1 
	--train_steps=100 
        --npu_auto_tune=False  
        ```
        
        2. 启动训练。
        
        ```
        python3.7 main.py
        ```
        


<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码

```
├── main.py                              //网络训练与测试代码
├── README.md                                 //代码说明文档
├── preprocessing.py                                 
├── model_builder_factory.py                          
├── lars_optimizer.py                         
├── inspect_model_architecture.py    
├── imagenet_input.py   
├── export_model.py
├── autoaugment.py
├── efficientnet_model.py
├── efficientnet_builder.py                         
├── test                          
│    ├──train_full_1p.sh                //训练验证full脚本
│    ├──train_performance_1p.sh              //训练验证perf性能脚本
├──requirements.txt
```

## 脚本参数<a name="section6669162441511"></a>

```
--data_dir
--model_dir
--model_name 
--batch_size
--obs_dir 
--npu_dump_data
--npu_profiling 
--npu_dump_graph
--use_async_checkpointing
--mode 
--log_step_count_steps 
--train_steps
--npu_auto_tune 
```

## 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡。模型存储路径为${cur_path}/output/$ASCEND_DEVICE_ID，包括训练的log以及checkpoints文件。以1卡训练为例，loss信息在文件${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。