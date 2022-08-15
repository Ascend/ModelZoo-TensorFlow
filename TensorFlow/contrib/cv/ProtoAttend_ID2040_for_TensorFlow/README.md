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

**修改时间（Modified） ：2022.8.12**

**大小（Size）：30172KB**

**框架（Framework）：TensorFlow_1.15**

**模型格式（Model Format）：ckpt**

**精度（Precision）：FP32**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的注意力机制的原型学习训练代码**

<h2 id="概述.md">概述</h2>

## 简述<a name="section194554031510"></a>

ProtoAttend是一种可解释的机器学习方法，该方法基于原型的少数相关示例做出决策。ProtoAttend 可以集成到各种神经网络架构中，包括预训练模型。它利用一种注意力机制，将编码表示与样本相关联，以确定原型。在不牺牲原始模型准确性的情况下，生成的模型在三个高影响问题上优于现有技术：（1）它实现了高质量的可解释性，输出与决策最相关的样本（即基于样本的可解释性方法）； (2) 它通过量化原型标签之间的不匹配来实现最先进的置信度估计； (3) 它获得了分布不匹配检测的最新技术。所有这些都可以通过最少的额外测试时间和实际可行的训练时间计算成本来实现。

- 参考论文：
  
  https://arxiv.org/pdf/1902.06292.pdf

- 参考实现：

  https://github.com/google-research/google-research/tree/master/protoattend

- 适配昇腾 AI 处理器的实现：
  
  https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/ProtoAttend_ID2040_for_TensorFlow  

- 通过Git获取对应commit\_id的代码方法如下：
  
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    

## 默认配置<a name="section91661242121611"></a>

-   训练超参（单卡）：
  - Batch size: 128
  - Train step: 100000
  - init_learning_rate: 0.001


## 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 是       |
| 混合精度   | 否      |
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

1、模型训练使用Fashion-MNIST数据集，数据集请用户自行获取

2、ProtoAttend训练的模型及数据集可以参考"简述 -> 参考实现"


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
        flags.DEFINE_integer("random_seed", 1, "Random seed.")
        flags.DEFINE_integer("num_steps", 100000, "Number of training steps.")
        flags.DEFINE_integer("display_step", 500, "Display step.")
        flags.DEFINE_integer("val_step", 400, "Validation step.")
        flags.DEFINE_integer("save_step", 4000, "Save step.")
        flags.DEFINE_float("init_learning_rate", 0.001, "Initial learning rate.")
        flags.DEFINE_integer("decay_every", 2000, "Decay interval.")
        flags.DEFINE_float("decay_rate", 0.9, "Decay rate.")
        flags.DEFINE_integer("gradient_thresh", 20, "Gradient clipping threshold.")
        flags.DEFINE_integer("batch_size", 128, "Batch size.")
        flags.DEFINE_integer("example_cand_size", 1024)
        flags.DEFINE_integer("eval_cand_size", 1024)
        flags.DEFINE_string( "train_url", "../output","The output directory where the model checkpoints will be written.")
        flags.DEFINE_string("data_url", "../dataset",  "dataset path")
        flags.DEFINE_string("obs_dir", "obs://npuprotoattend/log", "obs result path, not need on gpu and apulis platform")
        flags.DEFINE_string("chip", "npu", "Run on which chip, (npu or gpu or cpu)")
        flags.DEFINE_string("platform", "modelarts", "Run on linux/apulis/modelarts platform. Modelarts Platform has some extra data copy operations")
        flags.DEFINE_string("result", "/cache/result", "The result directory where the model checkpoints will be written.")
        flags.DEFINE_boolean("profiling", False, "profiling for performance or not")
        
        ```
        
        2. 启动训练。
        
        ```
        python3.7 main_protoattend.py
        ```
        


<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码

```
├── dataset 			// 数据集
├── test
│   ├── train_full_1p.sh        //单卡全量训练启动脚本
│   ├── train_performance_1p.sh //单卡训练验证性能启动脚本
├── input_data.py 	        //处理数据
├── load_data.py 	        //读取数据
├── main_protoattend.py 	//训练模型
├── model.py 	                //模型定义
├── options.py 	                //参数配置
├── utils.py 	                //工具类
├── requirements.txt 		//训练python依赖列表
└── README.md 			// 代码说明文档
```

## 脚本参数<a name="section6669162441511"></a>

```
--data_url              
--train_url             
--batch_size", 128
--decay_rate", 0.9
--num_steps", 100000
--display_step", 500
--save_step", 4000
--init_learning_rate", 0.001
--eval_cand_size", 1024
--img_size", 28
--num_classes", 10
--example_cand_size", 1024
```

## 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡。模型存储路径为${cur_path}/output/$ASCEND_DEVICE_ID，包括训练的log以及checkpoints文件。以1卡训练为例，loss信息在文件${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。