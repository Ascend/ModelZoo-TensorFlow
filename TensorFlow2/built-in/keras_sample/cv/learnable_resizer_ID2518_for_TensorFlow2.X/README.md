- [基本信息](#基本信息.md)
- [概述](#概述.md)
- [训练环境准备](#训练环境准备.md)
- [快速上手](#快速上手.md)
- [迁移学习指导](#迁移学习指导.md)
- [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Image Classification**

**版本（Version）：1.1**

**修改时间（Modified） ：2022.4.8**

**大小（Size）：51KB**

**框架（Framework）：TensorFlow_2.6.2**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：对于给定的图像分辨率和模型，如何最好地调整给定图像的大小，基于TensorFlow框架的训练代码**

<h2 id="概述.md">概述</h2>

## 简述<a name="section194554031510"></a>


人们普遍认为，如果我们将视觉模型限制为像人类一样感知事物，他们的表现可以提高。人类大多使用形状描述符来开发共同的认知。但是这种信念是否始终适用，尤其是在改进方面视觉模型的表现？
事实证明，情况可能并非总是如此。在训练视觉模型时，通常将图像大小调整为较低尺寸（（224 x 224）、（299 x 299）等）以允许小批量学习并保持计算限制。我们一般使用图像此步骤的调整大小方法，如 双线性插值和调整大小的图像不会在人眼中失去太多的感性特征。在 [Learning to Resize Images for Computer Vision Tasks ]， Talebi et al. 中显示如果我们尝试优化视觉模型的图像感知质量而不是人眼，它们的性能可以进一步提高。他们调查以下问题：
**对于给定的图像分辨率和模型，如何最好地调整给定图像的大小？**
如论文所示，这个想法有助于持续提高常见的视觉模型（在 ImageNet-1k 上预训练），如 DenseNet-121、ResNet-50、MobileNetV2 和 EfficientNets。在这个例子中，我们将实现可学习的图像按照论文中的建议调整模块大小，并使用Cats and Dogs数据集和DenseNet-121结构。

- 参考论文：

  https://arxiv.org/abs/2103.09950v1 

- 参考实现：

  https://github.com/keras-team/keras-io/blob/master/examples/vision/learnable_resizer.py

- 适配昇腾 AI 处理器的实现：
    
  https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow2/built-in/keras_sample/cv/learnable_resizer_ID2518_for_TensorFlow2.X

- 通过Git获取对应commit\_id的代码方法如下：
    
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    

## 默认配置<a name="section91661242121611"></a>

-   网络结构：
    -   DenseNet-121
    
-   训练超参（单卡）：
    -   Batch size: 64
    -   Input size: (300, 300)
    -   Target size: (150, 150)
    -   Interpolation: "bilinear"
    -   Train epoch: 5


## 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 否       |
| 混合精度   | 是       |
| 数据并行   | 否       |


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
flags.DEFINE_string(name='precision_mode', default= 'allow_fp32_to_fp16',
                    help='allow_fp32_to_fp16/force_fp16/ ' 
                    'must_keep_origin_dtype/allow_mix_precision.')

npu_device.global_options().precision_mode=FLAGS.precision_mode
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

1、用户自行准备好数据集，本网络使用的数据集是Cats and Dogs数据集

数据集目录参考如下：

```
├── cats_vs_dogs
│    ├──4.0.0
│    │    ├──cats_vs_dogs-train.tfrecord-00000-of-00008   
│    │    │ ......
│    │    ├──cats_vs_dogs-train.tfrecord-00007-of-00008 
│    │    ├──dataset_info.json
│    │    ├──features.json
│    │    ├──label.labels.txt
```



## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练。

    1. 启动训练之前，首先要配置程序运行相关环境变量。
    
    	环境变量配置信息参见：
    
       [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)
       
    2. 单卡训练
       
        2. 1单卡训练指令（脚本位于learnable_resizer_ID2518_for_TensorFlow2.X/test/train_full.sh）,需要先使用cd命令进入test目录下，再使用下面的命令启动训练。请确保下面例子中的“--data_path”修改为用户的数据路径,这里选择将cats_vs_dogs文件夹放在home目录下。
        
            bash train_full_1p.sh --data_path=/home



<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码

```
|--LICENSE
|--README.md									#说明文档
|--learnable_resizer.py    						#训练代码
|--requirements.txt		   						#所需依赖
|--test			           						#训练脚本目录
|	|--train_full_1p.sh							#全量训练脚本
|	|--train_performance_1p_dynamic_eval.sh		#performance动态shape训练脚本
|   |--train_performance_1p_static_eval.sh		#performance静态shape训练脚本
```

## 脚本参数<a name="section6669162441511"></a>

```
--data_path						# the path to train data
--epoch							# epochs of training
--static						# Whether it is a static shape
--log_steps						# TimeHis log Step
--precision_mode				# the path to save over dump data
--over_dump						# if or not over detection, default is False
--data_dump_flag				# data dump flag, default is False
--data_dump_step				# data dump step, default is 10
--profiling						# if or not profiling for performance debug, default is False
--profiling_dump_path			# the path to save profiling data
--over_dump_path				# the path to save over dump data
--data_dump_path				# the path to save dump data
--use_mixlist					# use_mixlist flag, default is False
--fusion_off_flag				# fusion_off flag, default is False
--mixlist_file					# mixlist file name, default is ops_info.json
--fusion_off_file				# fusion_off file name, default is fusion_switch.cfg
```

## 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡，8卡网络训练。模型存储路径为${cur_path}/output/$ASCEND_DEVICE_ID，包括训练的log以及checkpoints文件。以8卡训练为例，loss信息在文件${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。