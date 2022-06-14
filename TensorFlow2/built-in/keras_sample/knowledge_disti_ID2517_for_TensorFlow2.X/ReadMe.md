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

**修改时间（Modified） ：2022.06.14**

**大小（Size）：11MB**

**框架（Framework）：TensorFlow_2.6.2**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）知识蒸馏是一种模型压缩过程，其中训练一个小型（学生）模型以匹配一个大型预训练（教师）模型。通过最小化损失函数，知识从教师模型转移到学生，旨在匹配真实标签。**

<h2 id="概述.md">概述</h2>

## 简述<a name="section194554031510"></a>


知识蒸馏是一种模型压缩过程，其中训练一个小型（学生）模型以匹配一个大型预训练（教师）模型。通过最小化损失函数，知识从教师模型转移到学生，旨在匹配训练真实标签。通过在 softmax 中应用“温度”缩放函数来软化 logits，有效地平滑概率分布并揭示教师学习的类间关系

- 参考论文：

  https://arxiv.org/abs/1503.02531

- 参考实现：

  https://github.com/keras-team/keras-io/blob/master/examples/vision/knowledge_distillation.py

- 适配昇腾 AI 处理器的实现：
    
  https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow2/built-in/keras_sample/cv/knowledge_disti_ID2517_for_TensorFlow2.X

- 通过Git获取对应commit\_id的代码方法如下：
    
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    

## 默认配置<a name="section91661242121611"></a>

-   网络结构：
    -   任何 Keras 模型即可
    
-   训练超参（单卡）：
    -   Batch size: 32
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

1、用户自行准备好数据集，本网络使用的数据集是MNIST数据集

数据集目录参考如下：

```
├── MNIST
│    ├── mnist.npz
│    ├── t10k-images-idx3-ubyte.gz     
│    ├── t10k-labels-idx1-ubyte.gz
│    ├── train-images-idx3-ubyte.gz
│    ├── train-labels-idx1-ubyte.gz
```



## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练。

    1. 启动训练之前，首先要配置程序运行相关环境变量。
    
    	环境变量配置信息参见：
    
       [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)
       
    2. 单卡训练
       
        2. 1单卡训练指令（脚本位于knowledge_disti_ID2517_for_TensorFlow2.X/test/train_full.sh）,需要先使用cd命令进入test目录下，再使用下面的命令启动训练。请确保下面例子中的“--data_path”修改为用户的数据路径,这里选择将cats_vs_dogs文件夹放在home目录下。
        
            bash train_full_1p.sh --data_path=/home



<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码

```
|--LICENSE
|--README.md								#说明文档
|--knowledge_distillation.py   						#训练代码
|--requirements.txt		   					#所需依赖
|--test			           					#训练脚本目录
|   |--train_full_1p.sh							#全量训练脚本
|   |--train_performance_1p_dynamic_eval.sh		                #performance动态shape训练脚本
|   |--train_performance_1p_static_eval.sh		                #performance静态shape训练脚本
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


