- [基本信息](#基本信息.md)
- [概述](#概述.md)
- [训练环境准备](#训练环境准备.md)
- [快速上手](#快速上手.md)
- [迁移学习指导](#迁移学习指导.md)
- [高级参考](#高级参考.md)
## 基本信息

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Recommendation**

**版本（Version）：1.1**

**修改时间（Modified） ：2022.6.11**

**大小（Size）：44KB**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的推荐网络训练代码**

## 概述


DeepCTR 是一个**易于使用**、**模块化**和**可扩展**的基于深度学习的 CTR 模型包以及许多可用于轻松构建自定义模型的核心组件层，在该网络中我们定义了FwFM,MMoE,DeepFM,FLEN,DCNMix五个模型。

- 参考论文：

  https://arxiv.org/pdf/1806.03514.pdf

  https://dl.acm.org/doi/abs/10.1145/3219819.3220007

  https://www.ijcai.org/proceedings/2017/0239.pdf

  https://arxiv.org/pdf/1911.04690.pdf

  https://arxiv.org/pdf/2008.13535

- 参考实现：

  https://github.com/shenweichen/deepctr

- 适配昇腾 AI 处理器的实现：
  
  https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/built-in/recommendation/DeepCTR_Series_for_TensorFlow

- 通过Git获取对应commit\_id的代码方法如下：
  
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    

#### 默认配置<a name="section91661242121611"></a>

-   训练超参（单卡）：
    - Batch size: 128
    - epoch: 10


#### 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 否       |
| 混合精度   | 是       |
| 数据并行   | 是       |


#### 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

#### 开启混合精度<a name="section20779114113713"></a>

拉起脚本中，传入--precision_mode='allow_mix_precision'

```
 ./train_ID3057_FwFM_performance_1p.sh --help

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
parser.add_argument('--precision_mode', default='allow_fp32_to_fp16',
                     help='allow_fp32_to_fp16/force_fp16/ '
                     'must_keep_origin_dtype/allow_mix_precision.')

custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes(args.precision_mode)
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

1、FwFM、MoME、FLEN、DeepFM模型的数据集为examples目录下的criteo_sample.txt

2、DCNMix模型的数据集是Kaggle-Criteo数据集，需要使用gen_kaggle_criteo_tfrecords.py转换为tfrecord

```
# 脚本中src_filename需要修改为用户实际的数据集路径
python3 gen_kaggle_criteo_tfrecords.py

```


#### 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练。

    1. 启动训练之前，首先要配置程序运行相关环境变量。

    	环境变量配置信息参见：

       [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)
       
    2. 单卡训练

        2.1 FwFM单卡任务训练指令

        ```
        bash train_ID3057_FwFM_performance_1p.sh --data_path=../examples/criteo_sample.txt
        ```

        2.2 MMoE单卡任务训练指令

        ```
        bash train_ID3058_MMoE_performance_1p.sh --data_path=../examples/criteo_sample.txt
        ```

        2.3 DeepFM单卡任务训练指令
        
        ```
        bash train_ID3062_DeepFM_performance_1p.sh --data_path=../examples/criteo_sample.txt
        ```
        
        2.4 FLEN单卡任务训练指令
        
        ```
        bash train_ID3204_FLEN_performance_1p.sh --data_path=../examples/criteo_sample.txt
        ```
       
        2.5 DCNMix单卡任务训练指令
        
        ```
        bash train_ID4032_DCNMix_performance_1p.sh --data_path=/data/criteo.tfrecord
        ```
        
        


## 高级参考

#### 脚本和示例代码

```
|--LICENSE
|--README.md                                                    #说明文档									
|--requirements.txt                                             #所需依赖
|--test                                                         #训练脚本目录
|	|--train_ID3057_FwFM_full_1p.sh                         #全量训练脚本
|	|--train_ID3057_FwFM_performance_1p.sh		        #performance训练脚本
|--examples                                                     #训练模型目录
|	|--run_fwfm.py                                          #FwFM模型训练主入口
|	|--run_flen.py                                          #FLEN模型训练主入口
|	|--run_mtl.py                                           #MOME模型训练主入口
|	|--run_classification_criteo.py                         #DeepFM模型训练主入口
|	|--run_dcnmix.py                                        #DCNMix模型训练主入口
|	|--criteo_sample.txt                                    #criteo样例数据集
|	|--gen_kaggle_criteo_tfrecords.py                       #kaggle-criteo数据集转换为tfrecord脚本
```

#### 脚本参数<a name="section6669162441511"></a>

```
--data_dir
--precision_mode
--profiling
--profiling_dump_path
```

#### 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡，8卡网络训练。模型存储路径为${cur_path}/output/$ASCEND_DEVICE_ID，包括训练的log以及checkpoints文件。以8卡训练为例，loss信息在文件${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。