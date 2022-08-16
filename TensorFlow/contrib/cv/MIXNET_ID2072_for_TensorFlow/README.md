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

**描述（Description）：基于TensorFlow框架的MixNets深度卷积训练代码**

<h2 id="概述.md">概述</h2>

## 简述<a name="section194554031510"></a>

MixNets 是一系列配备了 MixConv 的移动尺寸图像分类模型，MixConv 是一种新型混合深度卷积。它们是基于AutoML MNAS Mobile 框架开发的，具有包括 MixConv 在内的扩展搜索空间。

- 参考论文：
  
  [https://arxiv.org/abs/1907.09595](MixConv: Mixed Depthwise Convolutional Kernels)

- 参考实现：

  https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet/mixnet.

- 适配昇腾 AI 处理器的实现：
  
  https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/ MIXNET_ID2072_for_TensorFlow

- 通过Git获取对应commit\_id的代码方法如下：
  
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    

## 默认配置<a name="section91661242121611"></a>

-   训练超参（单卡）：
    - data_dir
    - train_steps
    - mode=train 
    - log_step_count_steps
    - model_dir
    - model_name
    - npu_dump_data
    - npu_profiling
    - npu_dump_graph
    - npu_auto_tune


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

1、模型训练使用ILSVRC2012数据集，数据集请用户自行获取

2、MIXNET训练的模型及数据集可以参考"简述 -> 参考实现"


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
        --data_dir=./ILSVRC2012 
        --train_steps=5
        --mode=train 
        --log_step_count_steps=1 
        --model_dir=${cur_path}/result 
        --model_name='mixnet-s' 
        --npu_dump_data=True 
        --npu_profiling=True 
        --npu_dump_graph=False 
        --npu_auto_tune=False 
        ```
        
        2. 启动训练。
        
        ```
        python3.7 mnasnet_main.py
        ```
        


<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码

```
├── mnasnet_main.py                             //网络训练与测试代码
├── README.md                                 //代码说明文档
├── preprocessing.py                                 
├── imagenet_input.py                         
├── lars_optimizer.py                        
├── boot_modelarts.py  
├── imagenet_input.py   
├── mnasnet_model.py
├── mnasnet_models.py
├── modelzoo_level.txt
├── post_quantization.py                        
├── test                          
│    ├──train_full_1p.sh                //训练验证full脚本
│    ├──train_performance_1p.sh              //训练验证perf性能脚本
├──requirements.txt
```

## 脚本参数<a name="section6669162441511"></a>

```
-- data_dir
-- train_steps
-- mode=train 
-- log_step_count_steps
-- model_dir
-- model_name
-- npu_dump_data
-- npu_profiling
-- npu_dump_graph
-- npu_auto_tune
```

## 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡。模型存储路径为${cur_path}/output/$ASCEND_DEVICE_ID，包括训练的log以及checkpoints文件。以1卡训练为例，loss信息在文件${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。