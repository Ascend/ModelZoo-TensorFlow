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

**修改时间（Modified） ：2022.8.10**

**大小（Size）：359616KB**

**框架（Framework）：TensorFlow_1.15**

**模型格式（Model Format）：ckpt**

**精度（Precision）：FP32**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架Glow处理网络训练代码**

<h2 id="概述.md">概述</h2>

## 简述<a name="section194554031510"></a>

基于流的生成模型（Dinh等人，2014）在概念上具有吸引力，因为精确对数似然的可操作性，精确潜在变量推理的可操作性，以及训练和综合的并行性。在本文中，我们提出了发光，使用可逆1×1卷积的简单生成流。我们证明了对数似然性在标准上的显著改善，也许最引人注目的是，我们证明了一个生成模型针对普通对数似然目标进行优化，能够有效地进行逼真的合成和操作大图像。

- 参考论文：
  
  [https://d4mucfpksywv.cloudfront.net/research-covers/glow/paper/glow.pdf](Glow: Generative Flow with Invertible 1x1 Convolutions)

- 参考实现：

  https://github.com/openai/glow

- 适配昇腾 AI 处理器的实现：
  
  https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/Glow_ID2085_for_TensorFlow

- 通过Git获取对应commit\_id的代码方法如下：
  
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    

## 默认配置<a name="section91661242121611"></a>

-   训练超参（单卡）：
    - Batch size: 50
    - n_batches
    - data_url 
    - train_url
    - verbose：store_true
    - n_batch_train：50
    - n_train：50000
    - epochs：100


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

1、数据集链接ttps://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=dbviVeGq04cERsFrsux7Ctga7HCRO3Z1ucGAnMOBjZncpWkEX1jPl+MFK6jS5O4F7f+kvCP59S1vIcR9aTeBAnH0SPZT8F4Cw0uZlmkg+bHmsP0r2DCGqNejFCAp0A9rbW7ABl5nPV2JIcn+dSo2K0wQwc065KkccIUciDveKFpF/YL1wvS9xVDiH9xbJxTuFFVjffcbQ/jsZKuJmbGqMGmTZiWXlDAXVBM6Sj8PS5OXoJwahqp4VupvwRiJHlSLzZqEnqtRyOMz6R+/XOETclL3sgiafw+9ZcG5yeWp2US5TfJVYmL5cKEUkdczNb706C2AILI+cUtU5FwJEiHhG/dIpe3QCEAXQguyvbtAWc8BKSGeyqECFPQGXfaJbuCDIOkc/2IZHhdG8kKaO0YLu80r3fyxIexoxL826ozQ1v25llusdyS66dZ4Fhop6ss0T33VepElJFh7so+uQiYXEJ9y2FT5BWz7wTrFPkXTlZLSnRpvefxpP8J4F4QlZ7t7emllXLwb7LqcOedPjaoCUIcexjnz/2zsxy1w3kgVSFw= 提取码:123456

2、数据集分俩部分MNIST和cifar-10

3、GLOW训练的模型及数据集可以参考"简述 -> 参考实现"


## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练。

    - 启动训练之前，首先要配置程序运行相关环境变量。

      环境变量配置信息参见：

      [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)

    - 单卡训练


        1.首先在脚本test/train_full_1p.sh中, 训练需要根据安装教程，配置输入与输出的路径。配置训练数据集路径，请用户根据实际路径配置，数据集参数如下所示：

             ```

             python3 train.py --problem cifar10 --image_size 32 --n_level 3 --depth 32 --flow_permutation 2 --flow_coupling 1 --seed 0 --learntop --lr 0.001 --n_bits_x 8 --data_url=${data_path}/datasets --train_url=${output_path} --epochs=100

             ```

        2.启动训练
        
             启动单卡训练 （脚本为train.py） 
        
             ```
             python3 train.py

             ```
           


<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码

```
|--LICENSE
|--README.md                                                      #说明文档									
|--glow.json                                             
|--graphics.py
|--memory_saving_gradients.py
|--model.py									
|--requirements.txt                                               #所需依赖
|--modelzoo_level.txt
|--optim.py
|--tfops.py                                                       
|--train.py                                                       #训练代码
|--modelzoo_level.txt
|--train.py		   						
|--utils.py                                                     
|--test			           	                          #训练脚本目录
|	|--train_full_1p.sh
|	|--train_performance_1p.sh
```

## 脚本参数<a name="section6669162441511"></a>

```
--problem cifar10 
--image_size 32 
--n_level 3 
--depth 32 
--flow_permutation 2 
--seed 0 
--lr 0.001 
--epochs 300 
--n_bits_x 8
--data_url
--train_url
--n_train 50000
--n_batch_train 50
--epochs 100
```

## 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡，8卡网络训练。模型存储路径为${cur_path}/output/$ASCEND_DEVICE_ID，包括训练的log以及checkpoints文件。以1卡训练为例，loss信息在文件${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。