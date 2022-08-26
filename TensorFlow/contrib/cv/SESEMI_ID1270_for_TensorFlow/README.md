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

**修改时间（Modified） ：2022.8.26**

**大小（Size）：25.3MB**

**框架（Framework）：TensorFlow_1.15.0**

**模型格式（Model Format）：h5**

**精度（Precision）：FP32**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的MASF图像分类网络训练代码**

<h2 id="概述.md">概述</h2>

## 简述<a name="section194554031510"></a>

SESEMI的工作属于半监督学习(SSL)的框架，在图像分类的背景下，它可以利用大量的未标记数据，在有限的标记数据中显著改进监督分类器的性能。具体来说，我们利用自监督损失项作为正则化(应用于标记数据)和SSL方法(应用于未标记数据)类似于一致性正则化。尽管基于一致性正则化的方法获得了最先进的SSL结果，但这些方法需要仔细调优许多超参数，在实践中通常不容易实现。为了追求简单和实用，我们的模型具有自监督正则化，不需要额外的超参数来调整最佳性能

- 参考论文：

  [http://arxiv.org/abs/1809.05231](Exploring Self-Supervised Regularization for Supervised and Semi-Supervised Learning)

- 参考实现：

  https://github.com/vuptran/sesemi

- 适配昇腾 AI 处理器的实现：
  
  https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/SESEMI_ID1270_for_TensorFlow

- 通过Git获取对应commit\_id的代码方法如下：
  
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    

## 默认配置<a name="section91661242121611"></a>

-   训练超参（单卡）：

    - batch_size=16
    - dataset
    - epochs
    - labels
    - result
    - base_lr = 0.05
    - lr_decay_power = 0.5
    

## 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 否       |
| 混合精度   | 否       |
| 数据并行   | 否       |


## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

拉起脚本中，传入--precision_mode='allow_mix_precision'

```
 ./train_full_1p.sh --help

parameter explain:
    --precision_mode         precision mode(allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision)
    --over_dump                  if or not over detection, default is False
    --data_dump_flag         data dump flag, default is False
    --data_dump_step             data dump step, default is 10
    --profiling                  if or not profiling for performance debug, default is False
    --data_path                  source data of training
    -h/--help                    show help message
```

SESEMI模型未开启混合精度

<h2 id="训练环境准备.md">训练环境准备</h2>

-  硬件环境和运行环境准备请参见《[CANN软件安装指南](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=installation-update)》
-  运行以下命令安装依赖。
```
pip3 install requirements.txt
```
说明：依赖配置文件requirements.txt文件位于模型的根目录

<h2 id="快速上手.md">快速上手</h2>

## 数据集准备<a name="section361114841316"></a>

1、模型训练使用cifar-10数据集，数据集请用户自行获取(方法见https://github.com/vuptran/sesemi/tree/master/datasets)

2、数据集放入模型目录下，在训练脚本中指定数据集路径，可正常使用

3、SESEMI训练的模型及数据集可以参考"简述 -> 参考实现"


## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练。

    - 启动训练之前，首先要配置程序运行相关环境变量。

      环境变量配置信息参见：

      [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

    - 单卡训练

          1. 配置训练参数
        
             首先在脚本train_performance_1p.sh中，配置训练数据集路径和checkpoint保存路径，请用户根据实际路径配置，示例如下所示：
        
             ```
             # 路径参数初始化
             --dataset=${data_path}/cifar-10-batches-py 
             --epochs=${train_epochs} 
             --labels=2000 
             ```
        
          2. 启动训练（脚本为./test/train_performance_1p.sh） 
        
             ```
             bash train_performance_1p.sh --data_path
             ```

          3. 训练精度结果

            |样本数量|  论文精度 |GPU精度 | NPU精度 |
            |---------|-------- -- |----------|-----------|
            | 1000    |29.44±0.24|   0.2876 | 0.2983     |
            | 2000    |21.53±0.18|   0.2186 | 0.2179     |
           
          4. 训练性能结果

            | 样本数量 | GPU性能   | NPU性能   |
            | -------- | --------- | --------- |
            | 1000     | 75ms/step | 53ms/step |


<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码

```
├── LICENSE
├── README.md
├── modelzoo_level.txt
├── requirements.txt
├── train_evaluate_asl2.py                    //训练py脚本主入口
├── utils.py
├── networks                                  	
│    ├── convnet.py
│    ├── nin.py
│    ├── wrn.py
├── boot_modelarts.py                                 	
├── help_modelarts.py                               
├── dataset				        
│    ├── cifar10.py           		
│    ├── cifar100.py                       	
│    ├── dataset.txt                   		
│    ├── svhn.py                   	
├── test     
│    ├──train_performance_1p.sh                //训练性能入口
│    ├──train_full_1p.sh                       //训练精度入口，包含准确率评估
```

## 脚本参数<a name="section6669162441511"></a>

```
--batch_size=16
--dataset
--epochs
--labels
--result
--base_lr = 0.05
--lr_decay_power = 0.5
```

## 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡训练。模型存储路径为${cur_path}/output/$ASCEND_DEVICE_ID，包括训练的log以及checkpoints文件。以单卡训练为例，loss信息在文件${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。