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

**修改时间（Modified） ：2022.8.16**

**大小（Size）：2964676**

**框架（Framework）：TensorFlow_1.15**

**模型格式（Model Format）：ckpt**

**精度（Precision）：FP32**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的mixmatch方法的实现训练代码**

<h2 id="概述.md">概述</h2>

## 简述<a name="section194554031510"></a>

半监督学习已被证明是利用未标记数据来减轻对大型标记数据集的依赖的强大范例。在这项工作中，我们统一了当前用于半监督学习的主要方法，以产生一种新算法 MixMatch，该算法通过猜测数据增强未标记示例的低熵标签并使用 MixUp 混合标记和未标记数据来工作。我们表明，MixMatch 在许多数据集和标记数据量上都获得了最先进的结果。例如，在具有 250 个标签的 CIFAR-10 上，我们将错误率降低了 4 倍（从 38% 到 11%），在 STL-10 上降低了 2 倍。我们还展示了 MixMatch 如何帮助实现差异隐私的显着更好的准确性

- 参考论文：
  
  [https://arxiv.org/abs/1905.02249][MixMatch - A Holistic Approach to Semi-Supervised Learning.]

- 参考实现：

  https://github.com/google-research/mixmatch

- 适配昇腾 AI 处理器的实现：
  
  https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/mixmatch_ID0804_for_TensorFlow

- 通过Git获取对应commit\_id的代码方法如下：
  
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    

## 默认配置<a name="section91661242121611"></a>

-   训练超参（单卡）：
    - Dataset: cifar10.1@250-5000
    - arch: resnet
    - batch: 64                         
    - beta: 0.5
    - ema: 0.999
    - filters: 32
    - lr: 0.002
    - nclass: 10
    - repeat: 4
    - scales: 3
    - w_match: 100.0
    - wd: 0.02


## 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 是       |
| 混合精度   | 否      |
| 数据并行   | 是      |


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

1、用户需自行下载https://github.com/google-research/mixmatch#install-datasets数据集。

2、mixmatch训练的模型及数据集可以参考"简述 -> 参考实现"


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
        flags.DEFINE_float('wd', 0.02, 'Weight decay.')
        flags.DEFINE_float('ema', 0.999, 'Exponential moving average of params.')
        flags.DEFINE_float('beta', 0.5, 'Mixup beta distribution.')
        flags.DEFINE_float('w_match', 100, 'Weight for distribution matching loss.')
        flags.DEFINE_integer('scales', 0, 'Number of 2x2 downscalings in the classifier.')
        flags.DEFINE_integer('filters', 32, 'Filter size of convolutions.')
        flags.DEFINE_integer('repeat', 4, 'Number of residual layers per stage.')
        FLAGS.set_default('dataset', 'cifar10.3@250-5000')
        FLAGS.set_default('batch', 64)
        FLAGS.set_default('lr', 0.002)
        FLAGS.set_default('train_kimg', 1 << 16)
        flags.DEFINE_bool('use_fp16', True, '')
        flags.DEFINE_integer('num_gpus', 1, 'gpu number')
        #flags.DEFINE_string('obs_dir', 'home/ma-user/modelarts/outputs/train_url_0/', 'obs path')
        flags.DEFINE_integer('loss_scale', 999, 'Filter size of convolutions.')
        ```
        
        2. 修改参数/libml/data.py,/libml/train.py。
 
        ```
        DATA_DIR = "/cache/dataset"    //数据集路径
        train_dir', '/cache/result     //模型保存路径
        ```

        3. 脚本为mixmatch_ID0804_for_TensorFlow/test/train_full_1p.sh
    
         ```
         bash train_full_1p.sh
         ```
        4. 精度训练结果
     
        ```
           P（Precision）=（预测为真且正确预测的样本数）/（所有预测为真的样本数）
            
            - npu测试最优精度
            
            --filters=32 --dataset=cifar10.3@250-5000 --w_match=100 --beta=0.5 accuracy train/valid/test  100.00  87.92  87.35
            
            - gpu测试最优精度
            
            --filters=32 --dataset=cifar10.1@250-5000 --w_match=100 --beta=0.5 accuracy train/valid/test  100.00  88.52  87.90
            
            （其中accuracy train/valid/test 分别指训练、验证、测试的精度）
            
            - 论文中的精度要求
              cifar10 250models error rate of 11.08±0.87%   
         ```    


<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码

```
├── mixmatch.py         //网络训练与测试代码
├── README.md                                 //代码说明文档
├── libml                                     //初始化数据集，建立环境
│    ├──data.py        
│    ├──train.py                
│    ├──layers.py
│    ├──models.py
|    ├──data_pair.py
├── scripts                             //网络预处理
|    |── create_datasets.py
|    |── create_split.py
|    |── extract_accuracy.py 
|    |── inspect_dataset.py                    
├── test                          
│    ├──train_full_1p.sh                //训练验证full脚本
│    ├──train_performance_1p.sh              //训练验证perf性能脚本
├──requirements.txt
├──ops_info.json
├──mixup.py
├──pi_model.py
├──pseudo_label.py
├──vat.py
```

## 脚本参数<a name="section6669162441511"></a>

```
-- Dataset: cifar10.1@250-5000
-- arch: resnet
-- batch: 64                         
-- beta: 0.5
-- ema: 0.999
-- filters: 32
-- lr: 0.002
-- nclass: 10
-- repeat: 4
-- scales: 3
-- w_match: 100.0
-- wd: 0.02
```

## 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡。模型存储路径为${cur_path}/output/$ASCEND_DEVICE_ID，包括训练的log以及checkpoints文件。以1卡训练为例，loss信息在文件${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。