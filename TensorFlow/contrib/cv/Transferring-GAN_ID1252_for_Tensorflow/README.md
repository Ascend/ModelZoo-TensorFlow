- [基本信息](#基本信息.md)
- [概述](#概述.md)
- [训练环境准备](#训练环境准备.md)
- [快速上手](#快速上手.md)
- [迁移学习指导](#迁移学习指导.md)
- [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Image Synthesis**

**版本（Version）：1.1**

**修改时间（Modified） ：2022.08.08**

**大小（Size）：405664KB**

**框架（Framework）：TensorFlow_1.15**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的Transferring-GAN图像生成网络训练代码**

<h2 id="概述.md">概述</h2>

## 简述<a name="section194554031510"></a>

通过微调的方式将预训练网络的知识转移到新的领域，是基于判别模型的应用中广泛使用的一种做法。Transferring-GAN研究将领域适应应用于生成式对抗网络的图像生成。我们评估了领域适应的几个方面，包括目标领域大小的影响，源和目标领域之间的相对距离，以及条件GAN的初始化。Transferring-GAN使用来自预训练网络的知识可以缩短收敛时间，并能显著提高生成图像的质量，特别是当目标数据有限时.

- 参考论文：

  [https://arxiv.org/abs/1704.00028](https://github.com/igul222/improved_wgan_training)

- 参考实现：

  https://github.com/yaxingwang/Transferring-GANs

- 适配昇腾 AI 处理器的实现：
  
  https://gitee.com/ascend/ModelZoo-TensorFlow/edit/master/TensorFlow/contrib/cv/Transferring-GAN_ID1252_for_Tensorflow

- 通过Git获取对应commit\_id的代码方法如下：
  
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    

## 默认配置<a name="section91661242121611"></a>

-   训练超参（单卡）：
  - Batch size: 64
  - Learning rate(LR): 2e-4
  - ITERS: 1000
  - DIM_G: 128
  - DIM_D: 128
  - NORMALIZATION_G: True
  - NORMALIZATION_D: False
  - Optimizer: Adam


## 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 是       |
| 混合精度   | 是       |
| 数据并行   | 是       |


## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

拉起脚本中，传入--precision_mode='allow_mix_precision'

```
 python3 gan_cifar_resnet_acc.py --help

parameter explain:
    --GEN_BS_MULTIPLE         
    --batch_size                  
    --ITERS         
    --LR             
    --output_path                 
    --data_path                  
    -h/--help                   
```

相关代码示例:

```
config = tf.ConfigProto()
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True  # 在昇腾AI处理器执行训练
custom_op.parameter_map["precision_mode"].s=tf.compat.as_bytes("allow_mix_precision")

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

1、用户自行准备好数据集lsun-bedroom( http://lsun.cs.princeton.edu/2017/ )

2、Transferring-GAN训练的模型及数据集可以参考"简述 -> 参考实现"


## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练。

    - 启动训练之前，首先要配置程序运行相关环境变量。

      环境变量配置信息参见：

      [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)

    - 单卡训练

        1.首先在脚本test/train_full_1p.sh中，配置训练数据集路径，请用户根据实际路径配置，数据集参数如下所示：

             ```
              --data_path ./dataset
             ```

        2.启动训练
        
             启动单卡训练 （脚本为modelarts_entry_acc.py） 
        
             ```
             python3 modelarts_entry_acc.py 
             ```
               

<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码

```
VisionTransformer
└─ 
  ├─README.md
  ├─tflib 依赖包
  ├─dataset用于存放训练数据集和预训练文件
    ├─*batch* cifar10 数据集
  	├─ckpt/ 预训练权重存放地址
  	└─inception/ 计算inception score时所需测试模型的存放地址
  ├─test 用于测试
      ├─output 用于存放测试结果和日志 
  	└─train_full_1p.sh
  ├─gan_cifar_resnet_acc.py 精度性能测试脚本
```

## 脚本参数<a name="section6669162441511"></a>

```
--LR          学习率,默认是2e-4
--batch_size             训练的batch大小，默认是64
--data_path              训练集文件路径
--GEN_BS_MULTIPLE            生成器一次性生成的样本数量，默认是2
--ITERS                训练的迭代次数，默认是1000
--DIM_G                生成器输入维度，默认是128
--DIM_D                判别器输入维度，默认是128
--NORMALIZATION_G      生成器是否使用BN，默认是True
--NORMALIZATION_D      判别器是否使用BN，默认是False
--OUTPUT_DIM           生成器输出维度，默认是3072
--DECAY                是否使用学习率衰减，默认是True
--INCEPTION_FREQUENCY  每多少次迭代计算一次inception score，默认是1
--CONDITIONAL          是否训练一个条件模型
--ACGAN                如果CONDITIONAL为True, 是否使用ACGAN或者vanilla
--ACGAN_SCALE          ACGAN loss的缩放因子，默认是1.0
--ACGAN_SCALE_G        生成器ACGAN loss的缩放因子，默认是1.0
```

## 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡，8卡网络训练。模型存储路径为${cur_path}/output/$ASCEND_DEVICE_ID，包括训练的log以及checkpoints文件。以8卡训练为例，loss信息在文件${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。