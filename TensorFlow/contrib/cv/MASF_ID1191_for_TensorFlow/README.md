# MASF_for_TensorFlow
## 目录


## 基本信息

发布者（Publisher）：Huawei

应用领域（Application Domain）： Image Classification

版本（Version）：1.2

修改时间（Modified） ：2021.11.18

大小（Size）：25.3MB

框架（Framework）：TensorFlow 1.15.0

模型格式（Model Format）：ckpt

精度（Precision）：Mixed

处理器（Processor）：昇腾910

应用级别（Categories）：Research

描述（Description）：基于TensorFlow框架的MASF图像分类网络训练代码


## 概述
MASF是一个有着泛化能力的图像分类网络，主要特点是采用了元学习的学习策略，以及利用三胞胎损失和KL散度重构了损失函数，使训练的网络并不会过拟合至训练域，而会提取类的泛化特征，从而在未知域实现分类任务

    参考论文：

    Dou Q, Coelho de Castro D, Kamnitsas K, et al. Domain generalization via model-agnostic learning of semantic features[J]. Advances in Neural Information Processing Systems, 2019, 32: 6450-6461.

    参考实现：

    适配昇腾 AI 处理器的实现：

    https://gitee.com/ascend/modelzoo/tree/master/contrib/TensorFlow/Research/cv/MASF_ID1191_for_TensorFlow

    通过Git获取对应commit_id的代码方法如下：
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换

    
### 默认配置<a name="section91661242121611"></a>
    -   训练超参
          -  meta Batch size: 126
          -  inner_lr       : 1e-05
          -  outer_lr       : 1e-05
          -  metric_lr      : 1e-05
          -  margin         : 10
          -  gradients_clip_value : 2.0

    -   训练数据集：
          -  数据集采用PACS数据集

    -   测试数据集：
          -  测试数据集与训练数据集相同,使用PACS数据集


### 支持特性<a name="section1899153513554"></a>
| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 否       |
| 混合精度   | 是       |
| 数据并行   | 是       |


### 混合精度训练<a name="section168064817164"></a>
    混合精度训练昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

### 开启混合精度<a name="section20779114113713"></a>
相关代码示例
```
    config_proto = tf.compat.v1.ConfigProto()
    custom_op = config_proto.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes(args.precision_mode)
    config_proto.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    sess = tf.Session(config=npu_config_proto(config_proto=config_proto))
```

## 训练环境准备
1.  硬件环境准备请参见各硬件产品文档"[驱动和固件安装升级指南]( https://support.huawei.com/enterprise/zh/category/ai-computing-platform-pid-1557196528909)"。需要在硬件设备上安装与CANN版本配套的固件与驱动。
2.  宿主机上需要安装Docker并登录[Ascend Hub中心](https://ascendhub.huawei.com/#/detail?name=ascend-tensorflow-arm)获取镜像。

### 数据集准备
1. 模型训练使用PACS数据集，请用户自行获取。
2. 模型训练使用ALEXNET预训练权重bvlc_alexnet.npy，请用户自行获取。

### 模型训练

- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练    
    - 1. 启动训练之前，首先要配置程序运行相关环境变量。
       环境变量配置信息参见：
          [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)

    2. 启动训练 
       训练指令
       python3 main.py


验证
    1. 每训练1个interval会进行一次评估，具体步数可以在main.py中进行修改
       
## 迁移学习指导

- 数据集准备。
    1.  获取数据。请参见“快速上手”中的数据集准备。
    2.  数据目录结构如下：    
        ```
        |--PACS
        │   |--art_painting
        │   |--cartoon
        │   |--photo
        │   |--sketch
        ``` 
-   模型训练。
    参考“模型训练”中训练步骤。

-   模型评估。
    参考“模型训练”中验证步骤。

## 高级参考
- 脚本和示例代码
```
.
├── LICENSE
├── README
├── filelist
├── log
├── data_generator.py
├── main.py
├── masf_func.py
├── boot_modelarts.py
├── utils.py
└── special_grads.py
```

    -   脚本参数
       -  meta Batch size: 126
       -  inner_lr       : 1e-05
       - outer_lr       : 1e-05
       - metric_lr      : 1e-05
       -  margin         : 10
       -  gradients_clip_value : 2.0
       -  num_classes    : 7
       -  train_iterations: 300
       -  summary_interval: 1
       -  print_interval  : 1
       - test_print_interval : 1



### 训练过程<a name="section1589455252218"></a>
1. 通过“模型训练”中的训练指令启动训练。 
2. 将训练脚本（main.py）中的dataroot设置为训练数据集的路径。 
3. 将训练脚本（main.py）中的logdir设置为模型存储路径，包括训练的log以及checkpoints文件。
4. 将脚本（masf_func.py）中的self.WEIGHTS_PATH 设置为预训练权重存储路径。

### 推理/验证过程<a name="section1465595372416"></a>
1. 每个interval训练执行完成后模型会做一次eval，产生checkpoint以及eva文件： 
   脚本会自动执行验证流程，验证结果会输出到log文件夹下，
   下方给出我们的复现精度

### 训练精度
以下是开启混合精度，经过多次测试，得到的最佳精度为
![混合精度](%E6%B7%B7%E5%90%88%E7%B2%BE%E5%BA%A6.png)

以下是各精度对比图
![精度对比](%E7%B2%BE%E5%BA%A6.png)

我们测试了目标域为Cartoon的精度指标

由上图可知，我们得到的NPU训练精度（无论是否开启混合精度）均略高于GPU精度，但均低于论文精度

其中NPU训练得到的精度与GPU训练得到精度相比高了2.027%，NPU训练（开启混合精度）得到的精度与GPU训练得到精度相比高了0.25%，基本持平，开启混合精度之后相比于不开精度损失了1.7%。略有下降，但训练速度大幅提升（具体数据见下方训练性能）

低于论文指标是因为在原作者的论文及GITHUB中所提供的数据集链接失效，我们所用的数据集为自己网上搜索的相近小数据集，本模型是提取共同特征，得到泛化效果好的网络，数据越多效果越好，我们使用的数据集较小，故得到的精度也与论文存在一定差距

 
### 训练性能

我们在NPU和GPU程序中均去除了测试和保存模型等程序，测试了在相同条件下GPU与NPU训练所花费的时间,

我们每隔1个Iteration打印了当前时间，共测试了99个Iteration花费的时间，并取平均，得到每1个Iteration所需要的时间

以下是打印GPU训练所消耗的时间
![GPU](GPU%E6%80%A7%E8%83%BD.png)

以下是打印NPU训练所消耗的时间(开启混合精度)

![NPU](NPU%E6%80%A7%E8%83%BD.png)

由打印信息我们可以看到，GPU99Iterations需要133.377s，开启混合精度后NPU99Iterations需要115.603s 

我们还测试未开启混合精度时的训练时长，99Iterations需要大约198s，可见开启混合精度之后仅仅牺牲了少量的精度便大幅增加了训练的速度

故NPU训练性能优于GPU，所需要的时间为GPU的86.674%，比GPU快13.326%

## 训练过程中的程序文件汇总

### 程序文件汇总于以下OBS链接

URL:
https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=+Ang3FM+ea8yFRyrtuC1YtByeWBzWWShkNrTxwShcutO2dDITmdx1NkPIPxZqu0xJ6T5N0GL2fg4sM027902p5r7VRmHB1s33N2eYIRDCwmcltigZbUhaZSm0sRPy0TQ1rPZLoyl97Ix2Mhou9VY6scMTUwcLJ8UsT4kZDPsj5MVX5DYrg5aH0kOwQ2JEqIbE2FlMKHCx4XTCnzqIQkBlNNSlCh0hTPGjrFyrFxzZ/eJ9/MixI/Kew/N6mskdAnN5bpjvNUOWYBtIdIUxZKhgaE8NtocQ48syrFnytb21Tnfj3O7d02lJ9rYMNsnrh9q21pVB9h596AJ1Zp2f2y0k1EWWHrukIXP1t6rd5jd5unbPl7Haetgq3KxYRpqVFVaaRwRYMYxBnI+bYr45nkuVHfrtZECcf1SjS5ZdHoISuY6tzvosZizYYnukxPRfTWCXEogKNinPD6+GDoQKQMvNXJMCf3uLsHoENhpbFOAlpPDFUwZ1oq5PWwfnk50AmmVGQl8Ebn1xSiHQpcKO7SGI0Ar8knStq5U1l1gZw5tbEu6hGzYOBdPQlJyce+tNjTGujuRJ/JXaTuEDhKCLFOZGjifrfjWVeK/RIreTJ0VIAFNF9XEP79m3jym6DB9GvZxh9coQpDCBTHjt57wUgDqiB98HiD2jZsBFjxqwCr5hJkdwNf2/I6D5PPg6QVdju/+3AbhlVDdYXkMNxJ+xenXVhQTryvLhlUZCRYfWyCl6eQUBl14r/+MXFj4V4CosCEN

提取码:
123456

*有效期至: 2022/12/09 00:05:31 GMT+08:00
