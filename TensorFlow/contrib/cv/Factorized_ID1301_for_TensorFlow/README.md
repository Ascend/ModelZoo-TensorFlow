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

**修改时间（Modified） ：2022.8.26**

**大小（Size）：5.6MB**

**框架（Framework）：TensorFlow_1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的特征点检测网络训练代码**

<h2 id="概述.md">概述</h2>

## 简述<a name="section194554031510"></a>

 Factorized是一种通过分解空间嵌入对对象地标进行无监督学习的Tensorflow 实现。用于无监督地标检测

- 参考论文：

  [https://www.robots.ox.ac.uk/~vedaldi/assets/pubs/thewlis17unsupervised.pdf](Unsupervised learning of object landmarks by factorized spatial embeddings)

- 参考实现：

  https://github.com/alldbi/Factorized-Spatial-Embeddings

- 适配昇腾 AI 处理器的实现：
  
  https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/Factorized_ID1301_for_TensorFlow

- 通过Git获取对应commit\_id的代码方法如下：
  
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    

## 默认配置<a name="section91661242121611"></a>

-   训练超参（单卡）：

    - LANDMARK_N = 8
    - SAVE_FREQ = 500
    - SUMMARY_FREQ = 20
    - BATCH_SIZE = 32
    - DOWNSAMPLE_M = 4
    - DIVERSITY = 500.
    - ALIGN = 1.
    - LEARNING_RATE = 1.e-4
    - MOMENTUM = 0.5
    - RANDOM_SEED = 1234
    - WEIGHT_DECAY = 0.0005
    - SCALE_SIZE = 146
    - CROP_SIZE = 146
    - MAX_EPOCH = 200
    

## 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 是       |
| 混合精度   | 是       |
| 数据并行   | 是       |


## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

脚本默认开启混合精度，代码如下：

```
    config = tf.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"

    custom_op.parameter_map["use_off_line"].b = True
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")

    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭remap
    config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF

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

1、模型训练使用celebA数据集，数据集请用户自行获取（方法见https://github.com/alldbi/Factorized-Spatial-Embeddings）

2、数据集处理后，放入模型目录下，在训练脚本中指定数据集路径，可正常使用

4、Factorized训练的模型及数据集可以参考"简述 -> 参考实现"


## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练。

    - 启动训练之前，首先要配置程序运行相关环境变量。

      环境变量配置信息参见：

      [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

    - 单卡训练

          1. 配置训练参数
        
             首先在脚本train_full_1p.sh中，配置训练数据集路径和checkpoint保存路径，请用户根据实际路径配置data_path，output_dir，train_url，示例如下所示：
        
             ```
             # 路径参数初始化
                --input_dir=${data_path}/data10w 
                --output_dir=${output_path} 
                --data_url=./data 
                --train_url=./workplace
             ```
        
          2. 启动训练（脚本为./test/train_full_1p.sh） 
        
             ```
             bash train_full_1p.sh --data_path
             ```

          3. 训练精度结果

            |             | NPU          | GPU          | 
            | ----------- | ------------ | ------------ | 
            | loss_align | loss_align 4-8) | loss_align 4-8 |
           


<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码

```
├── LICENSE
├── README.md
├── modelzoo_level.txt
├── requirements.txt
├── VDSR.py   			   //训练启动文件                                                                                                              
├── train.py                       //调用模块1                  
├── test.py            //调用模块2                     
├── utils/warp.py                        //调用模块3	                   
├── utils/ThinPlateSplineB.py                        //调用模块4             
├── test     
│    ├──train_performance_1p.sh                //训练性能入口
│    ├──train_full_1p.sh                       //训练精度入口，包含准确率评估
```

## 脚本参数<a name="section6669162441511"></a>

```
--data_url
--train_url
--mode
--input_dir
--K
--output_dir
--batch_size
--learning_rate
--beta1
--M
--weight_decay
--random_seed
--diversity_weight
--align_weight
--scale_size
--crop_size
--max_epochs
--checkpoint
--summary_freq
--save_freq
```

## 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡训练。模型存储路径为${cur_path}/output/$ASCEND_DEVICE_ID，包括训练的log以及checkpoints文件。以单卡训练为例，loss信息在文件${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。