##基本信息
**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：CV**

**版本（Version）：1.0**

**修改时间（Modified） ：2022.7.30**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架进行图像生成的训练代码**

##概述
Class-Splitting-GAN是一种改进的GAN网络，通过增加可用的类标签来增强样本生成。

+ 论文地址：

https://arxiv.org/abs/1709.07359v2

+ 源码地址：

https://github.com/CIFASIS/splitting_gan

+ 适配昇腾 AI 处理器的实现：

https://gitee.com/pengheng_48/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/Class-Splitting-GAN_ID1276_for_TensorFlow

+ 通过Git获取对应commit_id的代码方法如下：


    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换

##默认配置

图像的输入尺寸为32 * 32 * 3（cifar10数据集）

##支持特性

特性列表|是否支持
:----|-----:
分布式训练|否
混合精度|是
并行数据|否

##开启混合精度
脚本已默认开启混合精度，设置precision_mode参数的脚本参考如下：

    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["graph_memory_max_size"].s = tf.compat.as_bytes(str(15 * 1024 * 1024 * 1024))
    custom_op.parameter_map["variable_memory_max_size"].s = tf.compat.as_bytes(str(15 * 1024 * 1024 * 1024))
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭


##模型训练
**1. 数据集准备（其中包括模型和cifar10数据集）：**

obs://id1276/dataset/

**2. 单击“立即下载”，并选择合适的下载方式下载源码包**

**3. 启动训练之前，首先要配置程序运行相关环境变量，环境变量配置信息参见：**

[Ascend 910](https://gitee.com/ascend/modelzoo/wikis/.gitignore) 训练平台环境变量设置

安装依赖参考：requirement.txt

**4. 单卡训练**
+ **配置训练参数：**

首先在脚本test/train_full_1p.sh中，请用户根据实际路径配置参数data_path，或者在启动训练的命令行中以参数形式下发。

+ **启动训练：**

启动单卡训练 （脚本为/test/train_full_1p.sh），代码示例如下：

    bash train_full_1p.sh --data_path=../dataset


##训练结果
+ **精度结果对比**

精度指标项|论文发布|GPU实测|NPU实测
:----|:-----:|:-----:|:-----:
inceptionscore|8.73±0.08|8.46|8.56

+ **性能结果对比**

性能指标项|论文发布|GPU实测|NPU实测
:----|:-----:|:-----:|-----:
s/iteration|--|--

##高级参考
##脚本和示例代码

    ├── README.md                                 //代码说明文档
    ├── gan_cifar_resnet_kmeans.py.py               //训练及测试代码
    ├── LICENSE                                     
    ├── modelzoo_level.txt                           //模型状态
    ├── requirements.txt                             //训练python依赖列表
    ├── test
    │    ├──train_performance_1p.sh              //单卡训练验证性能启动脚本
    │    ├──train_full_1p.sh                    //单卡全量训练启动脚本
    ├── tflib                                    //训练相关代码
    │    ├──ops 
    │    │    ├──__init__.py              
    │    │    ├──batchnorm.py    
    │    │    ├──cond_batchnorm.py
    │    │    ├──conv1d.py
    │    │    ├──conv2d.py
    │    │    ├──deconv2d.py
    │    │    ├──layernorm.py
    │    │    ├──linear.py
    │    ├──__init__.py                   
    │    ├──cifar10.py   
    │    ├──inception_score.py   
    │    ├──mnist.py   
    │    ├──plot.py   
    │    ├──save_images.py   
    │    ├──small_imagenet.py   
      
##脚本参数
    --data_path  数据集路径，无默认需要自行配置
    其余超参固定
##训练过程
1. 通过“模型训练”中的训练指令启动单卡卡训练。

2. 参考脚本的模型和图像生成样本的存储路径为./result





