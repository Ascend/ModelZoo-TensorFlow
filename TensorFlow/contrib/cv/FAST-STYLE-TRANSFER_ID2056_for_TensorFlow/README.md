- [基本信息](#基本信息.md)
- [概述](#概述.md)
- [训练环境准备](#训练环境准备.md)
- [快速上手](#快速上手.md)
- [迁移学习指导](#迁移学习指导.md)
- [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Image Process**

**版本（Version）：1.1**

**修改时间（Modified） ：2022.7.29**

**大小（Size）：21373640KB**

**框架（Framework）：TensorFlow1.15**

**模型格式（Model Format）：ckpt/pb/om**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的Fast Style Transfer图像风格迁移训练代码**

<h2 id="概述.md">概述</h2>

## 简述<a name="section194554031510"></a>

Fast-Style-Transfer模型是基于Ulyanov等人(2016)中介绍的快速风格化方法的改进。主要改进点在于将原始风格化架构中的Batch normalization改为Instance Normalization从而导致生成的图像质量得到了显著的改善。

- 参考论文：

  [https://arxiv.org/abs/1810.04805](https://gitee.com/link?target=https%3A%2F%2Farxiv.org%2Fabs%2F1810.04805)

- 参考实现：

  https://github.com/lengstrom/fast-style-transfer

- 适配昇腾 AI 处理器的实现：

  https://gitee.com/ascend/modelzoo/tree/master/contrib/TensorFlow/Research/cv/FAST-STYLE-TRANSFER_for_TensorFlow
  
- 通过Git获取对应commit\_id的代码方法如下：
  
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    

## 默认配置<a name="section91661242121611"></a>

-   网络结构
    - 24-layer, 1024-hidden, 16-heads, 340M parameters
-   训练超参（单卡）：
    - Batch size：20
    - Content-weight： 1.5e1
    - Checkpoint-iterations： 1000
    - Epoch： 2
    - Content_weight：7.5e0
    - Style_weight：1e2
    - TV_weight：2e2
    - Learning_rate：1e-3


## 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 是       |
| 混合精度   | 是       |
| 数据并行   | 是       |


## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

 在code码cfg.py脚本中，custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")

 拉起的执行脚本
```
 ./train_full_1p.sh --help

parameter explain:
        --style=${data_path}/wave.jpg \         #风格图片输入
        --checkpoint-dir=${output_path} \       #保存训练模型
        --test=${data_path}/chicago.jpg \       #测试图片输入
        --test-dir=${output_path} \             #测试图片输出保存
        --content-weight 1.5e1 \                #内容在loss函数中的权重
        --checkpoint-iterations 1000 \            #迭代轮次
        --batch-size 20 \                        #批量大小
        --train-path=${data_path}/train_min_2014 \   #训练图片路径(完整数据集)
        --vgg-path=${data_path}/imagenet-vgg-verydeep-19.mat   #预训练VGG19模型路径
```

相关代码示例:

```
custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
custom_op.parameter_map["customize_dtypes"].s = tf.compat.as_bytes("switch_config.txt")

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

1、用户自行准备好数据集，本网络包括train2014和train_min_2014任务

2、FAST-STYLE-TRANSFER训练的模型及数据集可以参考"简述 -> 参考实现"



## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练。

    - 启动训练之前，首先要配置程序运行相关环境变量。

      环境变量配置信息参见：

      [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)

    - 单卡训练
       以数据集为train2014、train_min_2014、imagenet-vgg-verydeep-19.mat、wave.jpg、chicago.jpg为例
     ```
     cd test;
	 bash train_full_1p.sh --data_path=./data/
     ```
     启动训练。

     启动单卡训练 （脚本为FAST-STYLE-TRANSFER_ID2056_for_TensorFlow/test） 

     ```
     bash train_full_1p.sh
     ```



           
<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码
```
|——checkpoint                            //保存训练模型
|——examples
	|——content                           //示例内容图片
	|——results                           //迁移结果图片
	|——style                             //示例风格图片
	|——thumbs                            //示例图片
	|——output                            //模型输出
|——src
	|——optimize.py                       //模型训练模块
	|——transform.py					   //图片风格迁移模块
	|——utils.py                          //模型中使用到的一些基本函数
	|——vgg.py                            //损失网络VGG19模型
|——test
	|——train_full_1p.sh                  //模型精度训练脚本
	|——train_performance_1p.sh			//模型性能训练脚本
|——README.md						   //代码说明文档
|——modelarts_entry_acc.py                //模型精度启动文件
|——modelarts_entry_perf.py               //模型性能启动文件
|——cfg.py                                //模型配置参数
|——evaluate.py                           //结果评估
|——style.py                              //训练文件
|——change_pb.py                          //ckpt转pb
|——tobin.py                              //数据转bin
|——tojpg.py                              //将最终结果bin转为jpg
```


## 脚本参数<a name="section6669162441511"></a>

```
* 训练超参
  --Batch size：20
  --Content-weight： 1.5e1
  --Checkpoint-iterations： 1000
  --Epoch： 2
  --Content_weight：7.5e0
  --Style_weight：1e2
  --TV_weight：2e2
  --Learning_rate：1e-3

```

## 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡，8卡网络训练。模型存储路径为${cur_path}/output/$ASCEND_DEVICE_ID，包括训练的log以及checkpoints文件。以8卡训练为例，loss信息在文件${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。