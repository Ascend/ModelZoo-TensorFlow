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

**修改时间（Modified） ：2022.8.29**

**大小（Size）：203.01MB**

**框架（Framework）：TensorFlow_1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架对CIFAR-10数据集进行分类的训练代码**

<h2 id="概述.md">概述</h2>

## 简述<a name="section194554031510"></a>

神经网络使最先进的方法能够在计算机视觉上取得令人难以置信的结果目标检测等任务。然而，这样的成功在很大程度上依赖于昂贵的计算资源，这阻碍了拥有廉价设备的人欣赏先进技术。在本文中，我们提出了跨阶段部分网络（CSPNet），以缓解以前的工作需要从网络架构视角。我们把问题归结为网络优化中的重复梯度信息。拟议的网络尊重通过从网络阶段的开始和结束整合特征图来获得梯度，在我们的实验中，这将计算减少了20%，而等效或甚至在ImageNet数据集上具有卓越的精度，并且在MS COCO对象检测数据集上的AP50。CSPNet易于实现，并且足够通用，可以应对基于ResNet、ResNeXt和DenseNet的体系结构

- 参考论文：

  https://openaccess.thecvf.com/content_CVPRW_2020/papers/w28/Wang_CSPNet_A_New_Backbone_That_Can_Enhance_Learning_Capability_of_CVPRW_2020_paper.pdf

- 参考实现：

  https://github.com/WongKinYiu/CrossStagePartialNetworks

- 适配昇腾 AI 处理器的实现：
  
  https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/CSPNet_ID0840_for_TensorFlow

- 通过Git获取对应commit\_id的代码方法如下：
  
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    

## 默认配置<a name="section91661242121611"></a>

-   训练超参（单卡）：

      - EPOCHS    50
      - BATCH_SIZE    64
      - LEARNING_RATE    1e-03
      - MOMENTUM    0.9
      - LAMBDA    5e-04
    

## 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 否       |
| 混合精度   | 是       |
| 数据并行   | 否       |


## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

模型默认开启混合精度：

```
    config = tf.ConfigProto()
    custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name =  "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")

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

1、模型训练使用CIFAR-10数据集，数据集请用户自行获取

2、数据集处理后，放入模型目录下，在训练脚本中指定数据集路径，可正常使用

3、CSPNet训练的模型及数据集可以参考"简述 -> 参考实现"


## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练。

    - 启动训练之前，首先要配置程序运行相关环境变量。

      环境变量配置信息参见：

      [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

    - 单卡训练

          1. 配置训练参数
        
             首先在脚本train_full_1p.sh中，配置训练数据集路径和checkpoint保存路径，请用户根据实际路径配置data_path，epochs，output_path示例如下所示：
        
             ```
             # 路径参数初始化
             batch_size=64
             --data_path=${data_path}/dataset/cifar-10-batches-py 
             --output_path=${output_path} 
             --epochs=${train_epochs} 
             --batch_size=${batch_size} 
             ```
        
          2. 启动训练（脚本为./test/train_full_1p.sh） 
        
             ```
             bash train_full_1p.sh --data_path
             ```

          3. 训练精度结果

            |精度指标项|GPU实测|NPU实测|
            |---|---|---|
            |Top-1 Acc|0.6605|0.6645|
            
            - 性能结果比对  
            
            |性能指标项|GPU实测|NPU实测|
            |---|---|---|
            |FPS|1126.45|1245.03|
           


<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码

```
├── modelzoo_level.txt
├── README.md                                 //代码说明文档
├── utils.py                                  //工具文件
├── train.py                                  //网络训练
├── test.py                               //用于衡量模型在数据集上的精度
├── requirements.txt                          //依赖列表
├── LICENSE                                   
├── checkpoint                                //checkpoint模型保存地址
├── models                                  //模型定义
├── dataset                                 //cifar-10数据集文件夹
├── test
│    ├──train_performance_1p.sh             //单卡训练验证性能启动脚本
│    ├──train_full_1p.sh                    //单卡全量训练启动脚本
```

## 脚本参数<a name="section6669162441511"></a>

```
--MOMENTUM
--LAMBDA
--LEARNING_RATE
--epochs
--batch_size
--SUMMARY
--output_path
--data_path
--DISPLAY_STEP
--VAL_EPOCH
--resume
```

## 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡训练。模型存储路径为${cur_path}/output/$ASCEND_DEVICE_ID，包括训练的log以及checkpoints文件。以单卡训练为例，loss信息在文件${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。