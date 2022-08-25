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

**修改时间（Modified） ：2022.8.25**

**大小（Size）：8MB**

**框架（Framework）：TensorFlow_1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：FP32**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的FaceBoxes人脸检测网络训练代码**

<h2 id="概述.md">概述</h2>

## 简述<a name="section194554031510"></a>

FaceBoxes是一个实时人脸检测网络。其主要特点为速度快，可以在CPU上实时运行。FaceBoxes包含了1）RDCL层，用于快速缩减输入图片的尺寸，提取图片信息，从而使FaceBoxes能够在CPU上 实时运行。2）MSCL层，其包含了reception结构用于丰富感受野，并且通过在不同的层中设置anchors来识别不同尺寸大小的人脸。3）新的anchor稠密化策略，通过增加anchors的密度来增强对小尺寸人脸的识别能力。

- 参考论文：

  [http://cn.arxiv.org/pdf/1708.05234v4](FaceBoxes: A CPU Real-time Face Detector with High Accuracy)

- 参考实现：

  https://github.com/TropComplique/FaceBoxes-tensorflow

- 适配昇腾 AI 处理器的实现：
  
  https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/FaceBoxes_ID1074_for_TensorFlow

- 通过Git获取对应commit\_id的代码方法如下：
  
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    

## 默认配置<a name="section91661242121611"></a>

-   训练超参（单卡）：
    -   batch size: 16
    -   weight_decay: 1e-3
    -   score_threshold: 0.05
    -   iou_threshold: 0.3
    -   localization_loss_weight: 1.0
    -   classification_loss_weight: 1.0
    -   lr_boundaries: [160000, 200000]
    -   lr_values: [0.004, 0.0004, 0.00004]
    -   nms_threshold: 0.99


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

相关代码示例:

```
flags.DEFINE_string(name='precision_mode', default= 'allow_fp32_to_fp16',
                    help='allow_fp32_to_fp16/force_fp16/ ' 
                    'must_keep_origin_dtype/allow_mix_precision.')

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

1、用户自行准备好数据集(方法见http://shuoyang1213.me/WIDERFACE)

2、下载WIDAR数据集后，可以使用src/preparedata/preparedata.py生成训练所需要的tfrecords文件，其中的数据源路径和数据生成路径请自行按需修改

3、FaceBoxes训练的模型及数据集可以参考"简述 -> 参考实现"



## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练。

    - 启动训练之前，首先要配置程序运行相关环境变量。

      环境变量配置信息参见：

      [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

    - 单卡训练

          1. 配置训练参数
        
             首先在脚本train_full_1p.sh中，配置训练数据集路径和checkpoint保存路径，请用户根据实际路径配置，示例如下所示：
        
             ```
             # 路径参数初始化
    	     --data_path=${data_path}/WIDER/ 
    	     --output_path=./output_path 
    	     --log_step_count_steps=1 
             --step=240000 
             ```
        
          2. 启动训练（脚本为./test/train_full_1p.sh） 
        
             ```
             bash train_full_1p.sh --data_path
             ```
          3. 训练精度结果

             ```
            | 迁移模型    | 训练次数  |  NPU final loss  |  GPU final loss  | 
            | ---------- | --------  | --------  |  -------- |
            |FaceBoxes   | 240000   | 3.065      |3.0756    |
             ```
           


<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码

```
├── train.py　　　　　　　　　　　　　　　　//开启训练代码  
├── requirements　　　　　　　　　　　//环境依赖  
├── modelzoo_level　　　　　　　　　　　　//modelzoo分级  
├── modelarts_entry.py　　　　　　　　　　//modelarts_entry入口  
├── evaluate.py　　　　　　　　　　　　　　//评估代码  
├── config.json　　　　　　　　　　　　　　//训练参数与超参  
├── README.md　　　　　　　　　　　　　　　//说明文档  
├── LICENSE　　　　　　　　　　　　　　　　//证书  
├── src　　　　　　　　　　　　　　　　　　//FaceBoxes的模型代码  
│    ├──preparedata　　　　　　　　　　　　　　　　　　　　　　　　　    //准备数据  
│    │    ├──create_tfrecords.py　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　//创建tfrecord 数据
│    │    ├──preparedata.py　　　　　　　　　　　　　　　　　　　　　　　　　　　　　    　//准备数据  
│    ├──input_pipeline　　　　　　　　　　　　　　　　　　　　　　　　　//输入预处理  
│    │    ├──other_augmentations.py　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　//数据增强  
│    │    ├──random_image_crop.py　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　//图片随机分片  
│    │    ├──pipeline.py　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　//输入预处理流程   
│    ├──utils　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　//工具函数   
│    │    ├──box_utils.py　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　//box相关函数  
│    │    ├──nms.py　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　//极大抑制函数  
│    ├──anchor_generator.py　　　　　　　　　　　　　　　　　　　　　　　//anchor生成  
│    ├──constants.py　　　　　　　　　　　　　　　　　　　　　　　　　　　//一些常量定义   
│    ├──detector.py　　　　　　　　　　　　　　　　　　　　　　　　　　　//人脸检测类    
│    ├──evaluation_utils.py　　　　　　　　　　　　　　　　　　　　　　　//评估工具函数    
│    ├──losses_and_ohem.py　　　　　　　　　　　　　　　　　　　　　　　//损失函数    
│    ├──model.py 　　　　　　　　　　　　　　　　　　　　　　　　　　　　//用于evaluator的model_fn  
│    ├──network.py　　　　　　　　　　　　　　　　　　　　　　　　　　　　//网络定义   
│    ├──training_target_creation.py　　　　　　　　　　　　　　　　　　//训练目标转化  
├── model/run00　　　　　　　　　　　　　　　　　　　　　　　　　        //模型
|--test			           						#训练脚本目录
|	|--train_full_1p.sh
|	|--train_performance_1p.sh
```

## 脚本参数<a name="section6669162441511"></a>

```
--step
--data_path
--output_path
--log_step_count_steps
```

## 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡网络训练。模型存储路径为${cur_path}/output/$ASCEND_DEVICE_ID，包括训练的log以及checkpoints文件。以单卡训练为例，loss信息在文件${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。