- [基本信息](#基本信息.md)
- [概述](#概述.md)
- [训练环境准备](#训练环境准备.md)
- [快速上手](#快速上手.md)
- [迁移学习指导](#迁移学习指导.md)
- [高级参考](#高级参考.md)


## 基本信息

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Object Detection

**版本（Version）：1.1

**修改时间（Modified） ：2021.07.19

**大小（Size）：6M

**框架（Framework）：TensorFlow 1.15.0

**模型格式（Model Format）：ckpt

**精度（Precision）：Mixed

**处理器（Processor）：昇腾910

**应用级别（Categories）：Official

**描述（Description）：基于tensorflow实现，以Resnet50为backbone的SSD目标检测网络。

## 概述

SSD-Resnet50V1-FPN将边界框的输出空间离散为一组默认框，每个特征地图位置的纵横比和比例不同。在预测时，网络为每个默认框中每个对象类别生成分数，并对该框进行调整，以更好地匹配对象形状。此外，该网络结合了来自不同分辨率的多个特征图的预测，从而自然地处理不同尺寸的物体

- 参考论文：

    https://arxiv.org/abs/1512.02325

- 参考实现：
 
    https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Detection/SSD

- 适配昇腾 AI 处理器的实现：

    https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/built-in/cv/detection/SSD-Resnet50V1-FPN_ID1463_for_TensorFlow


- 通过Git获取对应commit\_id的代码方法如下：
    
    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    ```

### 默认配置<a name="section91661242121611"></a>

1、 训练超参（单卡）

 - [SGDR](https://arxiv.org/pdf/1608.03983.pdf) with cosine decay learning rate
 - Learning rate base = 0.02 
 - Momentum = 0.9
 - Warm-up learning rate = 0.0086664
 - Warm-up steps = 8000
 - Batch size per GPU = 32
 - Number of GPUs = 1

2、 训练超参（多卡）
 - [SGDR](https://arxiv.org/pdf/1608.03983.pdf) with cosine decay learning rate
 - Learning rate base = 0.16 
 - Momentum = 0.9
 - Warm-up learning rate = 0.0693312
 - Warm-up steps = 1000
 - Batch size per GPU = 32
 - Number of GPUs = 8

### 支持特性<a name="section1899153513554"></a>

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 是    |
| 混合精度  | 是    |

## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>
相关代码示例。

```
  config_proto = tf.ConfigProto(allow_soft_placement=True)
  custom_op = config_proto.graph_options.rewrite_options.custom_optimizers.add()
  custom_op.name = 'NpuOptimizer'
  custom_op.parameter_map["use_off_line"].b = True
  custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
  config_proto.graph_options.rewrite_options.remapping = RewriterConfig.OFF
  session_config = npu_config_proto(config_proto=config_proto)
```

## 训练环境准备

-  硬件环境和运行环境准备请参见《[CANN软件安装指南](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=installation-update)》
-  运行以下命令安装依赖。
```
pip3 install requirements.txt
```
说明：依赖配置文件requirements.txt文件位于模型的根目录


## 安装依赖

1、Dllogger1.0.0（https://github.com/NVIDIA/dllogger.git）

2、pycocotools 2.0（https://github.com/philferriere/cocoapi.git）

3、mpl_toolkits

## 快速上手

### 数据集准备<a name="section361114841316"></a>

- 数据集 COCO 2017和初始模型

```
download_all.sh nvidia_ssd <data_dir_path> <checkpoint_dir_path>
```
### 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练    
   
    1. 启动训练之前，首先要配置程序运行相关环境变量。

       环境变量配置信息参见：

     [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)

- 单卡训练 

    单卡训练，以训练数据集/data/coco2017_tfrecords/coco_train.record*、评估数据集/data/coco2017_tfrecords/coco_val.record*、backbone模型/checkpoints/resnet_v1_50/model.ckpt为例：
    ```
	cd test
	source ./env.sh
	export ASCEND_DEVICE_ID=0
	bash train_full_1p.sh --data_path=/data --ckpt_path=/checkpoints
    ```

- 8卡训练

    8卡训练，以训练数据集/data/coco2017_tfrecords/coco_train.record*、评估数据集/data/coco2017_tfrecords/coco_val.record*、backbone模型/checkpoints/resnet_v1_50/model.ckpt为例：
    ```
	cd test
	source ./env.sh
	bash train_full_8p.sh --data_path=/data --ckpt_path=/checkpoints
    ```


- 模型评估
    
	基于ckpt做eval，以评估数据集/data/coco2017_tfrecords/coco_val.record*、训练的模型目录存放在/checkpoints为例：
    ```
	source ./env.sh
	cd models/reaserch
	bash examples/SSD320_evaluate.sh /checkpoints/
	```

## 高级参考

## 脚本和示例代码<a name="section08421615141513"></a>

```
├── README.md                            //代码说明文档
├── requirements.txt                     //安装依赖
├── download_all.sh                      //数据集下载
├── scripts                              
│    ├──freeze_inference_graph.py        //模型固话
├── models                                  //模型结构
│    ├──research
│    │    ├──configs
│    │    │    ├──ssd320_full_1gpus.config    //1p训练参数
│    │    │    ├──ssd320_full_8gpus.config    //8p训练参数
│    │    ├──examples
│    │    │    ├──SSD320_FP16_1GPU.sh    //1p训练拉起脚本
│    │    │    ├──SSD320_FP16_8GPU.sh    //8p训练拉起脚本
│    │    │    ├──SSD320_evaluate.sh    //评测拉起脚本
│    │    │    ├──SSD320_inference.py    //评测脚本
│    │    ├──object_dection
│    │    │    ├──model_main.py   
│    │    │    ├──model_lib.py    
│    │    │    ├──model_hparams.py    
│    │    │    ├──inputs.py   
│    │    │    ├──exporter.py
│    │    │    ├──eval_util.py 
│    │    │    ├──builders 
│    │    │    ├──anchor_generators  
│    │    │    ├──box_coders
│    │    │    ├──configs 
│    │    │    ├──core 
│    │    │    ├──data 
│    │    │    ├──data_decoders
│    │    │    ├──dataset_tools 
│    │    │    ├──inference 
│    │    │    ├──legacy 
│    │    │    ├──matchers 
│    │    │    ├──meta_architectures 
│    │    │    ├──metrics 
│    │    │    ├──models 
│    │    │    ├──nets 
│    │    │    ├──predictors 
│    │    │    ├──protos 
│    │    │    ├──samples 
│    │    │    ├──test_data 
│    │    │    ├──test_images 
│    │    │    ├──utils
│    │    │    ├──__init__.py 
│    │    │    ├──eval_util.py
│    │    │    ├──export_inference_graph.py
│    │    │    ├──export_tflite_ssd_graph.py
│    │    │    ├──export_tflite_ssd_graph_lib.py
│    │    ├──slim
├── configs
│    ├──1p.json                          //1p rank 配置文件
│    ├──8p.json                          //8p rank 配置文件
├── test
│    ├──train_full_1p.sh                 //单卡运行启动脚本(train_steps=100000)
│    ├──train_full_8p.sh                 //8卡执行脚本(train_steps=25000*8)
│    ├──train_performance_1p.sh          //单卡性能运行启动脚本(train_steps=500)
│    ├──train_performance_8p.sh          //8卡性能执行脚本(train_steps=1000)
│    ├──env.sh                           //环境变量配置文件

```

### 脚本参数<a name="section6669162441511"></a>

```
--data_path                   数据集路径
--ckpt_path                   backbone模型存放路径
--batch_size                  训练的batch size
--num_train_steps             训练的steps
--model_dir                   训练模型存放路径
--checkpoint_dir              待评测的模型路径（仅eval模式使用）
```

## 训练过程<a name="section1589455252218"></a>
```
  NA
```