- [基本信息](#基本信息.md)
- [概述](#概述.md)
- [训练环境准备](#训练环境准备.md)
- [快速上手](#快速上手.md)
- [迁移学习指导](#迁移学习指导.md)
- [高级参考](#高级参考.md)

## 基本信息

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Instance Segmentation**

**版本（Version）：1.1**

**修改时间（Modified） ：2021.12.22**

**大小（Size）：829KB**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的OSMN网络训练代码**

## 概述

   OSMN是利用modulators模块快速地调整分割网络使其可以适应特定的物体，而不需要执行数百次的梯度下降；同时不需要调整所有的参数。在视频目标分割上有两个关键的点：视觉外观和空间中持续的移动。为了同时使用视觉和空间信息，将视觉modulator和空间modulator进行合并，在第一帧的标注信息和目标空间位置的基础上分别学习如何调整主体分割网络。

- 参考论文：
      https://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_Efficient_Video_Object_CVPR_2018_paper.pdf

- 参考实现：
      https://github.com/linjieyangsc/video_seg
  
- 适配昇腾 AI 处理器的实现：    
      https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/built-in/cv/image_segmentation/OSMN_ID1103_for_TensorFlow

- 通过Git获取对应commit\_id的代码方法如下：  
    
    ```
     git clone {repository_url}    # 克隆仓库的代码
     cd {repository_name}    # 切换到模型的代码仓目录
     git checkout  {branch}    # 切换到对应分支
     git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
     cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    ```
    
    

#### 默认配置<a name="section91661242121611"></a>

- 网络结构
    -   优化器：Adam 
    -   单卡batchsize：1 

#### 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 否       |
| 混合精度   | 是       |
| 数据并行   | 是       |

#### 混合精度训练<a name="section168064817164"></a>

 混合精度训练昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

#### 开启混合精度<a name="section20779114113713"></a>

    config = tf.ConfigProto()
    custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name =  "NpuOptimizer"
    if FLAGS.precision_mode == "allow_mix_precision":
         custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")

## 训练环境准备
-  硬件环境和运行环境准备请参见《[CANN软件安装指南](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=installation-update)》
-  运行以下命令安装依赖。
```
pip3 install requirements.txt
```
说明：依赖配置文件requirements.txt文件位于模型的根目录

## 快速上手

#### 数据集准备<a name="section361114841316"></a>
- 下载MS-COCO 2017数据集
- 在TF model zoo中下载VGG16 预训练模型vgg_16.ckpt，放到 `models/` 目录下

#### 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。

- 开始训练  

  以数据目录为./data、预训练模型目录为 ./models为例:
```
  cd test
  source ./env.sh
  bash train_full_1p.sh  --data_path=../data（全量）
  bash train_performance_1p.sh --data_path=../data（功能、性能测试）
```

## 高级参考

#### 脚本和示例代码

```
├── models	
├── preprocessing
│   ├── preprocess_davis.py
│   └── preprocess_youtube.py
├── test
│   ├── env.sh
│   ├── train_full_1p.sh	
│   └── train_performance_1p.sh 
├── LICENSE
├── README.md
├── common_args.py
├── dataset_coco.py
├── dataset_davis.py
├── davis_eval.py
├── image_utils.py
├── mobilenet_v1.py
├── model_func.py
├── model_init.txt
├── modelzoo_level.txt
├── ops.py
├── osmn.py
├── osmn_coco_pretrain.py
├── osmn_eval_youtube.py
├── osmn_online_finetune.py
├── osmn_online_finetune_ytvos.py
├── osmn_train_eval.py
├── osmn_train_eval_ytvos.py
├── requirements.txt
├── util.py
├── youtube_eval.py						
└── ytvos_merge_result.py
```

#### 脚本参数<a name="section6669162441511"></a>

```
--data_path                       train data dir, default : path/to/data
```

#### 训练过程<a name="section1589455252218"></a>

1. 通过“模型训练”中的训练指令启动单卡训练。 
2. 将训练脚本（train_full_1p.sh）中的data_path设置为训练数据集的路径。具体的流程参见“模型训练”的示例。 
3. 模型存储路径为“${cur_path}/output/$ASCEND_DEVICE_ID”，包括训练的log以及checkpoints文件。
4. 以单卡训练为例，loss信息在文件${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。 


