## 基本信息
**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Face Detection**

**版本（Version）：1.2**

**修改时间（Modified） ：2022.5.13**

**大小（Size）：8M**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Benchmark**

**描述（Description）：基于TensorFlow框架的FaceBoxes人脸检测网络训练代码**
## 概述
FaceBoxes是一个实时人脸检测网络。其主要特点为速度快，可以在CPU上实时运行。FaceBoxes包含了1）RDCL层，用于快速缩减输入图片的尺寸，提取图片信息，从而使FaceBoxes能够在CPU上 实时运行。2）MSCL层，其包含了reception结构用于丰富感受野，并且通过在不同的层中设置anchors来识别不同尺寸大小的人脸。3）新的anchor稠密化策略，通过增加anchors的密度来增强对小尺寸人脸的识别能力。
- 参考论文：
[FaceBoxes: A CPU Real-time Face Detector with High Accuracy](http://cn.arxiv.org/pdf/1708.05234v4)

- 参考实现：
[FaceBoxes-tensorflow](https://gitee.com/majunfu0519/FaceBoxes-tensorflow)




## 支持特性
| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 否    |
| 混合精度  | 否    |
| 并行数据  | 否    |

## 训练环境准备

1.  硬件环境准备请参见各硬件产品文档"[驱动和固件安装升级指南]( https://support.huawei.com/enterprise/zh/category/ai-computing-platform-pid-1557196528909)"。需要在硬件设备上安装与CANN版本配套的固件与驱动。
2.  宿主机上需要安装Docker并登录[Ascend Hub中心](https://ascendhub.huawei.com/#/detail?name=ascend-tensorflow-arm)获取镜像。

    当前模型支持的镜像列表如[表1](#zh-cn_topic_0000001074498056_table1519011227314)所示。

    **表 1** 镜像列表

    <a name="zh-cn_topic_0000001074498056_table1519011227314"></a>
    <table><thead align="left"><tr id="zh-cn_topic_0000001074498056_row0190152218319"><th class="cellrowborder" valign="top" width="47.32%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001074498056_p1419132211315"><a name="zh-cn_topic_0000001074498056_p1419132211315"></a><a name="zh-cn_topic_0000001074498056_p1419132211315"></a><em id="i1522884921219"><a name="i1522884921219"></a><a name="i1522884921219"></a>镜像名称</em></p>
    </th>
    <th class="cellrowborder" valign="top" width="25.52%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001074498056_p75071327115313"><a name="zh-cn_topic_0000001074498056_p75071327115313"></a><a name="zh-cn_topic_0000001074498056_p75071327115313"></a><em id="i1522994919122"><a name="i1522994919122"></a><a name="i1522994919122"></a>镜像版本</em></p>
    </th>
    <th class="cellrowborder" valign="top" width="27.16%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001074498056_p1024411406234"><a name="zh-cn_topic_0000001074498056_p1024411406234"></a><a name="zh-cn_topic_0000001074498056_p1024411406234"></a><em id="i723012493123"><a name="i723012493123"></a><a name="i723012493123"></a>配套CANN版本</em></p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="zh-cn_topic_0000001074498056_row71915221134"><td class="cellrowborder" valign="top" width="47.32%" headers="mcps1.2.4.1.1 "><a name="zh-cn_topic_0000001074498056_ul81691515131910"></a><a name="zh-cn_topic_0000001074498056_ul81691515131910"></a><ul id="zh-cn_topic_0000001074498056_ul81691515131910"><li><em id="i82326495129"><a name="i82326495129"></a><a name="i82326495129"></a>ARM架构：<a href="https://ascend.huawei.com/ascendhub/#/detail?name=ascend-tensorflow-arm" target="_blank" rel="noopener noreferrer">ascend-tensorflow-arm</a></em></li><li><em id="i18233184918125"><a name="i18233184918125"></a><a name="i18233184918125"></a>x86架构：<a href="https://ascend.huawei.com/ascendhub/#/detail?name=ascend-tensorflow-x86" target="_blank" rel="noopener noreferrer">ascend-tensorflow-x86</a></em></li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="25.52%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001074498056_p1450714271532"><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><em id="i72359495125"><a name="i72359495125"></a><a name="i72359495125"></a>20.2.0</em></p>
    </td>
    <td class="cellrowborder" valign="top" width="27.16%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001074498056_p18244640152312"><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><em id="i162363492129"><a name="i162363492129"></a><a name="i162363492129"></a><a href="https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software" target="_blank" rel="noopener noreferrer">20.2</a></em></p>
    </td>
    </tr>
    </tbody>
    </table>


## 快速上手

- 源码准备

  单击“立即下载”，并选择合适的下载方式下载源码包，并解压到合适的工作路径。
## 数据集准备
- 模型训练使用[WIDAR数据集](http://shuoyang1213.me/WIDERFACE/)进行训练，下载WIDAR数据集后，可以使用src/preparedata/preparedata.py生成训练所需要的tfrecords文件，其中的数据源路径和数据生成路径请自行按需修改

- 生成的数据集文件目录结构：

```
    ├── WIDER  
    │    ├── train_shards  
    │    │    ├──shard-0000.tfrecords  
    │    │    ├──shard-0001.tfrecords  
    │    │    ├──.....................  
    │    ├── val_shards  
    │    │    ├──shard-0000.tfrecords  
    │    │    ├──shard-0001.tfrecords  
    │    │    ├──.....................  
```
- 此处提供准备好的位于OBS的数据集供验证

```
obs://faceboxes/data/WIDER/
```

## 超参设置
在训练中，需要在本仓库根目录中的config.json文件中设定相应的超参，文件中已经给出了超参的默认设置

- 默认超参

```
    -    batch size: 16
    -    weight_decay: 1e-3
    -    score_threshold: 0.05
    -    iou_threshold: 0.3
    -    localization_loss_weight: 1.0
    -    classification_loss_weight: 1.0
    -    lr_boundaries: [160000, 200000]
    -    lr_values: [0.004, 0.0004, 0.00004]
    -    nms_threshold: 0.99
```

## 代码目录结构

```

    ├── test.sh　　　　　　　　　　　　　　　　//训练测试脚本  
    ├── train.py　　　　　　　　　　　　　　　　//开启训练代码  
    ├── pip-requirements　　　　　　　　　　　//环境依赖  
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

```

## 模型训练

- 启动训练之前，首先要配置程序运行相关环境变量。

  环境变量配置信息参见：

  [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend 910训练平台环境变量设置?sort_id=3148819)

  模型测试所需数据集：  使用src/preparedata/preparedata.py所生成的WIDER/train_shards中的tfrecords文件。

  以及参数文件：`config.json`。

  你可以通过训练来验证模型的正确性，执行命令：

  ```
  python3.7 ${code_dir}/train.py --data_path=${dataset_path} --output_path=${output_path} --step=100
  ```
  输出为位于model文件夹下的ckpt文件


## 模型测试
  模型测试所需数据集：使用src/preparedata/preparedata.py所生成的WIDER/val_shards中的tfrecords文件。

 ```
 python3.7 ${code_dir}/5_evaluation_bop_basic.py --data_path=${dataset_path} --output_path=${output_path}
 ```



## 训练结果

1.FaceBoxes在GPU(TeslaV100)训练的部分日志如下：
```
2021-12-23 11:14:36,327 - tensorflow - INFO - global_step/sec: 3.98735
2021-12-23 11:14:36,327 - tensorflow - INFO - loss = 2.3544056, step = 212600 (50.158 sec)
2021-12-23 11:15:26,352 - tensorflow - INFO - global_step/sec: 3.99802
2021-12-23 11:15:26,352 - tensorflow - INFO - loss = 3.1948407, step = 212800 (50.025 sec)
2021-12-23 11:16:16,443 - tensorflow - INFO - global_step/sec: 3.99272
2021-12-23 11:16:16,444 - tensorflow - INFO - loss = 2.8203194, step = 213000 (50.092 sec)
2021-12-23 11:17:06,369 - tensorflow - INFO - global_step/sec: 4.00593
2021-12-23 11:17:06,369 - tensorflow - INFO - loss = 2.8895893, step = 213200 (49.925 sec)
2021-12-23 11:17:56,151 - tensorflow - INFO - global_step/sec: 4.01751
2021-12-23 11:17:56,151 - tensorflow - INFO - loss = 3.014898, step = 213400 (49.782 sec)
2021-12-23 11:18:46,356 - tensorflow - INFO - global_step/sec: 3.98364
2021-12-23 11:18:46,357 - tensorflow - INFO - loss = 2.5961294, step = 213600 (50.205 sec)
2021-12-23 11:19:36,267 - tensorflow - INFO - global_step/sec: 4.00716
2021-12-23 11:19:36,267 - tensorflow - INFO - loss = 1.9023525, step = 213800 (49.910 sec)
2021-12-23 11:20:25,957 - tensorflow - INFO - global_step/sec: 4.02491
2021-12-23 11:20:25,958 - tensorflow - INFO - loss = 3.02169, step = 214000 (49.691 sec)
2021-12-23 11:21:15,638 - tensorflow - INFO - global_step/sec: 4.02573
2021-12-23 11:21:15,638 - tensorflow - INFO - loss = 3.2040527, step = 214200 (49.680 sec)
2021-12-23 11:22:05,567 - tensorflow - INFO - global_step/sec: 4.00565
2021-12-23 11:22:05,568 - tensorflow - INFO - loss = 3.0034096, step = 214400 (49.929 sec)
```


2.FaceBoxes在昇腾910训练的部分训练日志如下：
```

INFO:tensorflow:global_step/sec: 1.13204
INFO:tensorflow:loss = 2.0918577, step = 220720 (0.883 sec)
INFO:tensorflow:global_step/sec: 1.00185
INFO:tensorflow:loss = 2.2461889, step = 220721 (0.998 sec)
INFO:tensorflow:global_step/sec: 1.03824
INFO:tensorflow:loss = 2.5442588, step = 220722 (0.963 sec)
INFO:tensorflow:global_step/sec: 1.06501
INFO:tensorflow:loss = 2.4318302, step = 220723 (0.939 sec)
INFO:tensorflow:global_step/sec: 1.31278
INFO:tensorflow:loss = 2.220363, step = 220724 (0.762 sec)
INFO:tensorflow:global_step/sec: 0.696667
INFO:tensorflow:loss = 2.5294564, step = 220725 (1.435 sec)
INFO:tensorflow:global_step/sec: 1.11117
INFO:tensorflow:loss = 2.8285213, step = 220726 (0.900 sec)
INFO:tensorflow:global_step/sec: 1.08999
INFO:tensorflow:loss = 3.3644197, step = 220727 (0.917 sec)
INFO:tensorflow:global_step/sec: 1.04899
INFO:tensorflow:loss = 2.186323, step = 220728 (0.953 sec)

```

3.默认训练240000step后，可在obs/训练作业/output/model文件夹下得到相应的网络模型

   | 迁移模型    | 训练次数  |  NPU final loss  |  GPU final loss  | 
   | ---------- | --------  | --------  |  -------- |
   |FaceBoxes   | 240000   | 2.2218564      |2.480684    |

   |EVAL环境| AP | FN| FP | mean_iou | precision | recall | threshold |
   | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
   |GPU| 0.34621242 | 23307.0 | 628810.0 | 0.033031974 | 0.60916406 | 0.3432753 | 0.26472586 |
   |NPU| | | | | | | |

