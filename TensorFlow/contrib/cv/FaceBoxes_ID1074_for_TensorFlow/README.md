## 基本信息
**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Face Detection**

**版本（Version）：1.2**

**修改时间（Modified） ：2022.5.13**

**大小（Size）：7M**

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


## 默认配置

- 训练超参

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



## 支持特性
| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 否    |
| 混合精度  | 否    |
| 并行数据  | 否    |

### 训练环境准备

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

## 使用方式
下载至本地->数据集准备->使用modelarts训练

## 环境

    CANN镜像：ascend-share/5.1.rc2.alpha001_tensorflow-ascend910-cp37-euleros2.8-aarch64-training:1.15.0-21.0.2_0415
```
    opencv-python
    Pillow
    tqdm
    numpy==1.19.5

```

### 快速上手

- 源码准备

  单击“立即下载”，并选择合适的下载方式下载源码包，并解压到合适的工作路径。
## 数据集准备
- 模型训练使用[WIDAR数据集](http://shuoyang1213.me/WIDERFACE/)进行训练，可以使用src/preparedata/preparedata.py生成训练所需要的tfrecords文件，其中的数据源路径和数据生成路径请自行按需修改

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
URL:
https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=uBkmRmRznP5F3QkyOs67o8NfWi7yA58V3Wr2x/OQFu0fKMA74TI0tzzjHFbAanCs90HupbwZu/XcrEBu4khSLI7pTwxnpiDvuHzZnKiWvDNbCfeipENFdB16SukE2mrKGJt3ZK+82+JkgitwFf74KbImvclQyLBxMusBm2aU6udMqJbGzKz/fc7MoY3+fCKJNMnk7u3+huqwUxl/etMraUEi+G2VkXhWm6mEdf1QcWBRaw5SxR45cIYzKOz4LQAYmmUL41VO78R7woTTSJrlEg7SUSSkz4d+9xpJHhMA2092Q+uWuB77FULovt4j/8DvHvKKuAKjU/h984S61xkX3LwfBKqCSiMvc/ZCzBFTmUpchXqULcFJyLu7ISqzD/sNZ9isEYhDSuTFCmGy3wYW+0xOMW7xwQ3VZR7q5seGrnV8x4ziwktKEH5f/UP0DDX2iDPeR1ajlMb4LD0WXYql0V04CIadsycpPHbreup2cGI77S157JP2RyF8a7yYPdouyskV+XAxxEZkP37CZlzTas4P4D5mFjzXIz4y0Ux73bJ/zFV1qxkOa1TawtMa2NFD

提取码:
123456

*有效期至: 2022/12/17 16:03:42 GMT+08:00

obs://faceboxes/data/WIDER/
```


## 训练

->在pycharm中使用modelarts插件训练，将modelarts_entry.py作为启动文件。（启动文件中使用的为根目录下的train_testcase.sh命令）

- 代码目录结构

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

## 训练日志

在GPU(TeslaV100)训练的部分日志如下：
```
2021-12-23 18:32:08,363 - tensorflow - INFO - loss = 4.5040317, step = 8115 (0.283 sec)
2021-12-23 18:32:08,634 - tensorflow - INFO - global_step/sec: 3.69221
2021-12-23 18:32:08,634 - tensorflow - INFO - loss = 4.5335054, step = 8116 (0.271 sec)
2021-12-23 18:32:08,841 - tensorflow - INFO - global_step/sec: 4.83184
2021-12-23 18:32:08,841 - tensorflow - INFO - loss = 4.82244, step = 8117 (0.207 sec)
2021-12-23 18:32:09,092 - tensorflow - INFO - global_step/sec: 3.97911
2021-12-23 18:32:09,093 - tensorflow - INFO - loss = 4.9045005, step = 8118 (0.251 sec)
2021-12-23 18:32:09,336 - tensorflow - INFO - global_step/sec: 4.10673
2021-12-23 18:32:09,336 - tensorflow - INFO - loss = 4.113819, step = 8119 (0.244 sec)
2021-12-23 18:32:09,611 - tensorflow - INFO - global_step/sec: 3.62871
2021-12-23 18:32:09,612 - tensorflow - INFO - loss = 4.9714694, step = 8120 (0.276 sec)
2021-12-23 18:32:09,875 - tensorflow - INFO - global_step/sec: 3.79396
2021-12-23 18:32:09,875 - tensorflow - INFO - loss = 4.4656787, step = 8121 (0.264 sec)
2021-12-23 18:32:10,128 - tensorflow - INFO - global_step/sec: 3.95086
2021-12-23 18:32:10,128 - tensorflow - INFO - loss = 4.4792333, step = 8122 (0.253 sec)
2021-12-23 18:32:10,348 - tensorflow - INFO - global_step/sec: 4.54943
2021-12-23 18:32:10,348 - tensorflow - INFO - loss = 4.1617427, step = 8123 (0.220 sec)
2021-12-23 18:32:10,598 - tensorflow - INFO - global_step/sec: 3.99275
2021-12-23 18:32:10,599 - tensorflow - INFO - loss = 4.222328, step = 8124 (0.250 sec)
2021-12-23 18:32:10,865 - tensorflow - INFO - global_step/sec: 3.74715
2021-12-23 18:32:10,866 - tensorflow - INFO - loss = 3.948519, step = 8125 (0.267 sec)
2021-12-23 18:32:11,121 - tensorflow - INFO - global_step/sec: 3.91349
2021-12-23 18:32:11,121 - tensorflow - INFO - loss = 3.1790533, step = 8126 (0.256 sec)
2021-12-23 18:32:11,388 - tensorflow - INFO - global_step/sec: 3.73329
2021-12-23 18:32:11,389 - tensorflow - INFO - loss = 5.00976, step = 8127 (0.268 sec)
2021-12-23 18:32:11,641 - tensorflow - INFO - global_step/sec: 3.95447
2021-12-23 18:32:11,642 - tensorflow - INFO - loss = 3.6191146, step = 8128 (0.253 sec)
2021-12-23 18:32:11,860 - tensorflow - INFO - global_step/sec: 4.58014
2021-12-23 18:32:11,860 - tensorflow - INFO - loss = 3.7436574, step = 8129 (0.218 sec)
2021-12-23 18:32:12,093 - tensorflow - INFO - global_step/sec: 4.29105
2021-12-23 18:32:12,093 - tensorflow - INFO - loss = 4.023516, step = 8130 (0.233 sec)
2021-12-23 18:32:12,346 - tensorflow - INFO - global_step/sec: 3.94978
2021-12-23 18:32:12,346 - tensorflow - INFO - loss = 2.7456815, step = 8131 (0.253 sec)
2021-12-23 18:32:12,626 - tensorflow - INFO - global_step/sec: 3.56556
2021-12-23 18:32:12,627 - tensorflow - INFO - loss = 3.3784707, step = 8132 (0.281 sec)
2021-12-23 18:32:12,849 - tensorflow - INFO - global_step/sec: 4.48467
2021-12-23 18:32:12,850 - tensorflow - INFO - loss = 4.5038557, step = 8133 (0.223 sec)
2021-12-23 18:32:13,148 - tensorflow - INFO - global_step/sec: 3.34926
2021-12-23 18:32:13,148 - tensorflow - INFO - loss = 3.7825577, step = 8134 (0.299 sec)
2021-12-23 18:32:13,392 - tensorflow - INFO - global_step/sec: 4.08948
2021-12-23 18:32:13,393 - tensorflow - INFO - loss = 4.7722354, step = 8135 (0.244 sec)
2021-12-23 18:32:13,665 - tensorflow - INFO - global_step/sec: 3.67414
2021-12-23 18:32:13,665 - tensorflow - INFO - loss = 4.6426964, step = 8136 (0.272 sec)
2021-12-23 18:32:13,912 - tensorflow - INFO - global_step/sec: 4.04654
2021-12-23 18:32:13,912 - tensorflow - INFO - loss = 4.6764326, step = 8137 (0.247 sec)
2021-12-23 18:32:14,125 - tensorflow - INFO - global_step/sec: 4.68194
2021-12-23 18:32:14,126 - tensorflow - INFO - loss = 4.2825456, step = 8138 (0.214 sec)
2021-12-23 18:32:14,372 - tensorflow - INFO - global_step/sec: 4.04804
2021-12-23 18:32:14,373 - tensorflow - INFO - loss = 5.8155127, step = 8139 (0.247 sec)
2021-12-23 18:32:14,644 - tensorflow - INFO - global_step/sec: 3.68788
2021-12-23 18:32:14,644 - tensorflow - INFO - loss = 4.5302157, step = 8140 (0.271 sec)
2021-12-23 18:32:14,867 - tensorflow - INFO - global_step/sec: 4.47345
2021-12-23 18:32:14,868 - tensorflow - INFO - loss = 3.046206, step = 8141 (0.224 sec)
2021-12-23 18:32:15,124 - tensorflow - INFO - global_step/sec: 3.88845
2021-12-23 18:32:15,125 - tensorflow - INFO - loss = 4.4650183, step = 8142 (0.257 sec)
2021-12-23 18:32:15,371 - tensorflow - INFO - global_step/sec: 4.05365
2021-12-23 18:32:15,371 - tensorflow - INFO - loss = 4.321868, step = 8143 (0.247 sec)
2021-12-23 18:32:15,623 - tensorflow - INFO - global_step/sec: 3.96703
2021-12-23 18:32:15,623 - tensorflow - INFO - loss = 4.0855937, step = 8144 (0.252 sec)
2021-12-23 18:32:15,849 - tensorflow - INFO - global_step/sec: 4.42165
2021-12-23 18:32:15,850 - tensorflow - INFO - loss = 4.2479753, step = 8145 (0.226 sec)
2021-12-23 18:32:16,112 - tensorflow - INFO - global_step/sec: 3.80205
```

在昇腾910训练的部分训练日志如下：

```
[WARNING] TBE:2021-12-20-10:30:31 [cce_api_pass.cc:181] O0 mode fails to be compiled, the O2 mode is used
[WARNING] TBE:2021-12-20-10:30:31 [cce_api_pass.cc:181] O0 mode fails to be compiled, the O2 mode is used
[WARNING] TBE:2021-12-20-10:30:31 [cce_api_pass.cc:181] O0 mode fails to be compiled, the O2 mode is used
[WARNING] TBE:2021-12-20-10:30:37 [cce_api_pass.cc:181] O0 mode fails to be compiled, the O2 mode is used
[WARNING] TBE:2021-12-20-10:30:40 [cce_api_pass.cc:181] O0 mode fails to be compiled, the O2 mode is used
[WARNING] TBE:2021-12-20-10:30:59 [cce_api_pass.cc:181] O0 mode fails to be compiled, the O2 mode is used
[WARNING] TBE:2021-12-20-10:30:59 [cce_api_pass.cc:181] O0 mode fails to be compiled, the O2 mode is used
[WARNING] TBE:2021-12-20-10:30:59 [cce_api_pass.cc:181] O0 mode fails to be compiled, the O2 mode is used
INFO:tensorflow:loss = 11.475177, step = 1
2021-12-20 10:36:27.732359: W /home/jenkins/agent/workspace/Compile_GraphEngine_Centos_ARM/tensorflow/tf_adapter/optimizers/om_partition_subgraphs_pass.cc:2009] Dataset outputs have string output_type, please set enable_data_pre_proc=True.
2021-12-20 10:36:36.019027: I /home/jenkins/agent/workspace/Compile_GraphEngine_Centos_ARM/tensorflow/tf_adapter/kernels/geop_npu.cc:769] The model has been compiled on the Ascend AI processor, current graph id is:61
INFO:tensorflow:global_step/sec: 0.00232038
INFO:tensorflow:loss = 11.101694, step = 2 (430.852 sec)
INFO:tensorflow:global_step/sec: 0.798945
INFO:tensorflow:loss = 10.547352, step = 3 (1.249 sec)
INFO:tensorflow:global_step/sec: 0.810831
INFO:tensorflow:loss = 10.256687, step = 4 (1.233 sec)
INFO:tensorflow:global_step/sec: 0.699996
INFO:tensorflow:loss = 9.732241, step = 5 (1.429 sec)
INFO:tensorflow:global_step/sec: 0.857714
INFO:tensorflow:loss = 10.958122, step = 6 (1.166 sec)
INFO:tensorflow:global_step/sec: 0.880095
INFO:tensorflow:loss = 10.192037, step = 7 (1.136 sec)
INFO:tensorflow:global_step/sec: 0.791271
INFO:tensorflow:loss = 6.5856256, step = 8 (1.264 sec)
INFO:tensorflow:global_step/sec: 0.936172
INFO:tensorflow:loss = 9.7751045, step = 9 (1.068 sec)
INFO:tensorflow:global_step/sec: 0.910114
INFO:tensorflow:loss = 9.638695, step = 10 (1.098 sec)
INFO:tensorflow:global_step/sec: 0.866138
INFO:tensorflow:loss = 9.386629, step = 11 (1.155 sec)
INFO:tensorflow:global_step/sec: 0.474886
INFO:tensorflow:loss = 9.6329155, step = 12 (2.106 sec)
INFO:tensorflow:global_step/sec: 0.746687
INFO:tensorflow:loss = 10.039507, step = 13 (1.339 sec)
INFO:tensorflow:global_step/sec: 0.967915
INFO:tensorflow:loss = 9.833056, step = 14 (1.033 sec)
INFO:tensorflow:global_step/sec: 1.00899
INFO:tensorflow:loss = 5.5052996, step = 15 (0.991 sec)
INFO:tensorflow:global_step/sec: 1.0907
INFO:tensorflow:loss = 8.810926, step = 16 (0.917 sec)
INFO:tensorflow:global_step/sec: 0.818234
```

在昇腾910训练的部分训练日志如下：
```
INFO:tensorflow:loss = 4.859055, step = 8119 (1.307 sec)
INFO:tensorflow:global_step/sec: 0.870497
INFO:tensorflow:loss = 4.303714, step = 8120 (1.149 sec)
INFO:tensorflow:global_step/sec: 0.732091
INFO:tensorflow:loss = 4.519011, step = 8121 (1.366 sec)
INFO:tensorflow:global_step/sec: 0.811873
INFO:tensorflow:loss = 4.290802, step = 8122 (1.232 sec)
INFO:tensorflow:global_step/sec: 0.822713
INFO:tensorflow:loss = 4.555306, step = 8123 (1.216 sec)
INFO:tensorflow:global_step/sec: 0.842209
INFO:tensorflow:loss = 4.9782867, step = 8124 (1.187 sec)
INFO:tensorflow:global_step/sec: 0.826567
INFO:tensorflow:loss = 4.95797, step = 8125 (1.210 sec)
INFO:tensorflow:global_step/sec: 0.754938
INFO:tensorflow:loss = 4.736808, step = 8126 (1.325 sec)
INFO:tensorflow:global_step/sec: 0.835863
INFO:tensorflow:loss = 5.679203, step = 8127 (1.196 sec)
INFO:tensorflow:global_step/sec: 0.908829
INFO:tensorflow:loss = 5.56386, step = 8128 (1.100 sec)
INFO:tensorflow:global_step/sec: 0.801428
INFO:tensorflow:loss = 5.19129, step = 8129 (1.248 sec)
INFO:tensorflow:global_step/sec: 0.792093
INFO:tensorflow:loss = 5.604544, step = 8130 (1.262 sec)
INFO:tensorflow:global_step/sec: 0.861541
INFO:tensorflow:loss = 4.867715, step = 8131 (1.161 sec)
INFO:tensorflow:global_step/sec: 0.619856
INFO:tensorflow:loss = 3.6766508, step = 8132 (1.613 sec)
INFO:tensorflow:global_step/sec: 0.77682
INFO:tensorflow:loss = 3.7182162, step = 8133 (1.287 sec)
```

