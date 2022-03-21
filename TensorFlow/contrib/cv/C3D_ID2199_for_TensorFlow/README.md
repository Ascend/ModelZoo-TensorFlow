### C3D网络概述 
C3D网络是在论文《Learning Spatiotemporal Features with 3D Convolutional Networks》中提出的。论文进行了系统的研究，找到了3DConvNets的最佳时间核长度，展示了C3D可以同时对外观和运动信息进行建模，在各种视频分析任务上优于2DConvNet特征。论文主要发现了：1）3D ConvNets比2D ConvNets更适用于时空特征的学习；2）对于3D ConvNet而言，在所有层使用3×3×3的小卷积核效果最好；3）通过简单的线性分类器学到的特征名为C3D，在4个不同的基准上优于现有的方法，并在其他2个基准上与目前最好的方法相当。

- 参考论文
  
    [Learning Spatiotemporal Features with 3D Convolutional Networks](https://arxiv.org/pdf/1412.0767.pdf)

- 参考实现
    
    [https://github.com/2012013382/C3D-Tensorflow-slim](https://github.com/2012013382/C3D-Tensorflow-slim)

### 默认配置

- 数据集：UCF101，下载地址为：[https://www.crcv.ucf.edu/data/UCF101/UCF101.rar](https://www.crcv.ucf.edu/data/UCF101/UCF101.rar)

- 训练超参

  - batch_size = 60
  - learning_rate = 1e-4
  - epoch = 40
  - Optimizer = AdamOptimizer  


###  训练环境

1. 系统：
    - linux ubuntu   
2. 软件：

    - ffmpeg
    - athena-jot
    - tensorflow 1.15
    - python 3.7
    - numpy
    


### 代码及路径解释

```
C3D
└─ 
  ├─README.md
  ├─list 该目录存放视频文件处理脚本
  	├─convert_images_to_list.sh
  	└─convert_video_to_images.sh
  ├─C3D_model.py C3D 网络模型
  ├─data_processing.py 数据预处理 
  ├─crop_mean.npy 样本均值
  ├─npu_config.py npu配置信息
  ├─train_c3d.py 模型训练代码
  ├─run_npu.sh 启动脚本

```
crop_mean.npy文件地址为：[链接：https://pan.baidu.com/s/1A8duXw03tMfGVatBExINTw 
提取码：1234](https://pan.baidu.com/s/1A8duXw03tMfGVatBExINTw)

### 训练过程及结果

- 启动训练

    1）执行shell脚本处理视频文件：
    ```
    bash /home/test_user05/C3D_slim/list/convert_video_to_images.sh /home/test_user05/UCF101 5
    
    bash /home/test_user05/C3D_slim/list/convert_images_to_list.sh /home/test_user05/UCF101 4
    ```
    2）执行训练启动脚本:
    ```
    sh run_npu.sh
    ```
    3）在npu服务器上的部分训练日志如下：
    ```
    Epoch 10: Average loss is: 0.81364; Average accuracy is: 0.77684
    Validation loss is 1.62754; Accuracy is 0.65487
    10 Train Time=420.177
    Epoch 11: Average loss is: 0.64110; Average accuracy is: 0.82443
    Validation loss is 1.66329; Accuracy is 0.66513
    11 Train Time=416.794
    Epoch 12: Average loss is: 0.49321; Average accuracy is: 0.85967
    Validation loss is 1.58833; Accuracy is 0.68410
    12 Train Time=410.353
    Epoch 13: Average loss is: 0.41822; Average accuracy is: 0.88346
    Validation loss is 1.69819; Accuracy is 0.68974
    13 Train Time=405.952
    Epoch 14: Average loss is: 0.33561; Average accuracy is: 0.90522
    Validation loss is 1.71363; Accuracy is 0.69641
    14 Train Time=413.631
    Epoch 15: Average loss is: 0.31190; Average accuracy is: 0.91272
    Validation loss is 1.54347; Accuracy is 0.71231
    15 Train Time=411.198
    Epoch 16: Average loss is: 0.28212; Average accuracy is: 0.92163
    Validation loss is 1.66356; Accuracy is 0.71846
    16 Train Time=412.336
    Epoch 17: Average loss is: 0.22204; Average accuracy is: 0.93601
    Validation loss is 1.46969; Accuracy is 0.72308
    17 Train Time=416.376
    Epoch 18: Average loss is: 0.22028; Average accuracy is: 0.93893
    Validation loss is 1.59654; Accuracy is 0.72769
    18 Train Time=412.192
    Epoch 19: Average loss is: 0.19966; Average accuracy is: 0.94504
    Validation loss is 1.58471; Accuracy is 0.72872
    19 Train Time=417.527
    Epoch 20: Average loss is: 0.18936; Average accuracy is: 0.94364
    Validation loss is 1.52652; Accuracy is 0.73231
    20 Train Time=417.269
    Epoch 21: Average loss is: 0.17068; Average accuracy is: 0.95229
    Validation loss is 1.63979; Accuracy is 0.74051
    21 Train Time=410.263
    Epoch 22: Average loss is: 0.16450; Average accuracy is: 0.95394
    Validation loss is 1.63075; Accuracy is 0.72256
    22 Train Time=415.795
    Epoch 23: Average loss is: 0.16350; Average accuracy is: 0.95687
    Validation loss is 1.53371; Accuracy is 0.74308
    23 Train Time=410.093
    Epoch 24: Average loss is: 0.14698; Average accuracy is: 0.95992
    Validation loss is 1.55223; Accuracy is 0.73744
    24 Train Time=409.013
    Epoch 25: Average loss is: 0.13062; Average accuracy is: 0.96323
    Validation loss is 1.56390; Accuracy is 0.72821
    25 Train Time=405.778
    Epoch 26: Average loss is: 0.14313; Average accuracy is: 0.96132
    Validation loss is 1.50331; Accuracy is 0.74513
    26 Train Time=402.997
    Epoch 27: Average loss is: 0.12417; Average accuracy is: 0.96590
    Validation loss is 1.54760; Accuracy is 0.73692
    27 Train Time=406.434
    Epoch 28: Average loss is: 0.10954; Average accuracy is: 0.96896
    Validation loss is 1.49435; Accuracy is 0.75846
    28 Train Time=405.069
    Epoch 29: Average loss is: 0.10988; Average accuracy is: 0.96832
    Validation loss is 1.54092; Accuracy is 0.74923
    29 Train Time=402.427
    Epoch 30: Average loss is: 0.12573; Average accuracy is: 0.96552
    Validation loss is 1.51195; Accuracy is 0.73641
    30 Train Time=401.656
    Epoch 31: Average loss is: 0.12794; Average accuracy is: 0.96667
    Validation loss is 1.56171; Accuracy is 0.73897
    31 Train Time=401.581
    Epoch 32: Average loss is: 0.10912; Average accuracy is: 0.97176
    Validation loss is 1.50018; Accuracy is 0.75538
    32 Train Time=405.791
    Epoch 33: Average loss is: 0.09393; Average accuracy is: 0.97277
    Validation loss is 1.54000; Accuracy is 0.75590
    33 Train Time=404.834
    Epoch 34: Average loss is: 0.06990; Average accuracy is: 0.97952
    Validation loss is 1.60886; Accuracy is 0.75846
    34 Train Time=402.437
    Epoch 35: Average loss is: 0.08334; Average accuracy is: 0.97748
    Validation loss is 1.60528; Accuracy is 0.75949
    35 Train Time=403.879
    Epoch 36: Average loss is: 0.08880; Average accuracy is: 0.97697
    Validation loss is 1.81799; Accuracy is 0.76154
    36 Train Time=405.191
    Epoch 37: Average loss is: 0.10460; Average accuracy is: 0.97201
    Validation loss is 1.42906; Accuracy is 0.74769
    37 Train Time=402.165
    Epoch 38: Average loss is: 0.08713; Average accuracy is: 0.97545
    Validation loss is 1.51129; Accuracy is 0.75231
    38 Train Time=402.128
    Epoch 39: Average loss is: 0.08598; Average accuracy is: 0.97799
    Validation loss is 1.47772; Accuracy is 0.76667
    39 Train Time=402.467
    Epoch 40: Average loss is: 0.07357; Average accuracy is: 0.97863
    Validation loss is 1.46879; Accuracy is 0.75385
    40 Train Time=403.560
    2021-10-13 02:44:06.385274: W tf_adapter/util/ge_plugin.cc:127] [GePlugin] can not find Environment variable : JOB_ID
    2021-10-13 02:44:08.100216: W tf_adapter/util/infershape_util.cc:303] The InferenceContext of node _SOURCE is null.
    2021-10-13 02:44:08.100311: W tf_adapter/util/infershape_util.cc:303] The InferenceContext of node _SINK is null.
    2021-10-13 02:44:08.101941: W tf_adapter/util/infershape_util.cc:303] The InferenceContext of node init is null.
    2021-10-13 02:44:15.779491: W tf_adapter/util/infershape_util.cc:303] The InferenceContext of node _SOURCE is null.
    2021-10-13 02:44:15.779544: W tf_adapter/util/infershape_util.cc:303] The InferenceContext of node _SINK is null.
    2021-10-13 02:44:15.779941: W tf_adapter/util/infershape_util.cc:303] The InferenceContext of node save/restore_all is null.
    2021-10-13 02:44:17.263133: W tf_adapter/util/infershape_util.cc:303] The InferenceContext of node _SOURCE is null.
    2021-10-13 02:44:17.263210: W tf_adapter/util/infershape_util.cc:303] The InferenceContext of node _SINK is null.
    Test accuracy is 0.75652

    ```
- C3D网络模型在GPU和NPU上的性能对比

    |       训练环境        |        性能     |
    | -------------------- | ----------------|
    | GPU                  | 475 s / epoch   | 
    | NPU                  | 425 s / epoch   | 
