## C3D模型离线推理

C3D网络是在论文《Learning Spatiotemporal Features with 3D Convolutional Networks》中提出的。论文进行了系统的研究，找到了3DConvNets的最佳时间核长度，展示了C3D可以同时对外观和运动信息进行建模，在各种视频分析任务上优于2DConvNet特征。论文主要发现了：1）3D ConvNets比2D ConvNets更适用于时空特征的学习；2）对于3D ConvNet而言，在所有层使用3×3×3的小卷积核效果最好；3）通过简单的线性分类器学到的特征名为C3D，在4个不同的基准上优于现有的方法，并在其他2个基准上与目前最好的方法相当。

### 推理环境

- CANN软件包版本：Ascend-cann-[xxx]_5.0.4.alpha001_linux-x86_64 
- Ascend 310
- atc转模工具：请参考：[ATC快速入门_昇腾CANN社区版(5.0.3.alpha002)(推理)_ATC模型转换_华为云](https://support.huaweicloud.com/atctool-cann503alpha2infer/atlasatc_16_0005.html)
- 离线推理工具：[msame](https://gitee.com/ascend/tools/tree/master/msame)
- 数据集：UCF101，下载地址为：[https://www.crcv.ucf.edu/data/UCF101/UCF101.rar](https://www.crcv.ucf.edu/data/UCF101/UCF101.rar)

### 代码及路径解释

 ```shell                    
 |-- freeze_graph.py                       -- 模型固化代码(checkpoint转pb文件)
 |-- data_convert2bin.py                   -- 数据预处理代码(生成bin文件)
 |-- run_act.sh                            -- atc转模脚本(pb文件转om模型)
 |-- run_msame.sh                          -- msame离线推理脚本
 |-- C3D_model.py                          -- C3D模型代码
 |-- reqirements.txt                          
 |-- README.md  
 ```

### 模型固化 (checkpoint转pb文件)

```shell
python3.7 freeze_graph.py
```
C3D在npu上训练的checkpoint文件及固化pb文件地址为：链接：https://pan.baidu.com/s/1WnwkE4tI5DmI984GriWmzQ 
提取码：pvyv

### atc模型转换(pb文件转om模型)

1. 执行shell脚本将pb文件转换为om模型

   ```shell
   sh run_act.sh
   ```
2. shell脚本中的atc命令参数请参考：[参数说明](https://support.huaweicloud.com/atctool-cann503alpha2infer/atlasatc_16_0038.html)

### 数据预处理，将输入数据转换为bin文件

1. C3D模型的数据预处理请参考npu迁移代码：[Gitee链接](https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/C3D_ID2199_for_TensorFlow)

2. 运行data_convert2bin.py生成bin文件

   ```shell
   python3.7 data_convert2bin.py
   ```

   数据预处理后生成的bin文件地址为: 链接：https://pan.baidu.com/s/1jNR1QANPixzjNeqTTDX6TQ 
提取码：s08y
### 离线推理

1. 请参考https://gitee.com/ascend/tools/tree/master/msame，安装msame推理环境

2. 编译成功之后，将run_msame.sh上传至msame工具的out目录下执行

   ```shell
   sh run_msame.sh
   ```
   
   shell脚本中的msame命令参数请参考:https://gitee.com/ascend/tools/tree/master/msame

   Batch: 60

   batch_clips:60,16,112,112,3

   batch_labels:60,101

   推理性能：915.251ms
   
   ![输入图片说明](image.png)


### 推理精度
|       数据集       |  NPU     | 离线推理 |
| ----------------- | -------- |  --------|
|      UCF101       | 76.696%  |   66.65% |


