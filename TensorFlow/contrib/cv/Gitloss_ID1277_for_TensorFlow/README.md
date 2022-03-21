-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [Requirements](#requirements)
-   [数据集](#数据集)
-   [代码及路径解释](#代码及路径解释)
-   [Running the code](#running-the-code)
	- [run script](#run-command)
	- [Training Log](#training-log)
	
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）： Image Classification**

**版本（Version）：1.1**

**修改时间（Modified） ：2021.12.21**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Research**

**描述（Description）：基于TensorFlow框架的Gitloss图像分类网络训练代码** 

<h2 id="概述.md">概述</h2>

为了进一步增强深度特征的识别能力，Git loss能够利用softmax和center loss函数联合监监督信号。Git loss的目标是最小化类内的变化以及最大化的类间距离。这种深度特征提取方法被认为是人脸识别任务的理想方法。 

- 参考论文：

    [Calefati, A., Janjua, M. K., Nawaz, S., & Gallo, I. (2018). Git loss for deep face recognition. _arXiv preprint arXiv:1807.08512_.](https://https://arxiv.org/pdf/1807.08512.pdf) 

## Requirements
- Python 3.7.5.
- Tensorflow 1.15.0
- tflearn
- Huawei Ascend
	```
	支持镜像：ascend-share/5.0.4.alpha002_tensorflow-ascend910-cp37-euleros2.8-aarch64-training:1.15.0-21.0.2_1217
	```

## 数据集
Mnist数据集
```
URL:
https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=WsAwMbRO4eNexi34otPCYuqKa2L3IUVMHVOPwtmxXLHwxeTELHUPL1QERVXjaZXWUBCwNlSqkbKMsy6JpQMFxxiCSB1FYKwzHH1jgtrjbKWCp4kSRGUB2rgyA5toFOpI5bGdulMBIeAMxVmn4/hZQyG177T7h6bkiczDnZDbBt9E1p1p5+declLouBPYCSHUKqPxmofZH08hnS3TQn7CgRecXzM3gf+R13/3krZ0aOOKhJSnD2ECaxEdTiQTfZY4YyYMvccYed0d96t8lQvcdLDrveWjPefzU7W8AHRx3ApMng1TFF8crsOEs9tqy+ifXgY28rVeWgZnKja5tcmtIVV2Gc7NLGrQ1SgzYhxCPLLdpPMPxXvvZPhJYq4qR6NTAqeCc36zh4CkI++UY5OUcr1UyJPkAnyGgdoEcHDpoDOyubl4D2D17UkzRDrciPSos79jq5eYtcYtiCJKcXARvlaoOdQqAwSxTUi9wUD5pDWNysGH2ePvmZs+laAy80QExDYcoyPvxz7lPOcDPrieDRHYY9uR/SuwJpzhITxSLrEWbLoDaxMm+N475RJ1n7mm03av9SteIQAMUlxjFIqgGMBRW8m0TAO5K5DBvuXBelE=

提取码:
123456

*有效期至: 2022/12/15 21:07:08 GMT-08:00
```
数据集可手动下载后放入data文件夹，也可在代码运行过程中自动下载
## 代码及路径解释
```
GitLoss
└─
  ├─README.md
  ├─LICENSE  
  ├─ckpt_npu    存放checkpoint文件夹
  ├─data        存放数据集文件夹
  ├─test
	├─output    存放模型运行日志文件夹
	├─run_1p.sh 代码运行脚本
  ├─gitloss.py  模型定义及主函数
```

## Running the code
### Run command
#### Use bash
```
bash ./test/run_1p.sh
```
#### Run directly

```
python gitloss.py
```
参数注释：
```
update_centers: numbers of steps after which update the centers, default is 1000

lambda_c: The weight of the center loss, default is 1.0

lambda_g: The weight of the git loss, default is 1.0

steps: The train steps, default is 8000
```

### Training log
#### 训练性能分析
|  平台| 性能 |
|--|--|
|  GPU(V100)| 10ms/step |
|  NPU(Ascend910)| 25.5ms/step |
#### 精度结果
##### GPU结果
```
两个超参数C=G=1时：
Train_Loss:	0.264, Train_Acc:	0.9992,  Valid_Acc:	0.9902, inter_cls_dist:	8.30, intra_cls_dist:	0.21
```
	
##### NPU结果
```
Step:	1000, Train_Loss:	1.0369, Train_Acc:	0.9453, Valid_Loss:	9.1808, Valid_Acc:	0.9614, inter_cls_dist:	5.6150, intra_cls_dist:	0.4148
Step:	2000, Train_Loss:	0.5643, Train_Acc:	0.9844, Valid_Loss:	12.3012, Valid_Acc:	0.9856, inter_cls_dist:	6.1626, intra_cls_dist:	0.3280
Step:	3000, Train_Loss:	0.3798, Train_Acc:	1.0000, Valid_Loss:	10.8556, Valid_Acc:	0.9904, inter_cls_dist:	6.2115, intra_cls_dist:	0.2506
Step:	4000, Train_Loss:	0.3409, Train_Acc:	1.0000, Valid_Loss:	12.2054, Valid_Acc:	0.9906, inter_cls_dist:	6.3399, intra_cls_dist:	0.2008
Step:	5000, Train_Loss:	0.3261, Train_Acc:	1.0000, Valid_Loss:	11.5283, Valid_Acc:	0.9908, inter_cls_dist:	6.3794, intra_cls_dist:	0.1702
Step:	6000, Train_Loss:	0.3204, Train_Acc:	1.0000, Valid_Loss:	10.3438, Valid_Acc:	0.9910, inter_cls_dist:	6.3469, intra_cls_dist:	0.1494
Step:	7000, Train_Loss:	0.3427, Train_Acc:	1.0000, Valid_Loss:	8.2515, Valid_Acc:	0.9894, inter_cls_dist:	6.2211, intra_cls_dist:	0.1622
Step:	8000, Train_Loss:	0.3612, Train_Acc:	1.0000, Valid_Loss:	6.3607, Valid_Acc:	0.9916, inter_cls_dist:	6.0372, intra_cls_dist:	0.1192
```
#### 模型保存
```
chenckpoint文件
链接：https://pan.baidu.com/s/1SQy_nvEdU40eEmQlvK0jVA 
提取码：6wn5
```