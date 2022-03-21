# DDcGAN-master

#### 简介：
DDcGAN网络，一种用于多分辨率图像融合的双鉴别器条件生成对抗网络
通过双重鉴别器条件生成对抗性网络，用于融合不同分辨率的红外和可见光图像。支持融合不同分辨率的源图像，在公开数据集上进行的定性和定量实验表明在视觉效果和定量指标方面都优于最新技术。

启动训练
bash train_testcase.sh


#### 模型训练

1.  配置训练参数
首先在脚本DDcGAN-master/same-resolution-vis-ir-image-fusion/train_testcase.sh中，配置训练数据集路径、模型输出路径、图片输出路径，请用户根据实际路径配置，数据集参数。

2.  启动训练
第一步：模型预训练，首先修改DDcGAN-master/same-resolution-vis-ir-image-fusion/main.py 中IS_TRAINING参数为True，输入数据为数据集、输出数据为训练模型。
第二部：启动训练，模型训练完成后，修改DDcGAN-master/same-resolution-vis-ir-image-fusion/main.py中IS_TRAINING数据为False，输入数据为训练模型以及红外图形和可见光图像，输出文件为图片文件。
训练需要根据安装教程，配置输入与输出的路径。


#### 迁移学习指导

1.数据集准备。
将Training_Dataset.h5放到data_dir对应目录下。参考代码中的数据集存放路径如下：
数据集：user/modelarts/inputs/data_url_0/Training_Dataset.h5

训练模型：user/modelarts/inputs/data_url_0/model

红外图像和可见光图像源图片：user/modelarts/inputs/data_url_0/test_imgs


#### 环境设置
Ascend: 1*Ascend 910(32GB) | ARM: 24 核 96GB
python3.7
Tensorflow 1.15.0
opencv-python
scipy1.2.0
image
numpy
ModelArts配置：
![数据集及结果路径配置](https://images.gitee.com/uploads/images/2021/1111/092817_8b4129bc_9713443.png "屏幕[截](http://)图.png")

#### 训练精度
参考SSIM\EN两个主要指标作为精度验证。
|      | 论文精度|GPU精度 | NPU精度 |
|------|--------|--------|--------|
| SSIM | 0.5090 | 0.5425 | 0.5768 |
| EN   | 7.3493 | 7.4121 | 7.4955 |

精度NPU优于GPU.

#### 预训练
预训练checkpoint、结果checkpoint文件obs归档地址:obs://ddcgan-npu/dataset/model20211111/

#### 数据集说明：
模型训练使用TNO数据集，数据集请用户自行获取。数据集训练前需要做预处理操作，请用户修改DDcGAN-master/same-resolution-vis-ir-image-fusion/main.py 中IS_TRAINING参数为True，进行模型训练.数据集处理后，放入模型目录下，在训练脚本中指定数据集路径，可正常使用。
#### 性能数据 部分执行时间对比（单位：s）
| 序号 | GPU V100 | NPU   |
|----|----------|-------|
| 1  | 0.182    | 1.744 |
| 2  | 0.960    | 2.638 |
| 3  | 0.181    | 1.726 |
| 4  | 1.028    | 2.629 |
| 5  | 0.181    | 1.719 |
| 6  | 1.179    | 3.362 |
| 7  | 0.181    | 1.720 |
| 8  | 1.210    | 3.192 |
| 9  | 0.182    | 1.742 |

|     | V100           | NPU            |
|-----|----------------|----------------|
| 总用时 | 0:05:46.813785 | 0:31:38.089029 |
|     |                |                |

NPU性能比GPU差，已提交issue
[NPU训练性能差issue](https://gitee.com/ascend/modelzoo/issues/I4M3OI?from=project-issue)

####  gpu v100训练日志
[GPU训练日志](https://ddcgan-debug.obs.cn-north-4.myhuaweicloud.com:443/GPU_V100_Log_1211.txt?AccessKeyId=MYTHN9DR2FJJQWVQHNZK&Expires=1670327557&Signature=Gm1dBNluzTdPNIjM9GvVbkgyMWo%3D)

####  npu 训练日志
[NPU训练日志](https://ddcgan-debug.obs.cn-north-4.myhuaweicloud.com:443/NPU%E6%97%A5%E5%BF%97_1211.txt?AccessKeyId=MYTHN9DR2FJJQWVQHNZK&Expires=1670328976&Signature=uLD/tXZm2fuY0ua01JLRmThNwXM%3D)

####  数据集地址：
[数据集链接](https://ddcgan-npu.obs.myhuaweicloud.com:443/dataset/Training_Dataset.h5?AccessKeyId=MYTHN9DR2FJJQWVQHNZK&Expires=1670376463&Signature=IEpvUCVbfNDIrJw7esq6Oi6GutU%3D)
