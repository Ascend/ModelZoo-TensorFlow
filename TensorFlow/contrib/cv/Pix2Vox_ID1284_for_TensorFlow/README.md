# 3D_Object_Reconstruction

#### 基本信息
发布者（Publisher）：Huawei

应用领域（Application Domain）： 3D Object Reconstruction

版本（Version）：1.2

修改时间（Modified） ：2021.11.06

框架（Framework）：TensorFlow 1.15.0

模型格式（Model Format）：ckpt

精度（Precision）：Mixed

处理器（Processor）：昇腾910

应用级别（Categories）：Official

描述（Description）：基于TensorFlow框架的物体三维重建网络训练代码



#### 概述
A Novel Hybrid Ensemble Approach For 3D Object Reconstruction from Multi-View Monocular RGB images for Robotic Simulations. 

Code comes from: https://github.com/Ajithbalakrishnan/3D-Object-Reconstruction-from-Multi-View-Monocular-RGB-images

参考论文：https://arxiv.org/pdf/1901.11153.pdf

参考实现：

适配昇腾 AI 处理器的实现：https://toscode.gitee.com/ascend/modelzoo/pulls/5278

#### 算法架构
![Image text](https://gitee.com/zhangwx21/Object_Reconstruction/blob/master/structure_updated.png)


#### 默认配置

数据集

ShapeNet rendered images http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz

ShapeNet voxelized models http://cvgl.stanford.edu/data2/ShapeNetVox32.tg

运行环境

python 3.5

tensorflow 1.13.0

numpy 1.13.3

scipy 0.19.0

matplotlib

skimage

PyMCubes

超参数设置

batchsize:2

Learning rate(LR): 0.001/0.0005/0.0001

Train epoch: 50

#### 训练模型

The code runs on HUAWEI's Modelarts. The main program is ./Code/modelarts_entry.py


#### 其他说明

1.  Unzip the data set to ./Data/ShapeNetRendering and ./Data/ShapeNetVox32

2.  The training log on Modelarts has been saved in the log folder.

3.  Due to the limitation of running time, the program only trained on the airplane data set(02691156).
