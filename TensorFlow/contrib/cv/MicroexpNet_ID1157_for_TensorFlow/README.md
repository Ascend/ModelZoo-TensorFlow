# MicroExpNet

## 目录

1. [基本信息](#基本信息)
2. [介绍](#介绍)
3. [Citation](#citation)
4. [API](#api)
5. [Models](#models)

##基本信息
发布者（Publisher）：Huawei

应用领域（Application Domain）： Facial expression recognition

版本（Version）：1.0

修改时间（Modified） ：2021.12.18

大小（Size）：80K

框架（Framework）：TensorFlow 1.15.0

模型格式（Model Format）：ckpt

精度（Precision）：Mixed

处理器（Processor）：昇腾910

应用级别（Categories）：Demo

描述（Description）：基于TensorFlow框架的表情识别

## 介绍

MicroExpNet 是一个非常小 (不到 1MB) and 快速 (1851 FPS on i7 CPU) [TensorFlow](https://www.tensorflow.org/) 用于从正面人脸图像进行面部表情识别 (FER) 的卷积神经网络模型.其通过知识蒸馏的方式，从InceptionV3学习对于面部表情的识别。

 - 原文链接：https://arxiv.org/abs/1711.07011v4
## 引文

如果你在你的研究中使用这个模型，请引用:

```
@inproceedings{cugu2019microexpnet,
  title={MicroExpNet: An Extremely Small and Fast Model For Expression Recognition From Face Images},
  author={Cugu, Ilke and Sener, Eren and Akbas, Emre},
  booktitle={2019 Ninth International Conference on Image Processing Theory, Tools and Applications (IPTA)},
  pages={1--6},
  year={2019},
  organization={IEEE}
}
```

## 默认配置
- 数据集预处理
  - 图像输入尺寸为256*256
  - 图像标签需写入.txt文件中,训练集和测试集标签分别写在两个txt下
  - 标签格式为/cache/dataset+文件地址+空格+表情标签
  - 表情标签：0：自然 1：生气 2：鄙视 3：厌恶 4：害怕 5：高兴 6：悲伤 7：惊喜
  - 如果使用预训练好的教师网络进行预测，需先额外将整个数据集放入一个文件夹，以供教师网络预测
- 超参
  - learningRate：1e-4
  - temperature: 10
  - Optimizer: AdamOptimizer
## 环境准备
###所需包
 - matplotlib
 - opencv-python
###数据集
 所用数据集为CK+，拥有标签的人脸数据选取最后三帧，没有标签的选取第一帧，标记为0。
## 训练
 教师网络的训练通过teainTeacher进行，在scripts/train.sh中配置数据路径和超参，并启动训练，如：

 `
  --TrainTeacher=True --UseTeacherPrediction=True --studentlr=1e-4 --temperature=10 --trainlabel="/CK/label/train.txt" --testlabel="/CK/label/test.txt" --datapath="/CK/cohn-kanade"
 `
  ##精度
  |项目|paper|GPU|NPU|
  :--|-----|---|---|
  |精度|84.8|83.9$\pm$2|87.6$\pm$2|
  |性能|/|0.6|0.56|