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

**描述（Description）：基于TensorFlow框架的DeltaEncoder图像合成及分类网络训练代码** 

<h2 id="概述.md">概述</h2>

DeltaEncoder可以通过看到一些例子来为一个少镜头的类别合成新的样本。然后，合成的样本被用来训练分类器。

- 参考论文：

    [Schwartz, E., Karlinsky, L., Shtok, J., Harary, S., Marder, M., Feris, R., ... & Bronstein, A. M. (2018). Delta-encoder: an effective sample synthesis method for few-shot object recognition. _arXiv preprint arXiv:1806.04734_.](https://arxiv.org/pdf/1806.04734.pdf) 

## Requirements
- Python 3.7.5.
- Tensorflow 1.15.0
- scikit-learn
- Huawei Ascend
	```
	支持镜像：ascend-share/5.0.4.alpha002_tensorflow-ascend910-cp37-euleros2.8-aarch64-training:1.15.0-21.0.2_1217
	```
## 数据集
VGG16网络处理后的miniImageNet数据集，每张图像被编码成2048维的特征向量。
```
数据集下载地址：
链接：https://pan.baidu.com/s/1ozaXBWt4iQLUC1DhTq-wug 
提取码：9r3w
```
数据集下载后放入data文件夹。
## 代码及路径解释
```
Delta-Encoder
└─
  ├─README.md
  ├─LICENSE  
  ├─data        存放数据集文件夹
  ├─ckpt_npu    存放checkpoint文件夹
  ├─test
	├─output    存放模型运行日志文件夹
	├─run_1p.sh 代码运行脚本
  ├─main.py     执行主函数代码
  ├─deltaencoder.py   定义模型结构
```

## Running the code
### Run command
#### Use bash
```
bash ./test/run_1p.sh
```
#### Run directly

```
python main.py
```
参数注释：
```
data_set: dataset name, default is string 'mIN'

path: dataset path

num_shots: shot number. In the original paper, it was 1 and 5

num_epoch: train epoch number, default is 6

num_ways: ways number. In the original paper, it was 5

```

### Training log
#### 训练性能分析
|  平台| 性能 |
|--|--|
|  GPU(V100)| 5.18s/epoch |
|  NPU(Ascend910)| 10s/epoch |
####  1-shot 5-way 精度结果
##### GPU结果
```
Unseen classes accuracy without training: 26.05
epoch 1: Higher unseen classes accuracy reached: 42.88 (Saved in model_weights/mIN_1_shot_42.88_acc.npy)
epoch 2: Higher unseen classes accuracy reached: 52.6866666667 (Saved in model_weights/mIN_1_shot_52.69_acc.npy)
epoch 3: Higher unseen classes accuracy reached: 57.1933333333 (Saved in model_weights/mIN_1_shot_57.19_acc.npy)
epoch 4: Higher unseen classes accuracy reached: 60.1 (Saved in model_weights/mIN_1_shot_60.1_acc.npy)
epoch 5: Higher unseen classes accuracy reached: 60.3733333333 (Saved in model_weights/mIN_1_shot_60.37_acc.npy)
epoch 6: Higher unseen classes accuracy reached: 61.0033333333 (Saved in model_weights/mIN_1_shot_61.0_acc.npy)
```
	
##### NPU结果
```
Unseen classes accuracy without training: 26.410000000000007
epoch 1: Higher unseen classes accuracy reached: 42.09666666666667
epoch 2: Higher unseen classes accuracy reached: 52.983333333333334
epoch 3: Higher unseen classes accuracy reached: 57.31666666666666
epoch 4: Higher unseen classes accuracy reached: 58.493333333333354
epoch 5: Higher unseen classes accuracy reached: 59.89333333333333
epoch 6: Higher unseen classes accuracy reached: 61.436666666666675
```
#### 模型保存
```
chenckpoint文件
链接：https://pan.baidu.com/s/1fvyXpeeAKfAcVMBfOtFJ4g 
提取码：od8h
```