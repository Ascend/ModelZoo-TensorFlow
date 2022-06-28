# Efficientnet-CondConv
Condconv即有条件参数化卷积，为每个样本学习专有的卷积内核。用CondConv替换正常卷积，能够增加网络的大小和容量，同时保持有效的推理。参考文章为 CondConv: Conditionally Parameterized Convolutions for Efficient Inference   参考项目：https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/condconv.
   

# setup
* python 3.7.5+
* tensorflow-gpu 1.15.0+
* numpy 1.14.1+


# Train  

数据集： 请用户自行准备好数据集，包含训练集和验证集两部分，可选用的数据集包括ImageNet2012，CIFAR10、Flower等，包含train和val两部分。格式为TFRecord文件。

Boot File：boot_modelarts.py

Pretrained: obs://cann-lhz/MA-new-efficientnet-condconv_tensorflow_ID2074-11-26-22-22/output/result/archive/


# 精度对比
测试集：Imagenet2012

论文精度：

| Precision  |
| :--------: |
|   78.3%   |

GPU目标精度：

| Precision  |
| :--------: |
|   80.0%    |

Ascend精度：

| Precision  |
| :--------: |
|   80.9%    |


# 性能对比：

| GPU V100  | Ascend 910 | 
| :--------: | --------| 
|   1.90global_step/s  | 2.47global_step/s   | 



