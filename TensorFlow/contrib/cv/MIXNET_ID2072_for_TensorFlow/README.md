# MIXNET
mixnet即混合深度卷积，在一次卷积中自然地混合多个卷积核大小，大内核提取高级语义信息，小内盒提取位置边缘信息，以此获得更好的精度和效率。参考文章为 MixConv: Mixed Depthwise Convolutional Kernels   参考项目： https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet/mixnet.

# setup
* python 3.7.5+
* tensorflow-gpu 1.15.0+
* numpy 1.14.1+


# Train  

数据集： 请用户自行准备好数据集，包含训练集和验证集两部分，可选用的数据集包括ImageNet2012，CIFAR10、Flower等，包含train和val两部分。格式为TFRecord文件。

Boot File：boot_modelarts.py

Pretrained: obs://cann-lhz/data_url/result_mix

# 精度对比
测试集：Imagenet2012

论文精度：

| Model | Top1 Accuracy | Top5 Accuracy |
| :--------: |:----------: | :-----------: |
| MixNet-S |   75.8%    |   92.8%   |
| MixNet-M |   77.0%    |   93.3%   |
| MixNet-L |   78.9%    |   94.2%   |

GPU目标精度：

| Model | Top1 Accuracy | Top5 Accuracy |
| :--------: |:----------: | :-----------: |
| MixNet-S |   78.1%    |   94.3%   |
| MixNet-M |   79.8%    |   95.3%   |
| MixNet-L |   80.2%    |   94.6%   |

Ascend精度：

| Model | Top1 Accuracy | Top5 Accuracy |
| :--------: |:----------: | :-----------: |
| MixNet-S |   78.3%    |   94.8%   |
| MixNet-M |   80.1%    |   96.0%   |
| MixNet-L |   80.7%    |   95.2%   |

# 性能对比：

| GPU V100  | Ascend 910 | 
| :--------: | --------| 
|   5.2global_step/s  | 11.0global_step/s   | 



