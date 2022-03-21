# Prototypical-Networks: Prototypical Networks for Few-shot Learning

本文为小样本分类问题提出了原型网络。原型网络的思路非常简单：对于分类问题，原型网络将其看做在语义空间中寻找每一类的原型中心。针对Few-shot Learning的任务定义，原型网络训练时学习如何拟合中心。学习一个度量函数，该度量函数可以通过少量的几个样本找到所属类别在该度量空间的原型中心。测试时，Support Set中的样本来计算新的类别的聚类中心，再利用最近邻分类器的思路进行预测。本文主要针对Few-Show／Zero-Shot任务中过拟合的问题进行研究，将原型网络和聚类联系起来，和目前的一些方法进行比较，取得了不错的效果。

<img src="https://gitee.com/phoebe0507/img_gallery/raw/master/readme/prototypical-networks.png" style="zoom:67%;" />

原型模型参考[GitHub链接](https://github.com/abdulfatir/prototypical-networks-tensorflow/blob/master)，迁移训练代码到NPU。



## 环境配置

- Tensorflow 1.15
- Numpy
- Pillow
- Ascend910



## 代码路径解释

```
.
|-- LICENSE
|-- __init__.py
|-- checkpoint            ----存放训练ckpt的路径
|-- data                  ----数据集目录
|   |-- omniglot          ----omniglot数据集
|   |   |-- data          ----数据集存放
|   |   |-- splits        ----数据集读取序列
|-- scripts               ----执行目录
|   |-- run_1p.sh         ----npu执行入口
|   |-- run_gpu.sh        ----gpu执行入口
|-- boot_modelarts.py     ----modelarts平台启动接口
|-- boot_modelarts.py     ----npu/gpu/cpu配置文件
|-- help_modelarts.py     ----modelarts平台启动配置代码
|-- data.py               ----数据处理
|-- model.py              ----网络模型
|-- train_omniglot.py     ----训练代码
```



## 数据集

请用户自行准备好Omniglot数据集，包含image_background和image_evaluation两部分，解压后放在目录 /data/omniglot/data 下。

[百度网盘](https://pan.baidu.com/s/1l7gAEWIGryn1WxIccexVPg)，提取码：7bo6

或直接执行目录 /data/omniglot 下shell文件（需翻墙）：

```
bash download_omniglot.sh
```



## GPU训练及测试

在GPU上启动训练以及测试（每次训练完成后会自动进行测试），需要执行以下几步：

1. 在data.py中将数据集读取路径更改为：

   ```
   root_dir = './data/omniglot'
   ```

2. 执行 /scripts 目录下shell文件：

   ```
   bash /scripts/run_gpu.sh
   ```



## NPU训练及测试

在Modelarts平台上使用NPU进行训练以及测试，需要执行以下几步：

1. 在data.py中将数据集读取路径更改为：

   ```
   root_dir = './cache/data/omniglot'
   ```

2. 在Pycharm上完成如下配置并Apply and Run：

   ![](https://gitee.com/phoebe0507/img_gallery/raw/master/readme/modelarts.png)



## 精度指标

|                   | 论文指标 | GPU V100 | NPU Ascend910 |
| :---------------- | -------- | -------- | ------------- |
| 20-way 5-shot Acc | 98.9%    | 98.2%    | 98.2%         |



## 性能对比

| GPU V100     | NPU Ascend910 |
| ------------ | ------------- |
| 1700ms/epoch | 1550ms/epoch  |

