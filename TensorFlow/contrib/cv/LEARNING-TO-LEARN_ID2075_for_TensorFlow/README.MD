# **LEARNING-LREARN：** 
Learntolearn网络功能：深度学习领域经常使用的更新参数的优化方法有：momentum、Rprop、Adagrad、RMSprop、ADAM，这些优化算法都是手工设计的，而且在不同的问题上要选择不同的优化方法。本模型将优化算法的设计转化成一个学习问题，通过网络学习去优化参数更新的方法,生成一个神经网络作为优化器去优化其他网络。
# **setup** 
* python 3.7+
* tensorflow 1.15.0
* mock 4.0.3
* dm-sonnet 1.11
* dill 0.3.4
# Train  
数据集：mnist数据集，地址在：obs://modelart-zs/mnist_dataset/MNIST-data/

请将从obs拷出的mnist数据四个压缩文件放在工程的MNIST-data目录下，程序中读数据的地址设在了这里。

Boot File：boot_modelarts.py
# 精度对比
本论文中精度为优化器网络优化分类网络的LOSS下降曲线，论文中的精度为下图中橙色曲线：

![输入图片说明](image/%E8%AE%BA%E6%96%87%E7%B2%BE%E5%BA%A6.png)

GPU复现精度如下图所示：

![输入图片说明](image/GPU%E5%A4%8D%E7%8E%B0%E7%B2%BE%E5%BA%A6.png)

NPU迁移后精度如下图所示：

![输入图片说明](image/NPU%E5%A4%8D%E7%8E%B0%E7%B2%BE%E5%BA%A6.png)

上图可以看出，NPU迁移后loss下降速度（即优化器网络精度）与GPU保持一致。

# 性能对比
GPU上：Mean epoch time: 0.34 s

NPU上：Mean epoch time: 41.41 s