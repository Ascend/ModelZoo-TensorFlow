# 基本信息

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）： Image Classification**

**版本（Version）：1.0**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt/pb/om**

**精度（Precision）：FP32**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Research**

# 概述

MT-net是一种神经网络结构和任务特定的学习过程，基于[Gradient-Based Meta-Learning with Learned Layerwise Metric and Subspace](https://arxiv.org/abs/1801.05558)，原论文摘要如下：

Gradient-based meta-learning methods leverage gradient descent to learn the commonalities among various tasks. While previous such methods have been successful in meta-learning tasks, they resort to simple gradient descent during meta-testing. Our primary contribution is the **MT-net**, which enables the meta-learner to learn on each layer's activation space a subspace that the task-specific learner performs gradient descent on. Additionally, a task-specific learner of an {\em MT-net} performs gradient descent with respect to a meta-learned distance metric, which warps the activation space to be more sensitive to task identity. We demonstrate that the dimension of this learned subspace reflects the complexity of the task-specific learner's adaptation task, and also that our model is less sensitive to the choice of initial learning rates than previous gradient-based meta-learning methods. Our method achieves state-of-the-art or comparable performance on few-shot classification and regression tasks.

论文中实验代码的开源链接：https://github.com/yoonholee/MT-net

实验代码中有三个任务，分别是few-shot sine wave regression，Omniglot and miniImagenet few-shot classification。这里我们选择few-shot sine wave regression。

# 数据集准备

使用数据集sinusoid（源码里用numpy随机生成该数据集，不需要额外下载数据集sinusoid）

# 推理过程

* step1：ckpt转pb

  根据后面提供的ckpt模型下载链接，下载ckpt模型，运行ckpt2pb.py，得到mt-net.pb


* step2：pb转om

​       在华为云镜像服务器上，将mt-net.pb转为mt-net.om

```
atc --model=./mt-net.pb --framework=3 --output=./mt-net --soc_version=Ascend310 --input_shape="inputa:4,5,1;inputc:4,5,1" --log=info --out_nodes="output:0"
```


* step3：模型推理

  运行2bin.py，得到数据文件inputa.bin和labela.bin

  应用msame工具，对数据文件inputa.bin和labela.bin进行离线推理，在当前目录的output文件夹下得到推理输出结果

```
./msame --model /root/mt-net/mt-net.om  --input /root/mt-net/inputa.bin,/root/mt-net/labela.bin --output /root/mt-net/output --outfmt TXT --loop 2 --debug=true
```

# 推理模型下载

obs地址（包含ckpt模型，pb模型，om模型，数据文件inputa.bin和labela.bin）：

obs://cann-id1283/