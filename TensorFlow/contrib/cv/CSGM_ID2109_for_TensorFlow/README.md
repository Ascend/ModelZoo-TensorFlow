CSGM

-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [默认配置](#默认配置.md)
-   [训练过程](#训练过程.md)
-   [精度指标](#精度指标.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）： Compressed sensing**

**版本（Version）：1.1**

**修改时间（Modified） ：2021.11.20**

**大小（Size）：25M**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的CSGM图像分类网络训练代码** 

<h2 id="概述.md">概述</h2>

压缩感知是，在现有的传统的信号处理模式中，信号要采样、压缩然后再传输，接收端要解压再恢复原始信号。采样过程要遵循奈奎斯特采样定理，也就是采样速率不能小于信号最高频率的两倍，这样才能保证根据采样所得的信息可以完整地恢复出原始信号。压缩感知在接收端通过合适的重构算法就可以恢复出原始信号，因此可以避免在传统的信号处理模式中的数据浪费和资源浪费问题。这篇论文是在使用generative models做压缩感知

- 参考论文：

    [Compressed Sensing using Generative Models](https://arxiv.org/abs/1703.03208) 

- 参考实现：

    

- Tensorflow的实现：
  
  [https://github.com/AshishBora/csgm](https://github.com/AshishBora/csgm) 
 

- 通过Git获取对应commit\_id的代码方法如下：
  
    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
## 默认配置<a name="section91661242121611"></a>

- 训练数据集预处理（以MNIST训练集为例，仅作为用户参考示例）：

  - 图像的输入尺寸为784(28*28)

- 测试数据集预处理（以MNIST验证集为例，仅作为用户参考示例）

  - 图像的输入尺寸为784(28*28)

- 训练超参

  - Batch size: 100
  - Learning rate(LR): 0.001
  - Optimizer: AdamOptimizer
  - Train epoch: 750

  **npu实现：**

##### 支持特性

支持混合精度训练，脚本中默认开启了混合精度，参考示例，见“开启混合精度”。

##### 混合精度训练

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

-开启混合精度<a name="section20779114113713"></a>

```
custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
```


### 准备工作

##### 训练环境的准备

硬件环境：Ascend: 1*Ascend 910(32GB) | ARM: 24 核 96GB

运行环境：ascend-share/5.0.3.alpha005_tensorflow-ascend910-cp37-euleros2.8-aarch64-training:1.15.0-21.0.2_1101

```
## 训练

To train:

#### 环境依赖

制作数据集的环境上已安装Python3.7和TensorFlow 1.15.0。

#### 操作步骤

1. 数据集准备。

   a.请用户自行准备好数据集，包含训练集和验证集两部分，数据集包括Mnist等，包含train和 	val两部分。以Mnist数据集为例。

   b.上传数据压缩包到训练环境上,无需解压

   ```
   ├── /datasets/mnist
   │   ├──t10k-images-idx3-ubyte.gz
   │   ├──t10k-labels-idx1-ubyte.gz
   │   ├──train-images-idx3-ubyte.gz
   │   ├──train-labels-idx1-ubyte.gz
   ```
## 脚本和示例代码<a name="section08421615141513"></a>

```
├── src
│    ├──models                               //模型存储
│    ├──samples                              //样本保存
│    ├──data_input.py                        //数据加载
│    ├──main.py                              //主程序
│    ├──model_def.py                         //模型定义
│    ├──util.py                              //其他功能函数
```
2. 模型训练。

   运行脚本如下：

```shell
$ python ./src/main.py
$ python ./mnist_vae/src/main.py
```

Samples are stored in `./samples/`
  - `orig*` : Orignal images
  - `recontr*` : Reconstructed by VAE
Samples are stored in `./samples/`
  - `sampled*` : Sampled from the generator of the VAE

Trained models are stored in  `./models/`


3. 使用pycharm在ModelArts训练启动文件为：

   ```
   /mnist_vae/src/main.py 
   ```
## 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡训练
```
021-12-08 13:13:30.720550: I /home/jenkins/agent/workspace/Compile_GraphEngine_Centos_ARM/tensorflow/tf_adapter/kernels/geop_npu.cc:765] The model has been compiled on the Ascend AI processor, current graph id is:1
No checkpoint found
Extracting cache/dataset/train-images-idx3-ubyte.gz
Extracting cache/dataset/train-labels-idx1-ubyte.gz
Extracting cache/dataset/t10k-images-idx3-ubyte.gz
Extracting cache/dataset/t10k-labels-idx1-ubyte.gz
start training
2021-12-08 13:13:49.596612: I /home/jenkins/agent/workspace/Compile_GraphEngine_Centos_ARM/tensorflow/tf_adapter/kernels/geop_npu.cc:765] The model has been compiled on the Ascend AI processor, current graph id is:11
step 0, loss = 621.13 (3.9 examples/sec; 25.653 sec/batch)
step 0, loss = 473.88 (15663.8 examples/sec; 0.006 sec/batch)
step 0, loss = 393.31 (22854.8 examples/sec; 0.004 sec/batch)
step 0, loss = 329.65 (23974.3 examples/sec; 0.004 sec/batch)
step 0, loss = 269.96 (24415.3 examples/sec; 0.004 sec/batch)
step 0, loss = 244.54 (24506.6 examples/sec; 0.004 sec/batch)
step 0, loss = 240.48 (23996.2 examples/sec; 0.004 sec/batch)
step 0, loss = 232.23 (24784.6 examples/sec; 0.004 sec/batch)
step 0, loss = 231.09 (25579.7 examples/sec; 0.004 sec/batch)
step 0, loss = 227.22 (26723.8 examples/sec; 0.004 sec/batch)
step 0, loss = 234.37 (24330.3 examples/sec; 0.004 sec/batch)
step 0, loss = 228.46 (23971.6 examples/sec; 0.004 sec/batch)
step 0, loss = 224.78 (25015.2 examples/sec; 0.004 sec/batch)
step 0, loss = 237.21 (24771.5 examples/sec; 0.004 sec/batch)
step 0, loss = 228.91 (24708.7 examples/sec; 0.004 sec/batch)
step 0, loss = 218.84 (24153.8 examples/sec; 0.004 sec/batch)
step 0, loss = 213.00 (24834.5 examples/sec; 0.004 sec/batch)
step 0, loss = 214.56 (25046.6 examples/sec; 0.004 sec/batch)
step 0, loss = 217.74 (25662.7 examples/sec; 0.004 sec/batch)
step 0, loss = 217.85 (24633.3 examples/sec; 0.004 sec/batch)
step 0, loss = 210.44 (24662.2 examples/sec; 0.004 sec/batch)
step 0, loss = 208.78 (24595.7 examples/sec; 0.004 sec/batch)
step 0, loss = 208.42 (24978.0 examples/sec; 0.004 sec/batch)
step 0, loss = 210.64 (24394.0 examples/sec; 0.004 sec/batch)
step 0, loss = 210.62 (24500.9 examples/sec; 0.004 sec/batch)
step 0, loss = 214.60 (25016.7 examples/sec; 0.004 sec/batch)
step 0, loss = 207.89 (24577.0 examples/sec; 0.004 sec/batch)
step 0, loss = 202.35 (24890.5 examples/sec; 0.004 sec/batch)
step 0, loss = 205.51 (24786.1 examples/sec; 0.004 sec/batch)
step 0, loss = 211.48 (23782.6 examples/sec; 0.004 sec/batch)
step 0, loss = 216.67 (24902.4 examples/sec; 0.004 sec/batch)
step 0, loss = 212.29 (24771.5 examples/sec; 0.004 sec/batch)
step 0, loss = 204.17 (24849.2 examples/sec; 0.004 sec/batch)
step 0, loss = 210.40 (24969.1 examples/sec; 0.004 sec/batch)
step 0, loss = 210.41 (29371.9 examples/sec; 0.003 sec/batch)
step 0, loss = 203.19 (28197.0 examples/sec; 0.004 sec/batch)
step 0, loss = 210.02 (28771.5 examples/sec; 0.003 sec/batch)
```
<h2 id="精度指标.md">精度指标</h2>

精度：reconstruction error:
|gpu|npu|原论文|
|:----:|:----:|:----:|
|0.011|0.011|0.009|