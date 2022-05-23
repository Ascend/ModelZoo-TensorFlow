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
step 12, loss = 103.19 (51654.0 examples/sec; 0.002 sec/batch)
step 12, loss = 103.04 (53065.6 examples/sec; 0.002 sec/batch)
step 12, loss = 98.33 (51807.1 examples/sec; 0.002 sec/batch)
step 12, loss = 100.44 (52187.4 examples/sec; 0.002 sec/batch)
step 12, loss = 98.23 (52911.6 examples/sec; 0.002 sec/batch)
step 12, loss = 103.22 (52083.7 examples/sec; 0.002 sec/batch)
step 12, loss = 102.44 (51974.0 examples/sec; 0.002 sec/batch)
step 12, loss = 103.07 (52389.5 examples/sec; 0.002 sec/batch)
step 12, loss = 99.82 (52435.4 examples/sec; 0.002 sec/batch)
step 12, loss = 100.67 (52402.6 examples/sec; 0.002 sec/batch)
step 12, loss = 103.34 (52851.6 examples/sec; 0.002 sec/batch)
step 12, loss = 104.92 (52632.8 examples/sec; 0.002 sec/batch)
step 12, loss = 99.91 (51539.7 examples/sec; 0.002 sec/batch)
step 12, loss = 101.96 (53410.2 examples/sec; 0.002 sec/batch)
step 12, loss = 103.25 (53166.5 examples/sec; 0.002 sec/batch)
step 12, loss = 107.41 (53335.5 examples/sec; 0.002 sec/batch)
step 12, loss = 106.33 (53546.6 examples/sec; 0.002 sec/batch)
step 12, loss = 104.44 (52291.5 examples/sec; 0.002 sec/batch)
step 12, loss = 97.90 (51609.5 examples/sec; 0.002 sec/batch)
step 12, loss = 101.81 (52298.1 examples/sec; 0.002 sec/batch)
step 12, loss = 104.01 (51590.5 examples/sec; 0.002 sec/batch)
step 12, loss = 98.72 (52032.1 examples/sec; 0.002 sec/batch)
step 12, loss = 98.53 (53200.2 examples/sec; 0.002 sec/batch)
step 12, loss = 98.74 (52317.6 examples/sec; 0.002 sec/batch)
step 12, loss = 105.42 (52232.9 examples/sec; 0.002 sec/batch)
step 12, loss = 103.77 (52665.8 examples/sec; 0.002 sec/batch)
step 12, loss = 102.57 (52396.1 examples/sec; 0.002 sec/batch)
step 12, loss = 99.36 (53234.0 examples/sec; 0.002 sec/batch)
step 12, loss = 101.95 (53615.0 examples/sec; 0.002 sec/batch)
step 12, loss = 105.18 (52422.2 examples/sec; 0.002 sec/batch)
step 12, loss = 102.93 (51704.9 examples/sec; 0.002 sec/batch)
step 12, loss = 100.61 (52369.9 examples/sec; 0.002 sec/batch)
step 12, loss = 106.17 (51225.0 examples/sec; 0.002 sec/batch)
step 12, loss = 102.04 (51813.5 examples/sec; 0.002 sec/batch)
step 12, loss = 107.66 (52369.9 examples/sec; 0.002 sec/batch)
step 12, loss = 109.57 (51673.1 examples/sec; 0.002 sec/batch)
step 12, loss = 104.66 (51388.2 examples/sec; 0.002 sec/batch)
step 12, loss = 101.40 (52514.1 examples/sec; 0.002 sec/batch)
step 12, loss = 99.98 (51337.9 examples/sec; 0.002 sec/batch)
step 12, loss = 103.62 (51916.1 examples/sec; 0.002 sec/batch)
step 12, loss = 101.46 (53105.9 examples/sec; 0.002 sec/batch)
step 12, loss = 104.52 (52599.7 examples/sec; 0.002 sec/batch)
step 12, loss = 99.36 (52396.1 examples/sec; 0.002 sec/batch)
step 12, loss = 95.70 (52884.9 examples/sec; 0.002 sec/batch)
step 12, loss = 103.42 (47148.2 examples/sec; 0.002 sec/batch)
step 12, loss = 102.05 (50889.4 examples/sec; 0.002 sec/batch)
step 12, loss = 104.18 (52441.9 examples/sec; 0.002 sec/batch)
step 12, loss = 102.44 (52109.6 examples/sec; 0.002 sec/batch)
step 12, loss = 101.26 (51546.1 examples/sec; 0.002 sec/batch)
step 12, loss = 102.05 (52468.2 examples/sec; 0.002 sec/batch)
step 12, loss = 102.58 (52051.4 examples/sec; 0.002 sec/batch)
step 12, loss = 99.98 (52540.4 examples/sec; 0.002 sec/batch)
step 12, loss = 103.54 (53553.4 examples/sec; 0.002 sec/batch)
step 12, loss = 102.53 (53342.3 examples/sec; 0.002 sec/batch)
step 12, loss = 104.03 (52402.6 examples/sec; 0.002 sec/batch)
step 12, loss = 99.35 (52805.0 examples/sec; 0.002 sec/batch)
step 12, loss = 104.60 (51903.3 examples/sec; 0.002 sec/batch)
step 12, loss = 104.71 (51896.9 examples/sec; 0.002 sec/batch)
step 12, loss = 103.18 (52945.0 examples/sec; 0.002 sec/batch)
step 12, loss = 98.93 (52019.1 examples/sec; 0.002 sec/batch)
step 12, loss = 101.61 (51858.4 examples/sec; 0.002 sec/batch)
step 12, loss = 105.64 (52187.4 examples/sec; 0.002 sec/batch)
step 12, loss = 98.17 (52070.8 examples/sec; 0.002 sec/batch)
step 12, loss = 101.07 (51698.6 examples/sec; 0.002 sec/batch)
step 12, loss = 96.97 (52858.3 examples/sec; 0.002 sec/batch)
step 12, loss = 101.28 (52045.0 examples/sec; 0.002 sec/batch)
```
<h2 id="精度指标.md">精度指标</h2>

精度：reconstruction error:
|gpu|npu|原论文|
|:----:|:----:|:----:|
|0.011|0.011|0.009|