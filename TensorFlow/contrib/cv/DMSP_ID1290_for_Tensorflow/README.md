DMSP

-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [默认配置](#默认配置.md)
-   [训练过程](#训练过程.md)
-   [精度指标](#精度指标.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Image Restorationg**

**版本（Version）：1.1**

**修改时间（Modified） ：2022.0224.20**

**大小（Size）：25M**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的用于图像恢复的深度均值偏移先验代码** 

<h2 id="概述.md">概述</h2>

在本文中，作者介绍了一个自然图像先验，它直接表示自然图像分布的高斯平滑版本。 作者将先验包含在图像恢复的公式中，作为贝叶斯估计器，这允许解决噪声盲图像恢复问题。 实验表明先验梯度对应于自然图像分布上的均值偏移向量。 此外，作者使用去噪自编码器学习均值偏移向量场，并将其用于梯度下降方法以执行贝叶斯风险最小化。 论文展示了噪声盲去模糊、超分辨率和去马赛克的竞争结果
- 参考论文：

    [Deep Mean-Shift Priors for Image Restoration](https://arxiv.org/abs/1709.03749) 

- 参考实现：

    

- Tensorflow的实现：
  
  [https://github.com/siavashBigdeli/DMSP-tensorflow](https://github.com/siavashBigdeli/DMSP-tensorflow) 
 

- 通过Git获取对应commit\_id的代码方法如下：
  
    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
## 默认配置<a name="section91661242121611"></a>

- 训练数据集预处理（以ImageNet训练集为例，仅作为用户参考示例）：

  - 图像的输入尺寸统一为(180，180，3)

- 测试数据集预处理（以Berkeley验证集为例，仅作为用户参考示例）

  - 图像的输入尺寸为统一为(180，180，3)

- 训练超参

  - Batch size: 1
  - Gaussian noise levels = 11
  - Learning rate(LR): 0.01
  - momentum : 0.9
  - Optimizer: SGD with Momentum
  - Train epoch: 300 iterations

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

   
   ├── /datasets/imagenet
   │   ├──imagenet
   │   ├──Berkeley
   ```
   
   ```
## 脚本和示例代码<a name="section08421615141513"></a>

```
```
├── src
│    ├──config.py                            //训练定义
│    ├──DAE.py                               //模型定义
│    ├──DAE_model.py                         //重载模型
│    ├──demo_DMSP.py                         //主程序
│    ├──DMSPDeblur.py                        //先验去噪
│    ├──network.py                           //其他功能函数
│    ├──ops.py                               //算子定义
```
2. 模型训练。

   运行脚本如下：

```shell
$ python ./src/demo_DMSP.py
```

3. 使用pycharm在ModelArts训练启动文件为：

   ```
   /src/demo_DMSP.py
   ```
## 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡训练
```
2022-02-23 22:32:03.855277: W tensorflow/core/platform/profile_utils/cpu_utils.cc:98] Failed to find bogomips in /proc/cpuinfo; cannot determine CPU frequency
2022-02-23 22:32:03.864021: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0xaaaade38ad00 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2022-02-23 22:32:03.864068: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
============start non-blind deblurring on Berkeley segmentation dataset==============
WARNING:tensorflow:From /home/ma-user/modelarts/user-job-dir/code/dmsp-tensorflow/DAE_model.py:37: The name tf.train.AdamOptimizer is deprecated. Please use tf.compat.v1.train.AdamOptimizer instead.

WARNING:tensorflow:From /home/ma-user/modelarts/user-job-dir/code/dmsp-tensorflow/DAE_model.py:37: The name tf.train.AdamOptimizer is deprecated. Please use tf.compat.v1.train.AdamOptimizer instead.

WARNING:tensorflow:From /home/ma-user/modelarts/user-job-dir/code/dmsp-tensorflow/DAE_model.py:38: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.

WARNING:tensorflow:From /home/ma-user/modelarts/user-job-dir/code/dmsp-tensorflow/DAE_model.py:38: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.

WARNING:tensorflow:From /home/ma-user/modelarts/user-job-dir/code/dmsp-tensorflow/DAE_model.py:42: The name tf.variable_scope is deprecated. Please use tf.compat.v1.variable_scope instead.

WARNING:tensorflow:From /home/ma-user/modelarts/user-job-dir/code/dmsp-tensorflow/DAE_model.py:42: The name tf.variable_scope is deprecated. Please use tf.compat.v1.variable_scope instead.

WARNING:tensorflow:From /home/ma-user/modelarts/user-job-dir/code/dmsp-tensorflow/DAE_model.py:7: calling Constant.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.
Instructions for updating:
Call initializer instance with the dtype argument instead of passing it to the constructor
WARNING:tensorflow:From /home/ma-user/modelarts/user-job-dir/code/dmsp-tensorflow/DAE_model.py:7: calling Constant.__init__ (from tensorflow.python.ops.init_ops) with dtype is deprecated and will be removed in a future version.
Instructions for updating:
Call initializer instance with the dtype argument instead of passing it to the constructor
WARNING:tensorflow:From /home/ma-user/modelarts/user-job-dir/code/dmsp-tensorflow/DAE_model.py:8: The name tf.get_variable is deprecated. Please use tf.compat.v1.get_variable instead.

WARNING:tensorflow:From /home/ma-user/modelarts/user-job-dir/code/dmsp-tensorflow/DAE_model.py:8: The name tf.get_variable is deprecated. Please use tf.compat.v1.get_variable instead.

====================dae================
{'layer0': <tf.Tensor 'dae/BiasAdd:0' shape=(?, ?, ?, 64) dtype=float32>, 'layer1': <tf.Tensor 'dae/layer1:0' shape=(?, ?, ?, 64) dtype=float32>, 'layer2': <tf.Tensor 'dae/BiasAdd_1:0' shape=(?, ?, ?, 64) dtype=float32>, 'layer3': <tf.Tensor 'dae/layer3:0' shape=(?, ?, ?, 64) dtype=float32>, 'layer4': <tf.Tensor 'dae/BiasAdd_2:0' shape=(?, ?, ?, 64) dtype=float32>, 'layer5': <tf.Tensor 'dae/layer5:0' shape=(?, ?, ?, 64) dtype=float32>, 'layer6': <tf.Tensor 'dae/BiasAdd_3:0' shape=(?, ?, ?, 64) dtype=float32>, 'layer7': <tf.Tensor 'dae/layer7:0' shape=(?, ?, ?, 64) dtype=float32>, 'layer8': <tf.Tensor 'dae/BiasAdd_4:0' shape=(?, ?, ?, 64) dtype=float32>, 'layer9': <tf.Tensor 'dae/layer9:0' shape=(?, ?, ?, 64) dtype=float32>, 'layer10': <tf.Tensor 'dae/BiasAdd_5:0' shape=(?, ?, ?, 64) dtype=float32>, 'layer11': <tf.Tensor 'dae/layer11:0' shape=(?, ?, ?, 64) dtype=float32>, 'layer12': <tf.Tensor 'dae/BiasAdd_6:0' shape=(?, ?, ?, 64) dtype=float32>, 'layer13': <tf.Tensor 'dae/layer13:0' shape=(?, ?, ?, 64) dtype=float32>, 'layer14': <tf.Tensor 'dae/BiasAdd_7:0' shape=(?, ?, ?, 64) dtype=float32>, 'layer15': <tf.Tensor 'dae/layer15:0' shape=(?, ?, ?, 64) dtype=float32>, 'layer16': <tf.Tensor 'dae/BiasAdd_8:0' shape=(?, ?, ?, 64) dtype=float32>, 'layer17': <tf.Tensor 'dae/layer17:0' shape=(?, ?, ?, 64) dtype=float32>, 'layer18': <tf.Tensor 'dae/BiasAdd_9:0' shape=(?, ?, ?, 64) dtype=float32>, 'layer19': <tf.Tensor 'dae/layer19:0' shape=(?, ?, ?, 64) dtype=float32>, 'layer20': <tf.Tensor 'dae/BiasAdd_10:0' shape=(?, ?, ?, 64) dtype=float32>, 'layer21': <tf.Tensor 'dae/layer21:0' shape=(?, ?, ?, 64) dtype=float32>, 'layer22': <tf.Tensor 'dae/BiasAdd_11:0' shape=(?, ?, ?, 64) dtype=float32>, 'layer23': <tf.Tensor 'dae/layer23:0' shape=(?, ?, ?, 64) dtype=float32>, 'layer24': <tf.Tensor 'dae/BiasAdd_12:0' shape=(?, ?, ?, 64) dtype=float32>, 'layer25': <tf.Tensor 'dae/layer25:0' shape=(?, ?, ?, 64) dtype=float32>, 'layer26': <tf.Tensor 'dae/BiasAdd_13:0' shape=(?, ?, ?, 64) dtype=float32>, 'layer27': <tf.Tensor 'dae/layer27:0' shape=(?, ?, ?, 64) dtype=float32>, 'layer28': <tf.Tensor 'dae/BiasAdd_14:0' shape=(?, ?, ?, 64) dtype=float32>, 'layer29': <tf.Tensor 'dae/layer29:0' shape=(?, ?, ?, 64) dtype=float32>, 'layer30': <tf.Tensor 'dae/BiasAdd_15:0' shape=(?, ?, ?, 64) dtype=float32>, 'layer31': <tf.Tensor 'dae/layer31:0' shape=(?, ?, ?, 64) dtype=float32>, 'layer32': <tf.Tensor 'dae/BiasAdd_16:0' shape=(?, ?, ?, 64) dtype=float32>, 'layer33': <tf.Tensor 'dae/layer33:0' shape=(?, ?, ?, 64) dtype=float32>, 'layer34': <tf.Tensor 'dae/BiasAdd_17:0' shape=(?, ?, ?, 64) dtype=float32>, 'layer35': <tf.Tensor 'dae/layer35:0' shape=(?, ?, ?, 64) dtype=float32>, 'layer36': <tf.Tensor 'dae/BiasAdd_18:0' shape=(?, ?, ?, 64) dtype=float32>, 'layer37': <tf.Tensor 'dae/layer37:0' shape=(?, ?, ?, 64) dtype=float32>, 'layer38': <tf.Tensor 'dae/BiasAdd_19:0' shape=(?, ?, ?, 3) dtype=float32>}
====================dae output=========
Tensor("strided_slice_1:0", shape=(?, ?, ?, 3), dtype=float32)
WARNING:tensorflow:From /home/ma-user/modelarts/user-job-dir/code/dmsp-tensorflow/DAE_model.py:51: The name tf.global_variables_initializer is deprecated. Please use tf.compat.v1.global_variables_initializer instead.

WARNING:tensorflow:From /home/ma-user/modelarts/user-job-dir/code/dmsp-tensorflow/DAE_model.py:51: The name tf.global_variables_initializer is deprecated. Please use tf.compat.v1.global_variables_initializer instead.

2022-02-23 22:32:14.897055: I /home/jenkins/agent/workspace/Compile_GraphEngine_Centos_ARM/tensorflow/tf_adapter/kernels/geop_npu.cc:694] The model has been compiled on the Ascend AI processor, current graph id is:1
Initialized with PSNR: 17.78958876047073
2022-02-23 22:32:40.320450: I /home/jenkins/agent/workspace/Compile_GraphEngine_Centos_ARM/tensorflow/tf_adapter/kernels/geop_npu.cc:694] The model has been compiled on the Ascend AI processor, current graph id is:11
Finished psnr = 27.65 (1.5 examples/sec; 0.646 sec/batch)
Initialized with PSNR: 21.71935774044799
Finished psnr = 29.31 (1.4 examples/sec; 0.697 sec/batch)
Initialized with PSNR: 12.418238314349477
Finished psnr = 21.70 (1.4 examples/sec; 0.704 sec/batch)
Initialized with PSNR: 17.761670521195924
Finished psnr = 27.69 (1.5 examples/sec; 0.672 sec/batch)
Initialized with PSNR: 23.028104067351563
Finished psnr = 32.53 (1.4 examples/sec; 0.704 sec/batch)
Initialized with PSNR: 15.075084013742561
Finished psnr = 27.08 (1.4 examples/sec; 0.703 sec/batch)
Initialized with PSNR: 17.302924438930848
Finished psnr = 24.16 (1.2 examples/sec; 0.824 sec/batch)
Initialized with PSNR: 17.10059787725738
Finished psnr = 25.20 (1.3 examples/sec; 0.751 sec/batch)
Initialized with PSNR: 16.07467978560146
Finished psnr = 25.66 (1.4 examples/sec; 0.712 sec/batch)
Initialized with PSNR: 15.523285818788821
Finished psnr = 25.79 (1.4 examples/sec; 0.718 sec/batch)
Initialized with PSNR: 20.173765682212093
Finished psnr = 33.91 (1.5 examples/sec; 0.688 sec/batch)
Initialized with PSNR: 17.809478987327715
Finished psnr = 29.48 (1.6 examples/sec; 0.640 sec/batch)
Initialized with PSNR: 18.0941733503732
Finished psnr = 33.18 (1.4 examples/sec; 0.702 sec/batch)
Initialized with PSNR: 17.11170706335929
Finished psnr = 24.92 (1.4 examples/sec; 0.705 sec/batch)
Initialized with PSNR: 16.409065638468267
Finished psnr = 29.45 (1.4 examples/sec; 0.727 sec/batch)
Initialized with PSNR: 16.58872443970573
Finished psnr = 26.77 (1.4 examples/sec; 0.702 sec/batch)
Initialized with PSNR: 16.632015946049982
Finished psnr = 28.54 (1.2 examples/sec; 0.805 sec/batch)
Initialized with PSNR: 14.895557404412923
Finished psnr = 25.84 (1.3 examples/sec; 0.741 sec/batch)
Initialized with PSNR: 17.557421710572992
Finished psnr = 25.67 (1.4 examples/sec; 0.702 sec/batch)
Initialized with PSNR: 23.73822886222646
Finished psnr = 31.20 (1.1 examples/sec; 0.895 sec/batch)
Initialized with PSNR: 14.288116614544533
Finished psnr = 21.96 (1.4 examples/sec; 0.735 sec/batch)
Initialized with PSNR: 19.533104118880125
Finished psnr = 28.99 (1.4 examples/sec; 0.710 sec/batch)
```
<h2 id="精度指标.md">精度指标</h2>

精度：psnr:

|gpu|npu|原论文|
|:----:|:----:|:----:|
|26.06|26.06|26.00|