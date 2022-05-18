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

   a.请用户自行准备好数据集，包含训练集和测试集两部分，数据集包括BSDS300，包含train和test两部分

   b.上传数据压缩包到训练环境上,解压

   
   ├── /src
   │   ├──BSDS300
   │   │   ├──images
   │   │   │   ├──train
   │   │   │   ├──test
   ```
   
   ```
## 脚本和示例代码<a name="section08421615141513"></a>

```
```
├── src
│    ├──BSDS300/                              //数据集
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
2022-03-21 23:33:35.866972: W /home/jenkins/agent/workspace/Compile_GraphEngine_Centos_X86/tensorflow/tf_adapter/util/ge_plugin.cc:124] [GePlugin] can not find Environment variable : JOB_ID
2022-03-21 23:33:39.807011: I /home/jenkins/agent/workspace/Compile_GraphEngine_Centos_X86/tensorflow/tf_adapter/kernels/geop_npu.cc:749] The model has been compiled on the Ascend AI processor, current graph id is:1
Initialized with PSNR: 18.26756789065104
2022-03-21 23:33:52.281454: I /home/jenkins/agent/workspace/Compile_GraphEngine_Centos_X86/tensorflow/tf_adapter/kernels/geop_npu.cc:749] The model has been compiled on the Ascend AI processor, current graph id is:11
Finished psnr = 25.43 (20.0 examples/sec; 0.050 sec/batch)
Initialized with PSNR: 19.61013455418367
Finished psnr = 29.58 (20.0 examples/sec; 0.050 sec/batch)
Initialized with PSNR: 16.046844525072277
Finished psnr = 26.21 (19.3 examples/sec; 0.052 sec/batch)
Initialized with PSNR: 19.088294082853533
Finished psnr = 24.01 (20.3 examples/sec; 0.049 sec/batch)
Initialized with PSNR: 27.903391840839276
Finished psnr = 33.05 (19.9 examples/sec; 0.050 sec/batch)
Initialized with PSNR: 17.58393445793693
Finished psnr = 25.87 (19.3 examples/sec; 0.052 sec/batch)
Initialized with PSNR: 21.496189549703043
Finished psnr = 27.39 (20.3 examples/sec; 0.049 sec/batch)
Initialized with PSNR: 17.183577420828943
Finished psnr = 24.84 (19.2 examples/sec; 0.052 sec/batch)
Initialized with PSNR: 18.31449854593027
Finished psnr = 27.68 (20.2 examples/sec; 0.050 sec/batch)
Initialized with PSNR: 14.78985085202309
Finished psnr = 22.40 (19.9 examples/sec; 0.050 sec/batch)
Initialized with PSNR: 18.795507564810553
Finished psnr = 27.73 (19.6 examples/sec; 0.051 sec/batch)
Initialized with PSNR: 16.154563492696358
Finished psnr = 24.16 (19.9 examples/sec; 0.050 sec/batch)
Initialized with PSNR: 19.207686742438906
Finished psnr = 27.37 (19.9 examples/sec; 0.050 sec/batch)
Initialized with PSNR: 18.436603775139783
Finished psnr = 27.64 (20.2 examples/sec; 0.050 sec/batch)
```
<h2 id="精度指标.md">精度指标</h2>

精度：psnr:

|gpu|npu|原论文|
|:----:|:----:|:----:|
|26.06|26.06|26.00|