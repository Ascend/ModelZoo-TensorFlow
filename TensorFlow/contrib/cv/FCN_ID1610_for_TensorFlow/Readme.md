

<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Image segmentation**

**版本（Version）：1.1**

**修改时间（Modified） ：2021.12.24**

**大小（Size）：92.7k**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的ROI-FCN图像分类网络训练代码** 

<h2 id="概述.md">概述</h2>

卷积神经网络是图像分割，分类，检测的主流方法，图像识别中的许多问题只求对图像中特定感兴趣区域（ROI）进行识别，在这种情况下，将卷积神经网络（CNN）的注意力引导到给定的ROI区域而不丢失背景信息是一个主要的挑战，本程序（ROI-FCN）使用一个阀滤波器的方法集中卷积神经网络的注意力在一个给定的ROI图像上，ROI图像以二进制映射的方式与图像一起插入CNN中。网络中对ROI图像的处理是使用阀门过滤器的方法，对于作用与图像的每个过滤器，都存在这一个作用与ROI映射的相应阀过滤器。 valve滤波器卷积的输出按元素与图像滤波器卷积的输出相乘，以得到归一化的特征映射。此映射被用作网络下一层的输入。在这种情况下，该网络是一个标准的全卷积网络(FCN)用于语义分割（按像素分类）。阀滤波器可以看作是在图像的不同区域对图像滤波器的激活进行规整的一种阀。引入ROI图作为卷积神经网络输入的阀门滤波方法。图像和ROI输入分别通过一个单独的卷层，分别给出特征图和相关性图。特征映射中的每个元素与特征映射中的相应元素相乘，以给出一个规范化的特征映射，该映射被传递（在RELU之后）作为网络下一层的输入。

- 参考论文：

   [Setting an attention region for convolutional neural networks using region selective features, for recognition of materials within glass vessels](https://arxiv.org/abs/1708.08711)

- 参考实现：//github.com/sagieppel/Focusing-attention-of-Fully-convolutional-neural-networks-on-Region-of-interest-ROI-input-map-

- 适配昇腾 AI 处理器的实现：

  https://gitee.com/fb2330305086/modelzoo/tree/master/contrib/TensorFlow/Research/cv/FCN_ID1610_for_Tensorflow   


- 通过Git获取对应commit\_id的代码方法如下：

  ```
  git clone {repository_url}    # 克隆仓库的代码
  cd {repository_name}    # 切换到模型的代码仓目录
  git checkout  {branch}    # 切换到对应分支
  git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
  cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```

## 默认配置<a name="section91661242121611"></a>

​	用户自行准备好数据集  Materials_In_Vessels  训练需要预先下载好预训练网络模型vgg16

​	数据集预处理 图像输入尺寸480*854

​	随机裁剪图像尺寸 

​	随机翻转 

​	随机更改图像灰度 

​	随机添加阴影 

​	随机对图像添加噪声

​	训练超参数

​	Batch_Size：4

​	Weight_Loss_Rate=5e-4

​	learning_rate=1e-5


## 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 是       |
| 混合精度   | 是       |
| 并行数据   | 是       |

## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

未开启混合精度


<h2 id="训练环境准备.md">训练环境准备</h2>

1. 硬件环境准备请参见各硬件产品文档"[驱动和固件安装升级指南]( https://support.huawei.com/enterprise/zh/category/ai-computing-platform-pid-1557196528909)"。需要在硬件设备上安装与CANN版本配套的固件与驱动。

2. 宿主机上需要安装Docker并登录[Ascend Hub中心](https://ascendhub.huawei.com/#/detail?name=ascend-tensorflow-arm)获取镜像。

   当前模型支持的镜像列表如[表1](#zh-cn_topic_0000001074498056_table1519011227314)所示。

   **表 1** 镜像列表

   <a name="zh-cn_topic_0000001074498056_table1519011227314"></a>

   <table><thead align="left"><tr id="zh-cn_topic_0000001074498056_row0190152218319"><th class="cellrowborder" valign="top" width="47.32%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001074498056_p1419132211315"><a name="zh-cn_topic_0000001074498056_p1419132211315"></a><a name="zh-cn_topic_0000001074498056_p1419132211315"></a><em id="i1522884921219"><a name="i1522884921219"></a><a name="i1522884921219"></a>镜像名称</em></p>
   </th>
   <th class="cellrowborder" valign="top" width="25.52%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001074498056_p75071327115313"><a name="zh-cn_topic_0000001074498056_p75071327115313"></a><a name="zh-cn_topic_0000001074498056_p75071327115313"></a><em id="i1522994919122"><a name="i1522994919122"></a><a name="i1522994919122"></a>镜像版本</em></p>
   </th>
   <th class="cellrowborder" valign="top" width="27.16%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001074498056_p1024411406234"><a name="zh-cn_topic_0000001074498056_p1024411406234"></a><a name="zh-cn_topic_0000001074498056_p1024411406234"></a><em id="i723012493123"><a name="i723012493123"></a><a name="i723012493123"></a>配套CANN版本</em></p>
   </th>
   </tr>
   </thead>
   <tbody><tr id="zh-cn_topic_0000001074498056_row71915221134"><td class="cellrowborder" valign="top" width="47.32%" headers="mcps1.2.4.1.1 "><a name="zh-cn_topic_0000001074498056_ul81691515131910"></a><a name="zh-cn_topic_0000001074498056_ul81691515131910"></a><ul id="zh-cn_topic_0000001074498056_ul81691515131910"><li><em id="i82326495129"><a name="i82326495129"></a><a name="i82326495129"></a>ARM架构：<a href="https://ascend.huawei.com/ascendhub/#/detail?name=ascend-tensorflow-arm" target="_blank" rel="noopener noreferrer">ascend-tensorflow-arm</a></em></li><li><em id="i18233184918125"><a name="i18233184918125"></a><a name="i18233184918125"></a>x86架构：<a href="https://ascend.huawei.com/ascendhub/#/detail?name=ascend-tensorflow-x86" target="_blank" rel="noopener noreferrer">ascend-tensorflow-x86</a></em></li></ul>
   </td>
   <td class="cellrowborder" valign="top" width="25.52%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001074498056_p1450714271532"><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><em id="i72359495125"><a name="i72359495125"></a><a name="i72359495125"></a>20.2.0</em></p>
   </td>
   <td class="cellrowborder" valign="top" width="27.16%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001074498056_p18244640152312"><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><em id="i162363492129"><a name="i162363492129"></a><a name="i162363492129"></a><a href="https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software" target="_blank" rel="noopener noreferrer">20.2</a></em></p>
   </td>
   </tr>
   </tbody>
   </table>


<h2 id="快速上手.md">快速上手</h2>

- 数据集准备

1. 用户自行准备好数据集  Materials_In_Vessels  训练需要预先下载好预训练网络模型vgg16

## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。

- 启动训练之前，首先要配置程序运行相关环境变量。

  环境变量配置信息参见：

     [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)

- 单卡训练 

  1. 启动训练。

     启动单卡训练 

     ```
     python TRAIN.py
     ```

  

<h2 id="迁移学习指导.md">迁移学习指导</h2>

- 数据集准备。

  数据集要求如下：

  1. 获取数据。

     如果要使用自己的数据集，需要将数据集放到脚本参数data_dir对应目录下。参考代码中的数据集存放路径如下：

     - 训练集：/home/ma-user/modelarts/inputs/data_url_0/Train_Images/
     - 测试集： "/home/ma-user/modelarts/inputs/data_url_0/Test_Images_All/

     训练数据集和测试数据集以文件名中的train和Test加以区分。

     数据集也可以放在其它目录，则修改对应的脚本入参data_dir即可。

- 模型训练。

  参考“模型训练”中训练步骤。

## 训练过程<a name="section1589455252218"></a>

1.  通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡、8卡网络训练。

2.  参考脚本的模型存储路径为results/1p或者results/8p，训练脚本log中包括如下信息。

```
WARNING:tensorflow:From /usr/local/Ascend/tfplugin/latest/tfplugin/python/site-packages/npu_bridge/estimator/npu/npu_optimizer.py:273: The name tf.train.Optimizer is deprecated. Please use tf.compat.v1.train.Optimizer instead.

INFO:root:pid: None.	1000/2844
INFO:root:pid: None.	2000/2844
WARNING:tensorflow:From /home/ma-user/modelarts/user-job-dir/code/TRAIN.py:47: The name tf.ConfigProto is deprecated. Please use tf.compat.v1.ConfigProto instead.

WARNING:tensorflow:From /home/ma-user/modelarts/user-job-dir/code/TRAIN.py:47: The name tf.ConfigProto is deprecated. Please use tf.compat.v1.ConfigProto instead.

WARNING:tensorflow:From /home/ma-user/modelarts/user-job-dir/code/TRAIN.py:96: The name tf.reset_default_graph is deprecated. Please use tf.compat.v1.reset_default_graph instead.

WARNING:tensorflow:From /home/ma-user/modelarts/user-job-dir/code/TRAIN.py:96: The name tf.reset_default_graph is deprecated. Please use tf.compat.v1.reset_default_graph instead.

WARNING:tensorflow:From /home/ma-user/modelarts/user-job-dir/code/TRAIN.py:97: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.

WARNING:tensorflow:From /home/ma-user/modelarts/user-job-dir/code/TRAIN.py:97: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.

npy file loaded
build model started
WARNING:tensorflow:From /home/ma-user/modelarts/user-job-dir/code/BuildNetVgg16.py:140: The name tf.variable_scope is deprecated. Please use tf.compat.v1.variable_scope instead.

WARNING:tensorflow:From /home/ma-user/modelarts/user-job-dir/code/BuildNetVgg16.py:140: The name tf.variable_scope is deprecated. Please use tf.compat.v1.variable_scope instead.

WARNING:tensorflow:From /home/ma-user/modelarts/user-job-dir/code/BuildNetVgg16.py:51: The name tf.truncated_normal is deprecated. Please use tf.random.truncated_normal instead.

WARNING:tensorflow:From /home/ma-user/modelarts/user-job-dir/code/BuildNetVgg16.py:51: The name tf.truncated_normal is deprecated. Please use tf.random.truncated_normal instead.

WARNING:tensorflow:From /home/ma-user/modelarts/user-job-dir/code/BuildNetVgg16.py:124: The name tf.nn.max_pool is deprecated. Please use tf.nn.max_pool2d instead.

WARNING:tensorflow:From /home/ma-user/modelarts/user-job-dir/code/BuildNetVgg16.py:124: The name tf.nn.max_pool is deprecated. Please use tf.nn.max_pool2d instead.

WARNING:tensorflow:From /home/ma-user/modelarts/user-job-dir/code/TensorflowUtils.py:76: The name tf.get_variable is deprecated. Please use tf.compat.v1.get_variable instead.

WARNING:tensorflow:From /home/ma-user/modelarts/user-job-dir/code/TensorflowUtils.py:76: The name tf.get_variable is deprecated. Please use tf.compat.v1.get_variable instead.

WARNING:tensorflow:From /home/ma-user/modelarts/user-job-dir/code/BuildNetVgg16.py:83: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.
Instructions for updating:
Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.
WARNING:tensorflow:From /home/ma-user/modelarts/user-job-dir/code/BuildNetVgg16.py:83: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.
Instructions for updating:
Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.
WARNING:tensorflow:From /home/ma-user/modelarts/user-job-dir/code/BuildNetVgg16.py:119: calling argmax (from tensorflow.python.ops.math_ops) with dimension is deprecated and will be removed in a future version.
Instructions for updating:
Use the `axis` argument instead
WARNING:tensorflow:From /home/ma-user/modelarts/user-job-dir/code/BuildNetVgg16.py:119: calling argmax (from tensorflow.python.ops.math_ops) with dimension is deprecated and will be removed in a future version.
Instructions for updating:
Use the `axis` argument instead
FCN model built
WARNING:tensorflow:From /home/ma-user/anaconda/lib/python3.7/site-packages/tensorflow_core/python/util/dispatch.py:180: calling squeeze (from tensorflow.python.ops.array_ops) with squeeze_dims is deprecated and will be removed in a future version.
Instructions for updating:
Use the `axis` argument instead
WARNING:tensorflow:From /home/ma-user/anaconda/lib/python3.7/site-packages/tensorflow_core/python/util/dispatch.py:180: calling squeeze (from tensorflow.python.ops.array_ops) with squeeze_dims is deprecated and will be removed in a future version.
Instructions for updating:
Use the `axis` argument instead
WARNING:tensorflow:From /home/ma-user/modelarts/user-job-dir/code/TRAIN.py:110: The name tf.trainable_variables is deprecated. Please use tf.compat.v1.trainable_variables instead.

WARNING:tensorflow:From /home/ma-user/modelarts/user-job-dir/code/TRAIN.py:110: The name tf.trainable_variables is deprecated. Please use tf.compat.v1.trainable_variables instead.

WARNING:tensorflow:From /home/ma-user/modelarts/user-job-dir/code/TRAIN.py:89: The name tf.train.AdamOptimizer is deprecated. Please use tf.compat.v1.train.AdamOptimizer instead.

WARNING:tensorflow:From /home/ma-user/modelarts/user-job-dir/code/TRAIN.py:89: The name tf.train.AdamOptimizer is deprecated. Please use tf.compat.v1.train.AdamOptimizer instead.

600
WARNING:tensorflow:From /home/ma-user/modelarts/user-job-dir/code/TRAIN.py:119: The name tf.Session is deprecated. Please use tf.compat.v1.Session instead.

WARNING:tensorflow:From /home/ma-user/modelarts/user-job-dir/code/TRAIN.py:119: The name tf.Session is deprecated. Please use tf.compat.v1.Session instead.

2021-12-10 18:26:40.909361: W tensorflow/core/platform/profile_utils/cpu_utils.cc:98] Failed to find bogomips in /proc/cpuinfo; cannot determine CPU frequency
2021-12-10 18:26:40.917386: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0xaaaae7d0a6e0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2021-12-10 18:26:40.917435: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
Setting up Saver...
WARNING:tensorflow:From /home/ma-user/modelarts/user-job-dir/code/TRAIN.py:121: The name tf.train.Saver is deprecated. Please use tf.compat.v1.train.Saver instead.

WARNING:tensorflow:From /home/ma-user/modelarts/user-job-dir/code/TRAIN.py:121: The name tf.train.Saver is deprecated. Please use tf.compat.v1.train.Saver instead.

WARNING:tensorflow:From /home/ma-user/modelarts/user-job-dir/code/TRAIN.py:122: The name tf.global_variables_initializer is deprecated. Please use tf.compat.v1.global_variables_initializer instead.

WARNING:tensorflow:From /home/ma-user/modelarts/user-job-dir/code/TRAIN.py:122: The name tf.global_variables_initializer is deprecated. Please use tf.compat.v1.global_variables_initializer instead.

2021-12-10 18:27:05.604798: I /home/jenkins/agent/workspace/Compile_GraphEngine_Centos_ARM/tensorflow/tf_adapter/kernels/geop_npu.cc:694] The model has been compiled on the Ascend AI processor, current graph id is:1
2021-12-10 18:28:35.023314: I /home/jenkins/agent/workspace/Compile_GraphEngine_Centos_ARM/tensorflow/tf_adapter/kernels/geop_npu.cc:694] The model has been compiled on the Ascend AI processor, current graph id is:11
[WARNING] TBE:2021-12-10-18:29:08 [cce_api_pass.cc:181] O0 mode fails to be compiled, the O2 mode is used
2021-12-10 18:29:42.541449: I /home/jenkins/agent/workspace/Compile_GraphEngine_Centos_ARM/tensorflow/tf_adapter/kernels/geop_npu.cc:694] The model has been compiled on the Ascend AI processor, current graph id is:21
[WARNING] TBE:2021-12-10-18:29:54 [cce_api_pass.cc:181] O0 mode fails to be compiled, the O2 mode is used
Step 0 Train Loss=1.8704344
时间
 25.942367792129517 s/600itr
Step 500 Train Loss=0.26284456
时间
 0.33819103240966797 s/600itr
Saving Model to file in/home/ma-user/modelarts/user-job-dir/code/logs/
2021-12-10 18:39:45.250832: I /home/jenkins/agent/workspace/Compile_GraphEngine_Centos_ARM/tensorflow/tf_adapter/kernels/geop_npu.cc:694] The model has been compiled on the Ascend AI processor, current graph id is:31
Step 1000 Train Loss=0.060632046
时间
 0.3537735939025879 s/600itr
Step 1500 Train Loss=0.03761721
时间
 0.3471944332122803 s/600itr
Saving Model to file in/home/ma-user/modelarts/user-job-dir/code/logs/
Step 2000 Train Loss=0.15536883
时间
 0.34537529945373535 s/600itr
Step 2500 Train Loss=0.030780984
时间
 0.3463551998138428 s/600itr
Saving Model to file in/home/ma-user/modelarts/user-job-dir/code/logs/
Step 3000 Train Loss=0.032335356
时间
 0.3462545871734619 s/600itr
```

### 训练精度

精度指标采用训练生成图像与标签图像的像素交并比表示

|                     | background | empty | liquid | solid |
| ------------------- | ---------- | ----- | ------ | ----- |
| 论文精度            | 100%       | 82%   | 74%    | 42%   |
| GPU精度数据         | 99.7%      | 78.0% | 71.8%  | 44.8% |
| GPU性能数据         | 0.14s-0.15s/600  |       |        |       |
| NPU训练精度         | 99.7%      | 76.9% | 71.6%  | 43.3% |
| NPU性能数据         | 0.11s-0.12s/600  |       |        |       |
| 带数据增强的GPU精度 | 99.7%      | 81.4% | 77.0%  | 51.1% |

说明：因为训练集规模较小 数据量不足且分布不均匀 原论文需要对数据集进行大量数据增强操作，但是在NPU训练中由于不支持数据动态输入 因此去除对图片尺寸动态变换 随机裁剪的增强操作 导致一些类分割精度有所下降，在去除数据增强的GPU训练中，通过对三次测试取得平均精度，与NPU精度差距基本控制在1%左右。下表最后一栏 为保留数据增强情况下GPU训练精度 可以看出在保留数据增强的GPU训练中 网络可以获得更好的精度指标 。在训练过程中的LOSS不能对测试精度产生有效反映 因此不同训练批次测试结果会有细微精度差异。

 **性能优化** 
在更换tf.nn.dropout接口为npu_ops.dropout（）接口，替换文件BuildNetVgg16为BuildNetVgg161，并运行在裸机环境上时，修改原NPU测试代码中数据集路径为裸机上路径，性能达到要求，具体性能参数，详见NPU性能测试图片



