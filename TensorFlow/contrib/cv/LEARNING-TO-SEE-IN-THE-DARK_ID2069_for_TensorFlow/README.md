# 基本信息
**发布者（Publisher）**：Huawei

**应用领域（Application Domain）**： Computer Version

**版本（Version）**：1.1

**修改时间（Modified）**：2021.12.25

**大小（Size）**：390kb

**框架（Framework）**：TensorFlow_1.15.0

**模型格式（Model Format）**：ckpt

**精度（Precision）**：Mixed

**处理器（Processor）**：昇腾910

**应用级别（Categories）**：Research

**描述（Description）**：基于TensorFlow框架的极端暗条件下的图片处理训练代码

# 描述
本模型应用于极端暗光条件下成像的问题，将传统图像后处理链路替换为端到端的全卷积网络，以一个raw数据作为输入，输出一个sRGB的成片。本模型主要解决了噪声和偏色问题。此外，还建立了SID（See-in-the-Dark）图像库，以短曝光图像与其对应的长曝光参考图来训练全卷积神经网络。
* 参考论文：  
[Chen Chen, Qifeng Chen, Jia Xu, and Vladlen Koltun, "Learning to See in the Dark", in CVPR, 2018.](http://cchen156.github.io/paper/18CVPR_SID.pdf)
* 参考实现：
https://github.com/cchen156/Learning-to-See-in-the-Dark
* 适配昇腾 AI 处理器的实现：
 https://gitee.com/lu-shi-hang/modelzoo/tree/master/contrib/TensorFlow/Research/cv/LEARNING-TO-SEE-IN-THE-DARK_ID2069_for_TensorFlow
# 默认配置
* 网络结构
   * U-net
* 训练超参(单卡)：
   * depth: 512
   * width: 512
   * epochs: 4000
   * lr: 前2000epoch: 0.0001 后2000epoch: 0.00001
# 支持特性

| 特性列表   |  是否支持  | 
| --------   | -----:   | 
| 分布式训练 | 是      |  
| 混合精度   | 是 |  
| 数据并行   | 否|   
   
# 混合精度训练
混合精度训练昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。
# 开启混合精度
```
global_config = tf.ConfigProto(log_device_placement=False)  
custom_op = global_config.graph_options.rewrite_options.custom_optimizers.add()  
custom_op.name = "NpuOptimizer"  
custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")  
```
# 训练环境准备
1. NPU环境  
硬件环境：
```
NPU: 1*Ascend 910   
CPU: 24*vCPUs 96GB  
```
运行环境： 
```
ascend-share/5.0.4.alpha002_tensorflow-ascend910-cp37-euleros2.8-aarch64-training:1.15.0-21.0.2_1125
```
2. 第三方依赖 
```
rawpy  
scipy==1.2.1
```
# 快速上手
## 数据集准备
用户需自行下载SID数据集，已上传至obs中，obs路径如下：obs://sid-obs/ModelArts_SID/dataset。  
训练集的训练目录Sony_train_list.txt和测试集的测试目录Sony_test_list.txt已经给出，文件中每行写出了一张图片短曝光图片的路径以及其对应的长曝光图片的路径。  
数据集中每张图片的命名包含信息：
第一个数字表示对应的数据集("0"属于训练集，"1"属于测试集)，第2到第5个数字表示图片ID。


## 模型训练
* 单击“立即下载”，并选择合适的下载方式下载源码包。
* 开始训练
   *  启动训练之前，首先要配置程序运行相关环境变量。  环境变量配置信息参见：  
[Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)
* 验证。
    * 自行编译PSNR以及SSIM计算代码。 
## 迁移学习指导
* 数据集准备。
    * 获取数据。请参见“快速上手”中的数据集准备。
* 模型训练。  
  参考“模型训练”中训练步骤。
* 模型评估。  
  参考“模型训练”中验证步骤。
# 训练过程及结果
1. 执行train_Sony.py文件，开始训练所有图片名第一个数字为"0"的短曝光图片。训练过程中在屏幕中打印每个epoch的loss和训练时间，通过观察发现在1000 epoch时，loss收敛，停止训练。
```
python3.7 train_Sony.py --epochs=1001
```
2. 将训练得到的result_Sony中的checkpoint文件放入checkpoint文件夹。
3. 执行test_Sony.py文件，用训练得到的checkpoint得到测试集的输出结果，测试所有图片名第一个数字为"1"的短曝光图片。测试结果为短曝光图片网络训练后的png格式图片，以及其对应的真值长曝光png格式图片。
```
python3.7 test_Sony.py
```
4. 执行eval.py文件，对测试结果进行精度计算。计算训练结果图片与其真值图片的PSNR以及SSIM。
```
python3.7 eval.py
```
在GPU复现中，由于自行编写的脚本与原论文中不同，评估结果也有一定差异。
以下为复现者自行编写后的评估结果：

 |         | PSNR    |  SSIM  | 性能 | 
 | --------   | -----:   |  -----:   | :----: |
 | 原论文     | 28.88    | 0.78|     | 
 | GPU复现    | 27.63 |   0.719|  0.055 s/step | 
 | NPU复现    | 28.26 |   0.715| 0.052 s/step |
