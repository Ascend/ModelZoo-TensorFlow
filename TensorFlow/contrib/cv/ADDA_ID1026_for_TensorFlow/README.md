### 概述

##### ADDA网络简述

ADDA模型是在《Adversarial Discriminative Domain Adaptation》（2017 CVPR）中提出来的，其中作者结合了discriminative模型，untie weight sharing以及GAN 损失。利用源域的标签学习具有判别性的representation，之后通过域对抗损失和不对称映射将目标域的数据映射到相同的空间中，最终使得目标域的数据得到了良好的分类效果。

- 参考论文

  Adversarial Discriminative Domain Adaptation： https://arxiv.org/pdf/1702.05464v1.pdf 

- 参考实现

  adda ：https://github.com/erictzeng/adda

- npu训练权重链接

  链接：https://pan.baidu.com/s/1ZTZrDJaPGAXp0QwSZ7UHgA 
  提取码：b0mt

### npu实现

##### 支持特性

支持混合精度训练，脚本中默认开启了混合精度，参考示例，见“开启混合精度”。

##### 混合精度训练

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

##### 开启混合精度

```
custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
```



### 准备工作

##### 训练环境的准备

硬件环境：Ascend: 1*Ascend 910(32GB) | ARM: 24 核 96GB

运行环境：ascend-share/5.0.3.alpha005_tensorflow-ascend910-cp37-euleros2.8-aarch64-training:1.15.0-21.0.2_1101

```
├── npu_train.sh                            //单卡运行脚本
├── modelarts_entry.py                      //pycharm在modelArts训练的定义脚本
├── README.md                               //代码说明文档
├── adda
│    ├── data                               //数据处理脚本存放
│    │    ├──__init__.py 
│    │    ├──dataset.py                     //数据加载
│    │    ├──mnist.py                       //mnist数据集处理
│    │    ├──svhn.py                        //svhn数据集处理
│    │    ├──usps.py                        //usps数据集处理
│    │    ├──util.py                        //下载数据集的工具代码
│    ├── models                             
│    │    ├──__init__.py
│    │    ├──lenet.py                       //训练模型构建
│    │    ├──model.py                       //数据处理
│    ├──__init__.py                       
│    ├──adversary.py                        //判别器模型定义
│    ├──util.py                             //工具代码（权重选择加载）
│    ├──logging.yaml                        //打印logging信息配置
├── scripts
│    ├──svhn-mnist.sh                       //npu训练脚本
│    ├──usps-mnist.sh                       //npu训练脚本
├── tools
│    ├──eval_classification.py              //推理
│    ├──train.py                            //源域训练
│    ├──train_adda.y                        //adda模型训练
```

### 训练

#### 环境依赖

制作数据集的环境上已安装Python3.7和TensorFlow 1.15.0。

#### 操作步骤

1. 数据集准备。

   a.请用户自行准备好数据集，包含训练集和验证集两部分，数据集包括Mnist、usps、svhn等，包含train和 	val两部分。以Mnist数据集为例。

   b.上传数据压缩包到训练环境上,无需解压

   ```
   ├── tools/data/mnist
   │   ├──t10k-images-idx3-ubyte.gz
   │   ├──t10k-labels-idx1-ubyte.gz
   │   ├──train-images-idx3-ubyte.gz
   │   ├──train-labels-idx1-ubyte.gz
   ```

2. 模型训练。

   运行脚本如下：

   ```
   svhn-mnist.sh
   usps-mnist.sh
   ```

3. 使用pycharm在ModelArts训练启动文件为：

   ```
   modelarts_entry.py 
   ```

4. 脚本参数如下：

   ```
   --dataset                            \\数据集
   --split                              \\数据集的train or eval
   --model                              \\ lenet模型
   --output                              \\ 输出文件名
   --iterations                          \\ step数
   --batch_size                          \\batch_size数
   --display                             \\打屏日志
   --lr                                   \\学习率
   --snapshot                             \\ 开始权重保存
   --solver                               \\ 优化器
   --seed                                 \\随机种子
   ```

## 训练结果

论文指标：

| 迁移任务 | SVHN--MNIST    | MNIST--USPS    | USPS--MNIST    |
| :------- | -------------- | -------------- | -------------- |
| NPU精度  | 0.766          | 0.894          | 0.912          |
| 论文精度 | 0.760          | 0.894          | 0.901          |
| GPU性能  | 7.3s/(200step) | 7.3s/(200step) | 7.3s/(200step) |
| NPU性能  | 0.5s/(200step) | 0.5s/(200step) | 0.5s/(200step) |

