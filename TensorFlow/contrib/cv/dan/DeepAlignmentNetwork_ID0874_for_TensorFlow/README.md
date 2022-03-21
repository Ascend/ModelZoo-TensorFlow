### 基本信息
- **发布者** （Publisher）：huawei
- **应用领域** （Application Domain）：cv
- **修改时间** （Modified） ：2021.9.1
- **框架** （Framework）：TensorFlow 1.15.0
- **模型格式** （Model Format）：ckpt
- **处理器** （Processor）：昇腾910
- **描述** （Description）：基于TensorFlow框架的Deep Alignment Network代码

### 概述

DAN是“Deep Alignment Network: A convolutional neural network for robust face alignment”中描述的面部对齐方法的参考实现
- [参考实现](http://github.com/MarekKowalski/DeepAlignmentNetwork)（[tensorflow实现](https://github.com/zjjMaiMai/Deep-Alignment-Network-A-convolutional-neural-network-for-robust-face-alignment)）
- [适配昇腾 AI 处理器的实现](https://gitee.com/ascend/modelzoo/tree/master/contrib/TensorFlow/Research/cv/dan/DeepAlignmentNetwork_ID0874_for_TensorFlow)

### 默认配置

 **基础环境** ：TF-1.15-python3.7


### 脚本参数

1. 预处理

```
python preprocessing.py --input_dir=... --output_dir=... --istrain=True --repeat=10 --img_size=112 --mirror_file=./Mirror68.txt
```


```
   --input_dir               #数据集路径
   --output_dir              #结果保存路径
   --istrain                 #是否训练
   --repeat                  #每张图片重复使用次数
   --img_size                #图片大小，default=112
   --mirror_file             #Mirror68.txt拷贝到主机上的路径
```

2. DAN_V2

```
python DAN_V2.py -ds 1 --data_dir=preprocess_output_dir --data_dir_test=...orNone -nlm 68 -te=15 -epe=1 -mode train
python DAN_V2.py -ds 2 --data_dir=preprocess_output_dir --data_dir_test=...orNone -nlm 68 -te=45 -epe=1 -mode train
```


```
   --data_dir                #训练集
   --data_dir_test           #测试集
   --model_dir               #模型及预测图片保存路径
   -nlm                      #--num_lmark，default=68
   -te                       #--train_epochs，default=20
   -epe                      #--epochs_per_eval，default=1
   -ds                       #--dan_stage，default=1
   -mode                     #运行类型，default='train'，choices=['train', 'eval', 'predict'] 
   							 #predict模型下必须设置--data_dir_test，且								
   							 #目录下必须有对应图片的.ptv文件。输出推理图片，精度指标。
   --batch_size              #批大小，default=64
```

### 混合精度训练

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。


### 训练环境准备

通过pip install requirements.txt下载相应包

### 数据集准备

1、用户自行准备好数据集。

2、数据集的处理可以参考  "[模型来源](https://github.com/zjjMaiMai/Deep-Alignment-Network-A-convolutional-neural-network-for-robust-face-alignment)"

- [Mirror68.txt镜像下载地址](http://pan.baidu.com/s/1Ln_i00DRulDlgHJ8CmIqAQ)
- **数据集** ：
  在[300W竞争数据集](http://ibug.doc.ic.ac.uk/resources/facial-point-annotations/)（含AFW、LFPW、HELEN、IBUG、300W私人测试集五个数据集）上实现了DAN的训练及测试。我们对数据集进行分类处理，分为训练集trainest和测试集testest。

 [整理后预处理数据集百度网盘地址](http://pan.baidu.com/s/1pXUxDsdj-7C0CxV7mhal3g)

提取码：81hi


### 模型训练


- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 启动训练之前，首先要配置程序运行相关环境变量。

  [环境变量配置信息参见Ascend 910训练平台环境变量](http://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)


### 性能&精度

| 处理器  | 步数  |  loss  | 平均性能(步/每秒) |
|--------|-------|--------|------------------|
|   gpu  | * |  * |       9.5       |
|   npu  | * | * |       5.7       |
