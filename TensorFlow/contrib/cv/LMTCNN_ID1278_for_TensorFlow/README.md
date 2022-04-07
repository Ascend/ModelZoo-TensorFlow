# agegenderLMTCNN

## 概述

agegenderLMTCNN是一种同时预测年龄和性别的网络。

- 参考论文 [Joint Estimation of Age and Gender from Unconstrained Face Images using Lightweight Multi-task CNN for Mobile Applications](https://arxiv.org/abs/1806.02023)。

- 参考项目https://github.com/ivclab/agegenderLMTCNN

## 默认配置

- 数据图片resize为227*227
- 训练超参：
  - image_size：227
  - batch_size：32
  - max_steps：50000
  - steps_per_decay：10000
  - lr:0.01
  - eta_decay_rate:0.1

## 支持特性

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 否       |
| 混合精度   | 否       |
| 数据并行   | 否       |

## 文件目录

data.py:读取训练数据

datapreparation.py:将原始数据拆分为训练集、验证集和测试集，以进行五折交叉验证。此项目已在 DataPreparation/FiveFolds/train_val_test_per_fold_agegender 中生成此 txt 文件。

download_adiencedb.py:下载Adience 数据集。

download_model.py:下载训练好的模型。

eval.py:评估 LMTCNN 模型。

model.py:定义网络。

- multipreproc.py:预处理原始数据，在tfrecord目录下生成训练集、验证集和测试集的tfrecord文件。


train.py：训练模型文件

util.py

## 训练环境准备

- Python 3.0及以上
- Numpy
- OpenCV
- [TensorFlow](https://www.tensorflow.org/install/install_linux) 1.15
- 昇腾NPU环境

## 快速上手

### 数据集准备

从[谷歌云端](https://docs.google.com/uc?export=download&amp;id=11Zv__6WvbjtovcQzZOjELdscmyx63Gov )下载对齐后的Adience数据集，或者从Adience官网下载数据集并自行对齐。运行multipreproc.py文件生成train, test,valid的tfrecord文件。生成的数据集按照五等分交叉验证分别存储在test_fold_is_0到test_fold_is_4文件夹下。

### 模型训练

参考下方的训练过程。参数可自行定义。

预训练模型下载链接：https://pan.baidu.com/s/1JniERocb6wBcOG23qiRbYA 
提取码：e722

## 高级参考

###  脚本和示例代码

```
|-- LICENSE
|-- README.md
|-- data.py                数据读入
|-- datapreparation.py     拆分为训练集、验证集和测试集
|-- download_adiencedb.py  下载adience数据库
|-- download_model.py      下载预训练好的模型
|-- eval.py                测试模型
|-- model.py               网络定义文件
|-- multipreproc.py        生成tfrecord
|-- modelzoo_level.txt 
|-- requirements.txt 
|-- script                 运行脚本                
|   |-- evalfold1.sh
|   |-- evalfold2.sh
|   |-- evalfold3.sh
|   |-- evalfold4.sh
|   |-- evalfold5.sh
|   |-- trainfold1.sh
|   |-- trainfold2.sh
|   |-- trainfold3.sh
|   |-- trainfold4.sh
|   `-- trainfold5.sh
|-- train.py                训练模型
`-- utils.py

```

### 脚本参数

```
    --model_type                LMTCNN-1-1/LMTCNN-2-1
    --pre_checkpoint_path       restore this pretrained model before beginning any training
    --data_dir                	the root path of tfrecords
    --model_dir             	the root of store models  
    --batch_size                batch size for training
    --image_size                227
    --eta                       Learning rate
    --pdrop       				Dropout probability
    --max_steps            		Number of iterations       
    --steps_per_decay           step for starting decay learning_rate
    --eta_decay_rate            learning rate decay
    --epochs             		Number of epochs
```

### 训练过程

```bash
# 根据数据集路径修改train_dir的值。根据数据集的不同划分分别运行trainfold1.sh到trainfold5.sh
$ ./script/trainfold1.sh ~ $ ./script/trainfold5t.sh ：
```

### 验证过程

```bash
# 根据数据集路径修改train_dir的值。根据数据集的不同划分分别运行trainfold1.sh到trainfold5.sh
$ ./script/evalfold1.sh ~ $ ./script/evalfold5.sh 
```

## 训练精度

五等分交叉验证的GPU与NPU运行结果如下：

| GPU  | Age（Top-1）（Acc） | Age（Top-2）（Acc） | Gender（Acc） |
| ---- | ------------------- | ------------------- | ------------- |
| 0    | 49.06               | 73.42               | 83.63         |
| 1    | 37.08               | 60.61               | 80.89         |
| 2    | 44.81               | 70.06               | 79.34         |
| 3    | 40.73               | 65.35               | 80.74         |
| 4    | 38.92               | 64.18               | 77.60         |
| Ave  | 42.12               | 66.72               | 80.44         |



| NPU  | Age（Top-1）（Acc） | Age（Top-2）（Acc） | Gender（Acc） |
| ---- | ------------------- | ------------------- | ------------- |
| 0    | 45.11               | 68.75               | 80.95         |
| 1    | 36.43               | 58.50               | 78.52         |
| 2    | 41.09               | 66.37               | 77.25         |
| 3    | 41.68               | 64.37               | 80.64         |
| 4    | 40.10               | 65.25               | 79.25         |
| Ave  | 40.88               | 64.65               | 79.32         |



| Ave  | Age（Top-1）（Acc） | Age（Top-2）（Acc） | Gender（Acc） |
| ---- | ------------------- | ------------------- | ------------- |
| 论文 | 40.84               | 66.10               | 82.04         |
| GPU  | 42.12               | 66.72               | 80.44         |
| NPU  | 40.88               | 64.65               | 79.32         |



## 训练性能对比

经过1000个batch训练后的平均性能分别如下：

| GPU NVIDIA V100    | NPU            |
| ------------- | -------------- |
| 0.084 s/batch | 0.04 s/batch |

