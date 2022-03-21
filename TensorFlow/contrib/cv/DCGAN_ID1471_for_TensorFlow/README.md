## DCGAN

### ModelArts 参数
dataset=faces;input_height=64;input_width=64;train=true;crop=true;input_fname_pattern=*.jpg

### 概述
迁移dcgan到ascend910平台上使用NPU运行，并将结果与原论文进行对比

原论文：[Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)

代码参考：[DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow)

### Requirements
* Python 3.6 
* Tensorflow 1.15
* SciPy
* pillow
* tqdm

### 硬件
* Ascend: 1*Ascend 910 CPU: 24vCPUs 96GiB

### 代码及路径解释
```
DCGAN
└─
  ├─README.md
  ├─LICENSE  
  ├─download.py       获取数据集，在NPU上不适用
  ├─main.py           执行主函数代码
  ├─model.py          定义模型结构
  ├─ops.py            自定义算子
  ├─utils.py          工具函数
  ├─classifier_svm.py 推理文件
  
  
```
### 数据集和预训练模型：

数据集：ImageNet2012


### 训练过程及结果



### 执行训练

```shell
python main.py --dataset imagenet --input_height=108 --crop
```


主要参数注释：
```
dataset: 数据集名称

data_dir: 数据集目录

train: 是否训练

input_height: 输入尺寸

output_height: 

```


### 推理部分
参照原论文，将 DCGAN 的鉴别网络参数存储，数据集 Cifar-10 利用鉴别网络提取特征保存到本地，对保存的特征文件进行 SVM 分类，最终得到分类的精确度为本论文的精度指标。

```shell
python classifier_svm.py --dataset materials --data_dir=./data/cifar10/ --input_height=64 --crop --train_svm True
```

classifier_svm.py说明

利用参数train_svm控制两个过程

train_svm为True，加载鉴别网络(net_D)，将cifar-10数据集输入网络得到特征，存入文件中(fname = 'cifar10_svm')

train_svm为False，载入特征文件，训练SVM(157-160行注释第一次训练时放开完成训练，得到SVM参数)
加载SVM参数进行predict，最后得到分类结果即为精度

![alt](./images/SVM.png)

### 模型精度及性能
* 训练精度

**GAN训练的最好情况为对抗网络和生成网络趋向0.5**

无训练精度指标

* 训练性能

| 平台                  | 性能       |
|---------------------|----------|
| 论文                  | 无        |
| GPU(v100 8vCPUs) | 4s/epoch |
| NPU                 | 3s/epoch |

* 推理精度

|  平台   | 性能    |
|  ----  |-------|
| 论文  | 82.8% |
| GPU(v100 8vCPUs)  | 78.4% |
| NPU  | 85.3% |2

### 结果预览
数据集 Imagenet 25 epoch
* 论文

![alt](./images/paper_iamgenet.png)

* GPU (v100 8vCPUs)


![alt](./images/gpu_imagenet.png)

* NPU

![alt](./images/npu_imagenet.png)

数据集 Celeba
* NPU

![alt](./images/npu_celeba.png)