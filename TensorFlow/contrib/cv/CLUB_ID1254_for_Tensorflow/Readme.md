-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [Requirements](#requirements)
-   [数据集](#数据集)
-   [代码及路径解释](#代码及路径解释)
-   [Running the code](#running-the-code)
	- [run script](#run-command)
	- [Training Log](#training-log)
	
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）： Image Classification**

**版本（Version）：1.1**

**修改时间（Modified） ：2022.7.26**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：CV**

**描述（Description）：基于TensorFlow框架的CLUB域适应的图像分类网络训练代码** 

<h2 id="概述.md">概述</h2>

- Official Source Code https://github.com/Linear95/CLUB
- Paper http://proceedings.mlr.press/v119/cheng20b/cheng20b.pdf

## Requirements

- python 3.7.5
- tensorflow 1.15.0
- numpy
- scikit-learn
- opencv-python
- scipy

### 数据集

- SVHN

- MNIST

  ```
  SVHN链接：https://pan.baidu.com/s/1Zi-wIc2588gmG5FZ27mMCA 
  提取码：vmob
  MNIST链接：https://pan.baidu.com/s/1Dpfq--sqpqCm0qEtJALd-Q 
  提取码：5c9d
  ```

### 代码及路径解释

```
TF-CLUB
└─
  ├─README.md
  ├─MI_DA	模型代码
	├─imageloader.py      数据集加载脚本
	├─main_DANN.py        模型训练脚本
	├─MNISTModel_DANN.py  模型脚本
	├─utils.py 	          实体脚本文件
```

## Running the code

### Run command

#### GPU

```python
python main_DANN.py --data_path /path/to/data_folder/ --save_path /path/to/save_dir/ --source svhn --target mnist
```

#### ModelArts

```
框架：tensorflow1.15
NPU: 1*Ascend 910 CPU: 24vCPUs 96GB
镜像：ascend-share/5.1.rc1.alpha005_tensorflow-ascend910-cp37-euleros2.8-aarch64-training:1.15.0-21.0.2_0401
OBS Path:/cann-id1254/
Data Path in OBS:/cann-id1254/dataset/
Debugger:不勾选
```

### Training log

```python
svhn-mnist_DANN_0.1 iter 3780  mi_loss: 0.0198  d_loss: 0.0000  p_acc: 0.9688
```

#### 精度结果

- GPU（V100）结果

      Source (svhn) 
      Target (Mnist) 
      p_acc: 0.9688

- NPU结果

      Source (svhn) 
      Target (Mnist) 
      p_acc: 0.9688



