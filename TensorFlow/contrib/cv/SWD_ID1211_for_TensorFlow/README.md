
- [概述](#概述)
	- [复现SWD论文](#复现swd论文)
- [Requirements](#requirements)
- [数据集](#数据集)
- [代码及路径解释](#代码及路径解释)
- [Running the code](#running-the-code)
	- [Toy](#toy)
	- [Training Log](#training-log)
	- [train model in Domain adpatation with svhn 2 mnist](#train-model-in-domain-adpatation-with-svhn-2-mnist)
# 概述

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：** CV 

**版本（Version）：1.2**

**修改时间（Modified） ：2020.10.28**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Research**

## 复现SWD论文
- [IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019.](http://cvpr2019.thecvf.com)
- [[Paper]](https://arxiv.org/abs/1903.04064)
- [apple/ml-cvpr2019-swd: Sliced Wasserstein Discrepancy for Unsupervised Domain Adaptation. In CVPR 2019.](https://github.com/apple/ml-cvpr2019-swd)

# Requirements
- Python 3.7.5.
- Tensorflow 1.15.0
- Huawei Ascend

# 数据集
- moon_data.npz：二维平面上，月牙形数据的简单二分类数据集。
- SVHN：The Street View House Numbers (SVHN) Dataset
- MNIST：MNIST handwritten digit database

# 代码及路径解释
```
.
├── CONTRIBUTING.md
├── datasets # 存放数据及数据读取python文件
│   ├── mnist 
│   ├── svhn
│   ├── LoadMNIST.py
│   └── LoadSVHN.py
├── LICENSE
├── moon_data.npz # 存放swd.py使用的demo数据
├── README.md
├── requirements.txt
├── svhn2mnist_test_npu.py #测试文件
├── svhn2mnist_train_npu.py #训练文件
├── swd.py
└── train_1p.sh #模型的启动脚本
```

# Running the code
## Toy 
- Use bash
```
bash train_1p.sh
```
- OR

To run the demo with adaptation:
```
python swd.py -mode adapt_swd
```

To run the demo without adaptation:
```
python swd.py -mode source_only
```

## Training Log

### 训练性能分析

|           | SWD   |
|-----------|---------------|
| GPU (V100) | 4 ms / epoch |
| NPU | 35.7 ms / epoch |
| NPU (开启混合精度)| 2.5 ms / epoch |

说明：从上表可以看出，由于网络模型规模较小，CPU上的运行速度反而是最快的，而NPU的运行速度与GPU相近。

### 精度达标
NPU上训练后的loss_s、loss_dis、source acc、target acc等指标均达到或超过了GPU上的水平。

- GPU
```
-> Perform training with domain adaptation.
2021-11-04 16:52:52.443193: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cublas64_100.dll
Iteration: 0 / 10000
loss_s: 1.40980, loss_dis:0.00108
source acc:0.28, target acc:0.32
Iteration: 1000 / 10000
loss_s: 0.68465, loss_dis:0.00450
source acc:0.87, target acc:0.90
Iteration: 2000 / 10000
loss_s: 0.19743, loss_dis:0.00016
source acc:0.96, target acc:0.91
Iteration: 3000 / 10000
loss_s: 0.01766, loss_dis:0.00038
source acc:1.00, target acc:0.96
Iteration: 4000 / 10000
loss_s: 0.00839, loss_dis:0.00287
source acc:1.00, target acc:0.98
Iteration: 5000 / 10000
loss_s: 0.00607, loss_dis:0.00244
source acc:1.00, target acc:0.99
Iteration: 6000 / 10000
loss_s: 0.00352, loss_dis:0.00047
source acc:1.00, target acc:1.00
Iteration: 7000 / 10000
loss_s: 0.00226, loss_dis:0.00026
source acc:1.00, target acc:1.00
Iteration: 8000 / 10000
loss_s: 0.00168, loss_dis:0.00017
source acc:1.00, target acc:1.00
Iteration: 9000 / 10000
loss_s: 0.00128, loss_dis:0.00011
source acc:1.00, target acc:1.00
Iteration: 10000 / 10000
loss_s: 0.00101, loss_dis:0.00009
source acc:1.00, target acc:1.00
[Finished]
-> Please see the current folder for outputs.
```
- NPU
```
-> Perform training with domain adaptation.
Iteration: 0 / 10000
loss_s: 1.40980, loss_dis:0.00108
source acc:0.28, target acc:0.32
Iteration: 1000 / 10000
loss_s: 0.68452, loss_dis:0.00450
source acc:0.87, target acc:0.90
Iteration: 2000 / 10000
loss_s: 0.19736, loss_dis:0.00016
source acc:0.96, target acc:0.91
Iteration: 3000 / 10000
loss_s: 0.01766, loss_dis:0.00038
source acc:1.00, target acc:0.96
Iteration: 4000 / 10000
loss_s: 0.00840, loss_dis:0.00287
source acc:1.00, target acc:0.98
Iteration: 5000 / 10000
loss_s: 0.00608, loss_dis:0.00247
source acc:1.00, target acc:0.99
Iteration: 6000 / 10000
loss_s: 0.00353, loss_dis:0.00047
source acc:1.00, target acc:1.00
Iteration: 7000 / 10000
loss_s: 0.00226, loss_dis:0.00026
source acc:1.00, target acc:1.00
Iteration: 8000 / 10000
loss_s: 0.00168, loss_dis:0.00017
source acc:1.00, target acc:1.00
Iteration: 9000 / 10000
loss_s: 0.00128, loss_dis:0.00011
source acc:1.00, target acc:1.00
Iteration: 10000 / 10000
loss_s: 0.00101, loss_dis:0.00008
source acc:1.00, target acc:1.00
[Finished]
-> Please see the current folder for outputs.

```
## train model in Domain adpatation with svhn 2 mnist
To Train
```
python svhn2mnist_train_npu.py -mode adapt_swd
```
To Test
```
python svhn2mnist_test_npu.py -mode adapt_swd
```
