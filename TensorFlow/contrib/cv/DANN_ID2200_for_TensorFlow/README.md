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

**修改时间（Modified） ：2021.12.21**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Research**

**描述（Description）：基于TensorFlow框架的DANN图像分类网络训练代码** 

<h2 id="概述.md">概述</h2>

提出了一种新的领域自适应表示学习方法,在MNIST图像分类问题中证明了该方法的成功
- 参考论文：

    https://github.com/pumpikano/tf-dann

    https://jmlr.org/papers/volume17/15-239/15-239.pdf

## Requirements
- python 3.7
- tensorflow 1.15
- numpy
- scikit-image
- matplotlib
- scikit-learn
- jupyter
- scipy

- Ascend: 1*Ascend 910 CPU: 24vCPUs 96GiB
```
支持镜像：ascend-share/5.0.4.alpha002_tensorflow-ascend910-cp37-euleros2.8-aarch64-training:1.15.0-21.0.2_1217
```
### 数据集
 **MNIST-M 数据集** 
由 MNIST 数字与来自 BSDS500（Berkeley Segmentation Data Set and Benchmarks 500） 数据集的随机色块混合而成。 要生成 MNIST-M 数据集，首先下载 BSDS500 数据集并运行脚本：create_mnistm.py
```
curl -L -O http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz
python create_mnistm.py
```
BSDS500 下载：
```
https://byq.obs.cn-north-4.myhuaweicloud.com:443/output/MA-new-tf-dann-master_npu_20210927032055-12-21-16-43/code/BSR_bsds500.tgz?AccessKeyId=CHRWGFJ4FCZCTEMBJYKY&Expires=1671182296&Signature=OmYVvb3AO2ncXanP93ugp/j15yE%3D
```

然后生成一个 mnistm_data.pkl 的文件

mnistm_data.pkl 下载： 
```
https://byq.obs.cn-north-4.myhuaweicloud.com:443/output/MA-new-tf-dann-master_npu_20210927032055-12-21-16-43/code/BSR_bsds500.tgz?AccessKeyId=CHRWGFJ4FCZCTEMBJYKY&Expires=1671182270&Signature=LZS7SWYeEO7anauTtYHQBrFzTRs%3D
```
 **Mnist数据集** 
```
URL:
https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=vE+Yw14jSUMJw3E1h2xlmyKxDIw46CoKaC5AEGOBBUU/z96xfhiG/L1mGrElHpX84lg01kTGz8EpN60STDOVY8CGINEYhRBIcoM2/fKBT4w8tMzsIT7OHCd3jnYjIbBC2I0M7iXXBiVQQGtHGuBl1FJ84y+/qfy3vVlM37JTG5NO9qTiTElCDAYsEa5WgsegvaCB8WQUY/6XVzzExrEPITuKzRPuKw0yQdhQFhzMS+Xtw/J/YtOMJo/idPlHma6R2GS/+7BctP3Ib02vpyAazMqB1lP0Kpw4Y50n4QGf83HXbQtX6D3yxO+b6rvLeEWLqOXYkWMZSeRSl6gVwxk+hm7aP/BRBraApYr0JJlO0f/umUfhQmEKAUaE/PShxxwEVV367Naud3oM1dpC+yPjI1QsKIUgY0eQceX3z3ws5dxb8Ug3nNKBuuz6EeVG0t++xFCx8w9NCGQ4b/+mV4ekyHFx1jaYtgPpDFMVpiCFt6WFgkf96v2bL3E3TFeu7BnRz5Ym0v11gK4Y1vZ7vn0A191DByfiHUACnUXX7HPnOmV9IfF1s/XBBR5w25hvJLgrJZetHxZy6wNnwzJJ5NmklrCm6J099AXgo6zcyosGyM+PYCue7WA+owypIUJTioLcgcJrPIzHmf9Y2lWJYjQOkMalc6ncykChQMy0O22XiUaPQRxMtqVn8ztNw138dCH+CnqGbRpHIDSkmoE5RfTkn0VF1IHKuCGD6nR+cO9XC9Hl6JDnWKA9aNCSM1DMj/1AfFVhZIs0ec0itR5DgkZok8w+4XXnxPEZKNClAOCpEJE=

提取码:
123456

*有效期至: 2022/12/16 17:23:45 GMT+08:00
```
### 代码及路径解释
```
TF-DANN
└─
  ├─README.md
  ├─LICENSE 
  ├─mnistm_data.pkl    合成数据集 
  ├─MNIST_data	存放数据集文件夹
	├─t10k-images-idx3-ubyte.gz
	├─t10k-labels-idx1-ubyte.gz
	├─train-images-idx3-ubyte.gz
	├─train-labels-idx1-ubyte.gz	
  ├─MNIST-DANN.py     MNIST数据集实验
  ├─Blobs-DANN.py   简单数据集实验示例
  ├─test_1p.sh      代码运行脚本
  ├─create_mnistm.py     创建合成数据集MNIST-M
  ├─flip_gradient.py   梯度反转层
  ├─utils.py   定义模型结构
  ├─ckpt_npu    存放checkpoint文件夹
  ├─output    存放模型运行日志文件夹    
   
```
## Running the code
### Run command
#### Use bash
```
bash test_1p.sh
```
#### Run directly

```
python3 MNIST-DANN.py
```
### Training log
#### 训练性能分析
|  平台| 性能 |
|-----------|---------------|
| GPU (V100) | 2.5 ms / epoch |
| NPU (Ascend910)| 34.8 ms / epoch |
#### 精度结果
- GPU结果

        Source only training
        Source (MNIST) accuracy: 0.9869
        Target (MNIST-M) accuracy: 0.5305

        Domain adaptation training
        Source (MNIST) accuracy: 0.9768
        Target (MNIST-M) accuracy: 0.7421
        Domain accuracy: 0.681

- NPU结果

        Source only training
        Source (MNIST) accuracy: 0.9857999
        Target (MNIST-M) accuracy: 0.50689995

        Domain adaptation training
        Source (MNIST) accuracy: 0.98009986
        Target (MNIST-M) accuracy: 0.7273001
        Domain accuracy: 0.664
#### 模型保存
```
chenckpoint文件
链接: https://pan.baidu.com/s/1NU9Pc9drKqe4tyc56JF1gw 
提取码: rd4e 