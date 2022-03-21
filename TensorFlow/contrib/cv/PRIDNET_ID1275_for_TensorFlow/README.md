### PRIDNET网络概述 
PRIDNET网络是在论文《Pyramid Real Image Denoising Network》中提出的。论文进行了系统的研究，解决了深度卷积神经网络 (CNN)在现实世界的噪点图像处理不佳的问题。PRIDNet的主要贡献包括以下三方面：(1)Channel Attention：在提取的噪声特征上利用通道注意机制，自适应地校正通道重要性，解决大多数基于CNN的去噪方法中所有通道的特征被平等对待的问题。(2)Multi-scale feature extraction：设计了一种金字塔去噪结构，其中每个分支都关注一个尺度的特征。利用它可以同时提取全局信息并保留局部细节，从而为后续的全面去噪做好准备。多尺寸特征提取解决了固定感受野无法携带多样信息的问题。(3)Feature self-adaptive fusion：级联的多尺度特征，每个通道代表一个尺度的特征，引入了核选择模块。采用线性组合的方法对不同卷积核大小的多分支进行融合，使得不同尺度的特征图可以通过不同的核来表达。特征自适应融合，解决了大多数方法对不同尺寸的特征进行不加区分的处理，不能自适应的表达多尺度特征的问题。

- 参考论文
  
    [Pyramid Real Image Denoising Network](https://ieeexplore.ieee.org/document/8965754)

- 参考实现
    
    [https://github.com/491506870/PRIDNet](https://github.com/491506870/PRIDNet)

### 默认配置

- 数据集：SIDD-Medium Dataset  Raw-RGB images only (~20 GB)，下载地址为：[https://www.eecs.yorku.ca/~kamel/sidd/dataset.php](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php)
- 验证数据集：SIDD Validation Noisy Raw-RGB data,SIDD Validation Ground-truth raw-RGB data（获取路径：obs://pridnetdata/test_dataset/）
```shell
数据集组织
├── dataset						----数据集文件
    ├── SIDD_Medium_Raw_Parts				---训练数据集
    │   ├── SIDD_Medium_Raw				
    │
    ├── validation_dataset		----验证数据集
    │   ├── ValidationNoisyBlocksRaw.mat	
    │   ├── ValidationGtBlocksRaw.mat       
```
- 训练超参

  - learning_rate = 1e-4
  - epoch = 4001
  - Optimizer = AdamOptimizer  


###  训练环境

1. 系统：
    - linux ubuntu   
2. 软件：

    - h5py
    - PIL
    - tensorflow 1.15
    - python 3.7
    - numpy
    


### 代码及路径解释

```
PRIDNET
└─ 
  ├─README.md
  ├─network.py PRIDNET 网络模型
  ├─boot_modelarts.py 将obs数据拷贝到modelarts，执行训练启动脚本
  ├─help_modelarts.py 将modelarts上的训练结果拷贝到obs上
  ├─train_SIDD_Pyramid.py 模型训练代码
  ├─run_modelarts.sh 启动脚本
  ├─test_SIDD_Pyramid.py 模型测试
  ├── test.py 计算SSIM和PSNR

```


### 训练过程及结果

- 启动训练

    1）执行训练启动脚本:
    ```
    sh run_modelarts.sh
    ```
    2）在npu服务器上的部分训练日志如下：
    ```
    3829 25 Loss=5.0449 Time=0.042
    3829 26 Loss=5.0378 Time=0.042
    3829 27 Loss=5.0312 Time=0.042
    3829 28 Loss=5.0230 Time=0.042
    3829 29 Loss=5.0232 Time=0.042
    3829 30 Loss=5.0161 Time=0.042
    3829 31 Loss=5.0246 Time=0.043
    3829 32 Loss=5.0248 Time=0.044
    3829 33 Loss=5.0251 Time=0.042
    3829 34 Loss=1.9025 Time=0.044
    3829 35 Loss=1.9094 Time=0.042
    3829 36 Loss=1.9093 Time=0.044
    3829 37 Loss=1.9015 Time=0.042
    3829 38 Loss=1.9017 Time=0.049
    3829 39 Loss=1.9016 Time=0.042
    3829 40 Loss=1.9018 Time=0.044
    3829 41 Loss=1.9089 Time=0.042
    3829 42 Loss=1.9090 Time=0.043
    3829 43 Loss=1.9088 Time=0.042
    3829 44 Loss=1.9161 Time=0.049
    3829 45 Loss=1.9162 Time=0.042
    3829 46 Loss=1.9161 Time=0.042
    3829 47 Loss=1.9088 Time=0.043
    ```

- PRIDNET网络模型在GPU和NPU上的性能对比

    |       训练环境        |        性能     |
    | -------------------- | ----------------|
    | GPU                  | 0.103 s / step   | 
    | NPU                  | 0.045 s / step   | 

- PRIDNET网络模型在GPU和NPU上的PSNR和SSIM指标对比
<table>
    <tr>
       <td>metrics</td>
       <td colspan="2" align="center">PSNR</td>
       <td colspan="2" align="center">SSIM</td>
    </tr>
    <tr>
      <td>chip</td>
      <td>gpu</td>
      <td>npu</td>
      <td>gpu</td>
      <td>npu</td>
    </tr>
    <tr>
      <td>PRIDNET</td>
      <td>45.0308</td>
      <td>45.0205</td>
      <td>0.9985</td>
      <td>0.9985</td>
    </tr>
</table>