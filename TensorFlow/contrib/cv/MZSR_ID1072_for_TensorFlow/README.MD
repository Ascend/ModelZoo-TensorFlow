## Introduction

MZSR[Meta-Transfer Learning for Zero-Shot Super-Resolution]模型,将元学习引入到Zero-Shot Super-Resolution 领域，使得训练后的模型可以在经过1次迭代就能适应特定的图片，且取得显著效果

    对于更详细的结果，可以参考[项目主页](https://github.com/JWSoh/MZSR)

## Dataset

**Training Data**

Download training dataset [DIV2K]
#### Generate TFRecord dataset
- Refer to [MainSR](https://www.github.com/JWSoh/MainSR) repo.
- Run generate_TFRecord_MZSR.py

**Test Input Data**

Degraded Images of Set5, B100, Urban100 on various kernel environments.


## NPU-Train

- 使用与GPU训练相同的数据集，请提前通过提供的链接下载 `dataset`,在作者仓库（https://github.com/JWSoh/MZSR）下载Pretain文件以及测试时用的图片和mat文件。

- 模型修改

  通过使用npu自动迁移工具进行模型的迁移，详细过程请参考[链接](https://support.huaweicloud.com/tfmigr-cann503alpha1training/atlasmprtgtool_13_0006.html)

- 配置启动文件`boot_modelarts.py`,启动训练时，需设置好`train_url` 和 `data_url` 两个路径，详情请参考[链接](https://support.huaweicloud.com/tfmigr-cann503alpha1training/atlasmprtgma_13_0004.html) 。

## Result
- 训练性能

| |DIV2K |
|---|---|
GPU | 100个step需130s |
NPU | 100个step需88S | 

说明：表中时间为训练所需的平均时间，可以看出，NPU上的性能远远超过GPU。

- 训练精度(psnr)

| |Set5(g20) | 
|---|---|
paper | 35.2 |
GPU | 34.8 |  
