# MVSNet

## 介绍
[武汉大学][高校贡献][TensorFlow]MVSNet网络

## 基本信息
**发布者（Publisher）：Huawei**

**应用领域（Application Domain）： Image Classification**

**版本（Version）：1.2**

**修改时间（Modified） ：2021.11.12**

**大小（Size）：74M**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的MVSNet多视图立体匹配网络训练代码**


## 概述
MVSNet将一张参考图像和多张源图像作为输入，为参考图像预测深度图。网络的关键之处在于可微分的单应变换操作，在从2D特征图构建3D代价体的过程中，网络将相机参数隐式地将编码入网络。为使网络能够适应任意数目的输入视图数，提出基于方差的指标，该指标将多个特征体映射为一个代价体。对代价体进行多尺度的3D卷积可以回归出一个初始的深度图。最后使用参考图像对深度图进行优化以提升边界区域的精度。
- 参考论文：

    [MVSNet: Depth Inference for
Unstructured Multi-view Stereo](https://openaccess.thecvf.com/content_ECCV_2018/papers/Yao_Yao_MVSNet_Depth_Inference_ECCV_2018_paper.pdf) 
- 参考实现：
- 适配昇腾AI处理器的实现：
https://gitee.com/tolovewang/modelzoo/upload/master/contrib/TensorFlow/Research/cv/MVSNet_ID2116_for_TensorFlow
- 通过Git获取对应commit\_id的代码方法如下：
    
    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    ```
##默认配置
- 输入参数：

  - view_num: 3
  - max_w: 640
  - max_h: 512
  - max_d: 192
  - sample_scale: 0.25
- 训练超参：

  - num_gpus: 1
  - batch_size: 1
  - epoch: 6
  - base_lr: 0.001
  - display: 1
  - stepvalue: 10000
  - snapshot: 5000
  - gamma: 0.9

## 支持特性

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 是    |
| 混合精度  | 是    |
| 并行数据  | 是    |

##快速上手
- 数据集准备
1. 模型训练数据集放在obs://mvsnet-bucket/mvsnet-data/data/中，请用户自行获取。


2. 数据集下载后请在训练脚本中指定数据集路径，可正常使用。

##模型训练
- 数据集准备

1. GPU复现版本的代码放置在obs://mvsnet-bucket/MA-MVSNet-master-09-02-22-23/code/中，请用户自行获取。
   

2. 代码的依赖在requirement.txt，请根据文档安装相关依赖。

3.配置ModelArts相关参数:

Boot File Path: （代码放置在本地的路径）\mvsnet_npu_for_TensorFlow\GPU训练脚本\mvsnet\train.py

Code Directory: （代码放置在本地的路径）\mvsnet_npu_for_TensorFlow\GPU训练脚本

OBS Path: /mvsnet-bucket/

Data Path in OBS: /mvsnet-bucket/mvsnet-data/

Running Parameters: regularization=3DCNNs;train_blendedmvs=true;max_w=768;max_h=576;max_d=128

##迁移学习指导
- 数据集准备

1. 获取训练数据，数据集在obs://mvsnet-bucket/mvsnet-data/data/中，请用户自行获取。

2. 数据集下载后请在训练脚本中指定数据集路径，可正常使用。

- 模型训练

1. NPU迁移版本的代码放置在obs://mvsnet-bucket/MA-new-mvsnet_npu_for_TensorFlow-11-08-16-09/code/中，请下载后使用。

2. 数据集下载后请在训练脚本中指定数据集路径，可正常使用。

3.配置ModelArts相关参数参照上面GPU版本;

另外设置Image Path(optional): ascend-share/5.0.3.alpha005_tensorflow-ascend910-cp37-euleros2.8-aarch64-training:1.15.0-21.0.2_1101