# c3ae_for_Tensoflow


## 基本信息

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Computer Vision**

**版本（Version）：1.0**

**修改时间（Modified） ：2022.1**

**大小（Size）：800KB**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Research**

**描述（Description）：根据人脸图像进行年龄估计**

## 概述

c3ae是一个简单但高效、基于上下文信息的级联性年龄估计模型，通过两点表征重新定义年龄估计问题，并将该问题由级联模型实现，此外，为了充分利用面部上下文信息，提出了多分支CNN网络来聚合多尺度上下文，相比于大模型其表现也具有较强的竞争力。

- 参考论文

  > https://arxiv.org/abs/1904.05059

- 参考开源实现

  > https://github.com/StevenBanama/C3AE

## 默认配置

- 训练超参
  - Batch size: 256
  - Train epoch: 600
  - Train step: 100

## 训练环境准备

1. 硬件环境准备请参见各硬件产品文档"[驱动和固件安装升级指南](https://gitee.com/link?target=https%3A%2F%2Fsupport.huawei.com%2Fenterprise%2Fzh%2Fcategory%2Fai-computing-platform-pid-1557196528909)"。需要在硬件设备上安装与CANN版本配套的固件与驱动。

2. 宿主机上需要安装Docker并登录[Ascend Hub中心](https://gitee.com/link?target=https%3A%2F%2Fascendhub.huawei.com%2F%23%2Fdetail%3Fname%3Dascend-tensorflow-arm)获取镜像。

   当前模型支持的镜像列表如[表1](https://gitee.com/alvin_yan/modelzoo_demo/tree/master/LeNet_ID0127_for_TensorFlow#zh-cn_topic_0000001074498056_table1519011227314)所示。

   **表 1** 镜像列表

   | *镜像名称*                                                   | *镜像版本* | *配套CANN版本*                                               |
   | ------------------------------------------------------------ | ---------- | ------------------------------------------------------------ |
   | *ARM架构：[ascend-tensorflow-arm](https://gitee.com/link?target=https%3A%2F%2Fascend.huawei.com%2Fascendhub%2F%23%2Fdetail%3Fname%3Dascend-tensorflow-arm)x86架构：[ascend-tensorflow-x86](https://gitee.com/link?target=https%3A%2F%2Fascend.huawei.com%2Fascendhub%2F%23%2Fdetail%3Fname%3Dascend-tensorflow-x86)* | *20.2.0*   | *[20.2](https://gitee.com/link?target=https%3A%2F%2Fsupport.huawei.com%2Fenterprise%2Fzh%2Fascend-computing%2Fcann-pid-251168373%2Fsoftware)* |

## 快速上手

### 环境配置

requirements.txt 内记录了所需要的第三方依赖与其对应版本，可以通过命令配置所需环境。

> **提示**：项目中的预处理环节需要对原始图像进行定位裁剪，需要使用mxnet完成，该依赖在x86平台下可直接PIP安装，在ARM平台下需要手动编译安装。

### 模型训练

1. 准备数据集

   **若直接使用预处理后的数据集，可跳过2、3步**

   > 数据集在OBS桶中（wiki_crop.tai），需要解压，解压后放到dataset下即可，OBS数据集下载链接：&#x2028;obs://cann-id1250/dataset/

2. 数据预处理

   在./路径下执行以下命令

   ```
   python preproccessing/dataset_proc.py -i ./dataset/wiki_crop --source wiki
   ```

   ```
   // 完整参数选项说明如下
   --source 选择数据集来源，可选“wiki"或"imdb“（若使用imdb需要额外下载数据集）
   -d       数据预处理后的存放位置，默认为./dataset/data
   -i			 数据集存放位置
   -p.      图像边界填充padding值，默认为0.4
   ```

3. 执行训练脚本

   - 可通过train_full_1p.sh直接进拉起训练

   ```
   sh train_full_1p.sh --data_path='./dataset/data/' --output_path='./c3ae_npu_train.h5'
   ```

   - 或在./路径下执行以下命令

   ```
   python nets/c3ae_npu.py --dataset dataset/data -white -se -gpu
   ```

   ```
   --dataset 数据集存放路径
   --source 选择数据集来源，可选“wiki"或"imdb“（若使用imdb需要额外下载数据集），默认wiki
   -b 				batch_size大小，默认50
   -d 				dropout参数值，默认0.2
   -lr 			学习率，默认0.002
   -se 			使用该参数则表示使用se块
   -white 		使用该参数则表示使用white_norm
   -s 				模型保存路径
   -gpu			使用该参数则表示使用GPU进行训练
   ```

## 高级参考

### 脚本和示例代码

```
.
├── dataset 					// 数据集下载存放路径
│   ├── data 					// 预处理后的数据集存放地址
├── detect 						// 预处理相关依赖文件
├── model 						// 模型保存路径
├── nets 
│   ├── c3ae_npu.py 	// 网络及训练脚本
│   └── utils.py 			// 工具代码
├── preproccessing
│   ├── dataset_proc.py // 预处理执行脚本
│   ├── pose.py 				// 预处理相关代码
│   └── serilize.py 		// 预处理相关代码
├── test
│   ├── train_full_1p.sh        //单卡全量训练启动脚本
│   ├── train_performance_1p.sh //单卡训练验证性能启动脚本
│   └── serilize.py 		 // 预处理相关代码
├── requirements.txt 		 //训练python依赖列表
└── README.md 					 // 代码说明文档
```

### 脚本参数

```
--data_path              数据集路径
--output_path            模型文件保存路径
```

## 训练结果

- 精度结果对比

| 精度指标项 | 论文发布 | GPU实测 | NPU实测 |
| ---------- | -------- | :------ | ------- |
| MAE（age） | 6.44     | 6.98    | 7.44    |
