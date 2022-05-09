## 基本信息

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Computer Vision**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Research**

**描述（Description）：基于TensorFlow框架对CIFAR-10数据集进行分类的训练代码** 

## 概述

	CSPNet

- 参考论文：

    https://openaccess.thecvf.com/content_CVPRW_2020/papers/w28/Wang_CSPNet_A_New_Backbone_That_Can_Enhance_Learning_Capability_of_CVPRW_2020_paper.pdf

- 参考实现：

    https://github.com/WongKinYiu/CrossStagePartialNetworks


## 默认配置<a name="section91661242121611"></a>

- 训练数据集预处理:
  
  - 解压CIFAR-10数据集
  
- 训练超参

  - EPOCHS    50
  - BATCH_SIZE    64
  - LEARNING_RATE    1e-03
  - MOMENTUM    0.9
  - LAMBDA    5e-04

## 快速上手

数据集准备
模型训练使用CIFAR-10数据集，数据集请用户自行获取.  
obs桶地址:
>obs://cann-id0840/npu/

## 模型训练
单卡训练 

1. 配置训练参数
2. 启动训练
```
bash train_full_1p.sh \    
    --data_path="./dataset" \  
    --output_path="./checkpoint"
```


## 训练结果

- 精度结果比对

|精度指标项|GPU实测|NPU实测|
|---|---|---|
|Top-1 Acc|0.6605|0.6645|

- 性能结果比对  

|性能指标项|GPU实测|NPU实测|
|---|---|---|
|FPS|1126.45|1245.03|


## 文件夹结构

```
├── README.md                                 //代码说明文档
├── utils.py                                  //工具文件
├── train.py                                  //网络训练
├── test.py                               //用于衡量模型在数据集上的精度
├── requirements.txt                          //依赖列表
├── LICENSE                                   
├── checkpoint                                //checkpoint模型保存地址
├── models                                  //模型定义
│    ├──cspdarknet53.py
├── dataset                                 //cifar-10数据集文件夹
│    ├──data_batch_1
│    ├──data_batch_2
│    ├──data_batch_3
│    ├──data_batch_4
│    ├──data_batch_5
│    ├──test_batch
│    ├──batches.meta
│    ├──readme.html
├── test
│    ├──train_performance_1p.sh             //单卡训练验证性能启动脚本
│    ├──train_full_1p.sh                    //单卡全量训练启动脚本

```


## 启动脚本说明
在test文件夹下, 有train_performace_1p.sh和train_full_1p.sh脚本,
可分别用于检测训练性能与训练精度.

### 检测性能
命令：
```
bash train_performance_1p.sh \    
    --data_path="./dataset" \  
    --output_path="./checkpoint"
```
打屏信息:

>------------------ INFO NOTICE START------------------  
>INFO, your task have used Ascend NPU, please check your result.  
>------------------ INFO NOTICE END------------------  
>------------------ Final result ------------------  
>Final Performance images/sec : 1245.0326379815197  
>Final Performance sec/step : 0.05140427491423721  
>E2E Training Duration sec : 536  
>Final Train Accuracy : 0.5850  


### 检测精度
命令:
```
bash train_full_1p.sh \    
    --data_path="./dataset" \  
    --output_path="./checkpoint"
```

打屏信息:
>------------------ INFO NOTICE START------------------  
>INFO, your task have used Ascend NPU, please check your result.  
>------------------ INFO NOTICE END------------------  
>------------------ Final result ------------------  
>Final Performance images/sec : 1785.9793972748832  
>Final Performance sec/step : 0.03583467989477017  
>E2E Training Duration sec : 1993  
>Final Train Accuracy : 0.6645  


