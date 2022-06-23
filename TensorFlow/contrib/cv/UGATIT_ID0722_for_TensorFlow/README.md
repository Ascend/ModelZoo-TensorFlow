## 基本信息

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Computer Vision**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Research**

**描述（Description）：基于TensorFlow框架的风格迁移代码** 

## 概述

	U-GAT-IT

- 参考论文：

    https://arxiv.org/abs/1907.10830

- 参考实现：

    https://github.com/taki0112/UGATIT


## 默认配置<a name="section91661242121611"></a>

- 训练数据集预处理:

  - 图像的输入尺寸为 1080*720
  - 图像输入格式：jpg

- 训练超参

  - Batch size： 1
  - epoch: 10001   
  **注:** epoch表示stop epoch 而非new epoch,当epoch*iteration< checkpoint中step时，将不会进行新的训练.

## 快速上手

数据集准备
模型训练使用selfie2anime数据集，数据集请用户自行获取.
官方数据集地址：
https://drive.google.com/file/d/1xOWj1UVgp6NKMT3HbPhBbtq2A4EDkghF
obs桶地址:
>obs://cann-id0722/dataset/

## 文件夹结构

```
├── README.md                                 // 代码说明文档
├── main.py                                  // 网络训练/推理入口
├── ops.py                                   //网络层定义
├── UGATIT.py                                //网络定义
├── utils.py                                 
├── requirements.txt                          //依赖列表
├── LICENSE                                   
├── output                                  //输出文件夹
│    ├──checkpoint                          //模型输出
│    ├──logdir                              //日志输出
│    ├──results                             //结果输出
│    └──sample
├── dataset
│    └── selfie2anime
│        ├── trainA
│            ├── xxx.jpg (name, format doesn't matter)
│            ├── yyy.png
│            └── ...
│        ├── trainB
│            ├── zzz.jpg
│            ├── www.png
│            └── ...
│        ├── testA
│            ├── aaa.jpg 
│            ├── bbb.png
│            └── ...
│        ├── testB
│            ├── ccc.jpg 
│            ├── ddd.png
│            └── ...
│        └── init_model
│            └── model_name                      //模型名称,如:UGATIT_lsgan_4resblock_6dis_1_1_10_10_1000_sn_smoothing
│                ├── checkpoint
│                ├── UGATIT.model-1000000.data-00000-of-00001
│                ├── UGATIT.model-1000000.index
│                └── UGATIT.model-1000000.meta
├── test
│    ├── train_performance_1p.sh             //单卡训练验证性能启动脚本
│    └── train_full_1p.sh                    //单卡全量训练启动脚本

```

## 模型训练
单卡训练 

1. 配置训练参数
2. 启动训练
```
bash train_full_1p.sh \    
    --data_path="./dataset/selfie2anime" \  
    --output_path="./output"
```

## 特别说明
(1)由于ModelArts目前只能指定一个输入文件夹,因此将图片数据集和训练所需的初始checkpoint文件均保存在./dataset文件夹中(具体位置请参考上文文件夹结构).  

(2)在进行在线推理(即将main.py中phase设置为'test'时),需要将推理所用的checkpoint文件存放到./dateset/init_model文件夹中对应位置.  
因此,若需使用训练生成的checkpoint进行推理,需要手动将./output/checkpoint文件夹中的checkpoint文件移动到./dateset中存放checkpoint的位置(具体位置请参考上文文件夹结构).  

(3)当前版本的ModelArts输入参数为data_url,输出参数为train_url.  
本代码中,为配合训练train_performace_1p.sh脚本文件中参数定义，已分别改为 data_path 和 output_path.  
请使用者注意 

## 训练结果

- 精度结果比对  
观察NPU生成结果,与GPU生成结果相同,如图所示.


- 性能结果比对  

|性能指标项|GPU实测|NPU实测|
|---|---|---|
|FPS|1.71|0.23|



## 启动脚本说明
在test文件夹下, 有train_performace_1p.sh和train_full_1p.sh脚本,
可分别用于检测训练性能与训练精度.  
注:
  - train_performace_1p.sh可用于计算FPS.  
  - train_full_1p.sh可训练一个完整的epoch.  
    训练完成后需手动进行推理(将生成的checkpoint移动到init_model中,并将main.py中pahse设为test),可观察GPU结果和NPU结果是否相同.若相同,可知精度达标.

### 检测性能
命令：
```
bash train_performace_1p.sh \    
    --data_path="./dataset/MPII" \  
    --output_path="./checkpoint/model.ckpt"
```
打屏信息:

> ------------------ INFO NOTICE START------------------  
> INFO, your task have used Ascend NPU, please check your result.  
> ------------------ INFO NOTICE END------------------  
> ------------------ Final result ------------------  
> Final Performance images/sec : 0.23  
> Final Performance sec/step : 4.29  
> E2E Training Duration sec : 1075  
> Final Train Accuracy :  

### 检测精度
命令:
```
bash train_full_1p.sh \    
    --data_path="./dataset/MPII" \  
    --output_path="./checkpoint/model.ckpt"
```

打屏信息:
> ------------------ INFO NOTICE START------------------  
> INFO, your task have used Ascend NPU, please check your result.  
> ------------------ INFO NOTICE END------------------  
> ------------------ Final result ------------------  
> Final Performance images/sec : 0.23  
> Final Performance sec/step : 4.29  
> E2E Training Duration sec : 1075  
> Final Train Accuracy :  


