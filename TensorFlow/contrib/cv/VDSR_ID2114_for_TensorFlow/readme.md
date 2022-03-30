# 基本信息：
发布者（Publisher）：Huawei  
应用领域（Application Domain）： Super-Resolution  
版本（Version）：1.0  
修改时间（Modified） ：2021.12.26  
框架（Framework）：TensorFlow 1.15.0  
模型格式（Model Format）：ckpt  
处理器（Processor）：昇腾910  
应用级别（Categories）：Research  
描述（Description）：基于TensorFlow框架的超分辨率网络训练代码  

# 概述：
VDSR是一个可以解决超分辨率问题的网络，即对给定的低分辨率图像进行处理，得到对应的清晰的高分辨率图像。该网络据SRCNN改进而来，主要是加深了卷积网络的深度，并解决了深度网络带来的一系列问题。此外，该网络同时适用于处理Scale Factor为x2,x3,x4的模糊图片。  
参考论文及源代码地址：  
Accurate Image Super-Resolution Using Very Deep Convolutional Networks  
https://github.com/Jongchan/tensorflow-vdsr

# 默认配置：
1.训练数据集预处理  
图像的输入尺寸为41*41  
图像输入格式：mat（Matlab保存数据的格式）  
2.测试数据集预处理  
图像输入格式：mat（Matlab保存数据的格式）  
3.训练超参  
Batch size: 64  
BASE_LR = 0.0001  
LR_RATE = 0.1  
LR_STEP_SIZE = 120  
MAX_EPOCH = 100  

# 训练环境准备：
ascend-share/5.0.4.alpha005_tensorflow-ascend910-cp37-euleros2.8-aarch64-training:1.15.0-21.0.2_1224

# 快速上手：
数据集的准备：  
Training dataset：Train291；Test dataset：Set5、Set14、Urban100、B100。  
数据集的处理：  
训练图片：数据增强，分块成41x41，mat格式，生成不同Scale Factor下的模糊图像，命名（后标显示Scale Factor，如0_2.mat表示第0张图，Scale Factor为2，处理后的图片）  
测试图片：mat格式，生成不同Scale Factor下的模糊图像，命名（后标显示Scale Factor）  

# 模型训练：
启动训练文件：VDSR.py  
准备好训练集后，把VDSR.py中DATA_PATH改为训练集的路径

# 模型测试：
启动训练文件：test.py  
准备好训练好的模型，其路径传入test.py中的model_ckpt  
准备好测试集，其路径传入test.py中的DATA_PATH  

# 结果
![输入图片说明](%E7%B2%BE%E5%BA%A6.png)


![输入图片说明](T%7DR5%7DO6UHSG4%60%25D%7DNN$Q7%5BM.png)


﻿# 文件说明

├── README.md                      //说明文档                        
├── requirements.txt	           //依赖                   
├── modelzoo_level.txt             //进度说明文档                         
├── LICENSE                        //license                                        
├── test		           //测试启动文件  
                                                                                                                                                             ├── VDSR.py   			   //训练启动文件                                                                                                              
├── MODEL.py                       //调用模块1                  
├── MODEL_FACTORIZED.py            //调用模块2                     
├── PSNR.py                        //调用模块3	                   
├── PLOT.py                        //调用模块4             
