# 基本信息：
发布者（Publisher）：Huawei  
应用领域（Application Domain）： Signature-detection
版本（Version）：1.0  
修改时间（Modified） ：2021.12.26  
框架（Framework）：TensorFlow 1.15.0  
模型格式（Model Format）：ckpt  
处理器（Processor）：昇腾910  
应用级别（Categories）：Research  
描述（Description）：基于TensorFlow框架的特征点检测网络训练代码  

# 概述：
Factorized是一个通过分解空间嵌入对对象地标进行无监督学习
参考论文及源代码地址：  
Unsupervised learning of object landmarks by factorized spatial embeddings
https://github.com/alldbi/Factorized-Spatial-Embeddings

# 默认配置：
1.训练数据集预处理  
celebA
2.测试数据集预处理  
celebA  
3.训练超参  
LANDMARK_N = 8
SAVE_FREQ = 500
SUMMARY_FREQ = 20
BATCH_SIZE = 32
DOWNSAMPLE_M = 4
DIVERSITY = 500.
ALIGN = 1.
LEARNING_RATE = 1.e-4
MOMENTUM = 0.5
RANDOM_SEED = 1234
WEIGHT_DECAY = 0.0005
SCALE_SIZE = 146
CROP_SIZE = 146
MAX_EPOCH = 200
# 训练环境准备：
ascend-share/5.1.rc1.alpha003_tensorflow-ascend910-cp37-euleros2.8-aarch64-training:1.15.0-21.0.2_0317

# 快速上手：
数据集的准备：  
Training dataset：celebA。  

# 模型训练：
启动训练文件：train.py  


# 模型测试：
启动训练文件：test.py  
 


﻿# 文件说明

├── README.md                      //说明文档                        
├── requirements.txt	           //依赖                   
├── modelzoo_level.txt             //进度说明文档                         
├── LICENSE                        //license                                        
                                                                                                                                                             ├── VDSR.py   			   //训练启动文件                                                                                                              
├── train.py                       //调用模块1                  
├── test.py            //调用模块2                     
├── utils/warp.py                        //调用模块3	                   
├── utils/ThinPlateSplineB.py                        //调用模块4             



