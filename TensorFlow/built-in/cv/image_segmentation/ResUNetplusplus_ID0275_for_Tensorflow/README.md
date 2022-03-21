
# ResUNet++: An advanced architecture for medical image segmentation
Tensoflow implementation of resunet++
# ResUNet++

The ResUNet++ architecture is based on the Deep Residual U-Net (ResUNet), which is an architecture that uses the strength of deep residual learning and U-Net. The proposed ResUNet++ architecture takes advantage of the residual blocks, the squeeze and excitation block, ASPP, and the attention block. 
More description about the archicture can be in the paper [ResUNet++: An Advanced Architecture for Medical Image Segmentation] (https://arxiv.org/pdf/1911.07067.pdf).

原始模型参考[[github链接](https://github.com/DebeshJha/ResUNetplusplus)]，迁移训练代码到NPU Ascend 910

## 结果展示
|                   | 精度（val_precision） | 性能（s/epoch）      |
| :---------------: | :-------------------: | -------------------- |
|       基线        |        87.85%         | 518.32               |
| NPU（Ascend 910） | 83.49%（CANN 5.0.2）  | 398.87（CANN 5.0.2） |

##训练超参数
	image_size = 256
    batch_size = 8
    lr = 1e-5
    epochs = 200



## Requirements:

    os
    numpy
    cv2
    tensorflow
    glob
    tqdm

## 快速启动

##开始训练
执行python3 run.py即可启动训练脚本  
显示进度后就算正常开始训练了，脚本会自动保存精度最优的ckpt 

``````
python3 run.py
``````

也可以执行test目录下的归一化脚本：

```bash train_full_1p.sh --data_path=./new_data/Kvasir-SEG```  或

```bash train_performance_1p.sh --data_path=./new_data/Kvasir-SEG```

## 目录

	├-- data					----原始数据集存放路径
	└-- new_data				----数据集增强处理后存放路径
	├-- precision_tool			----精度调优工具
	|-- test                    ----NPU训练归一化shell
	    |--env.sh
	    |--launch.sh
	    |--train_full_1p.sh
	    |--train_performance_1p.sh
	│-- data_generator.py		----数据读取
	│-- infer.py				----ckpt测试脚本
	│-- metrics.py				 
	│-- m_resunet.py			----网络定义
	│-- process_image.py		----数据增强处理
	│-- resunet.py 				----模型文件
	│-- run.py					----训练脚本
	│-- unet.py 				----模型文件

## Setup

Download this repository	

## Dataset

首先需要下载Kvasir-SEG(https://datasets.simula.no/kvasir-seg/Kvasir-SEG.zip) 数据集解压到data目录下  

`unzip Kvasir-SEG.zip`

执行python3 process_image.py进行数据集的划分和增强  

``````
python3 process_image.py
``````

## Run

``````
python3 run.py 
``````

也可以执行归一化脚本：

```bash train_full_1p.sh --data_path=./new_data/Kvasir-SEG```  或

```bash train_performance_1p.sh --data_path=./new_data/Kvasir-SEG```