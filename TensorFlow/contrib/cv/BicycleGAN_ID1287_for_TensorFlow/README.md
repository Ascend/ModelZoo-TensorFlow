###   **BicycleGAN** 
BicycleGAN模型是Toward Multimodal Image-to-Image Translation论文的Tensorflow的实现，该论文的核心思想体现在确保输入噪声向量与输出图像的双向映射一致性。BicycleGAN通过结合cVAE-GAN和cLR-GAN这两个方法来共同地促进隐层向量和输出图像在两个方向上的连接。通过BicycleGAN生成的图像多样性更好，且更具有视觉上的真实性。

###   **概述** 

迁移NIMA到ascend910平台
将结果与原论文进行比较

 |                | 论文   | ascend |
|----------------|------|--------|
| LIPIS Distance | 0.110±0.002 | 待测 |


###  Requirements

1. Tensorflow 1.15

###   **代码及路径解释** 



```
BicycleGAN
└─ 
  ├─README.md
  ├─folder_npu.py 用于检查文件夹结构
  ├─layers.py 用于创建基础的神经层
  ├─load_data_npu.py 用于创建数据流
  ├─log.py 用于创建训练日志
  ├─model_npu_tmp.py 用于定义模型结构
  ├─main_npu.py 用于启动训练和测试过程
  ├─maps 用于存放训练数据集 obs://bicyclegan/BicycleGAN2/maps/
  	├─train
  	  └─...
  	├─val
  	  └─...
  ├─checkpoints 用于存放训练好的模型文件
  ├─logs 用于存放训练日志
  ├─results 用于存放训练集和测试集的测试的结果
  ├─train_1p.sh 模型的启动脚本，
  ├─test_1p.sh 模型的启动测试脚本
```
###   **数据集和模型** 

BicycleGAN模型所使用的数据集为Google maps-satellites,是一个pixel to pixel的风格迁移数据集，其中包括1096张实际街景图片和与之对应的地图标签。



### 训练过程及结果
epoch=200 \
batch_size=1 \
lr=0.0002 \
耗费近1小时

### 数据集百度云链接及提取码
链接：https://pan.baidu.com/s/17rKdfkp_8_pvn89nII13fg 
提取码：zdx1


 **启动训练和测试过程**

执行shell脚本：
```
bash train_1p.sh
```