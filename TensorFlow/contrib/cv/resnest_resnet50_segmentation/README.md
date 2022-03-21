###   **Resnest**  

Resnest是一种特征提取主干网络，主要作用是提高特征表示能力，提高实例分割和语义分割的能力。

###   **概述** 

由于原论文采用了大的池化核(`28*28`) ，本实验将池化核改为了`7*7`  

原论文64卡，分布式训练。本实验是单机8卡分布式训练。  

实验过程： 首先resnest在GPU8卡上训练然后迁移到ascend910 8卡平台训练  

**目标**： NPU 和 GPU端结果一致，或者更好。

|       | 论文 | 目标精度 GPU 8p | npu完成精度 ascend |
|-------|------|----|-----|
| MIOU | 0.7987 | 0.6338 | 0.6645 |

###  Requirements

1. Tensorflow 1.15
2. GPU 8p
3. Ascend910 8p

###   **代码及路径解释** 

```
WIDERESNET
└─ 
  ├─README.md
  ├─train_data 用于存放训练数据集 #obs://public-dataset/resnest/cityscapes_train.tfrecords
  	├─train.record
  	└─...
  ├─test_data 用于存放测试数据集 #obs://public-dataset/resnest/cityscapes_val.tfrecords
  	├─val.record  
  	└─...
  ├─save_gpu_model 用于存放训练模型 # obs://huaweiinference/resnest/gpu_ckpt/
  	├─checkpoint
  	├─resnest.ckpt.data-00000-of-00001
  	├─resnest.index
  	├─resnest.meta
  	└─...
  ├─save_npu_model 用于存放经8卡npu过训练后的模型文件 # obs://huaweiinference/resnest/npu_ckpt/
  	├─checkpoint
  	├─resnest.ckpt.data-00000-of-00001
  	├─resnest.index
  	├─resnest.meta
  	└─...
  ├─premodel npu端的加载的gpu端的预训练模型 obs://huaweiinference/resnest/reload_tosave_ckpt/
  	├─checkpoint
  	├─resnest.ckpt.data-00000-of-00001
  	├─resnest.index
  	├─resnest.meta
  	└─...
  ├─premodel npu和gpu分布式训练模型使用单卡加载转存后的ckpt文件 obs://huaweiinference/resnest/reload_tosave_ckpt/
  	├─checkpoint
  	├─resnest.ckpt.data-00000-of-00001
  	├─resnest.index
  	├─resnest.meta
  	└─...
  ├─offline_inference 推理文件
  	├─convert.py
  	├─resnest.pb
  	├─resnest.om
  	└─...
  ├─gpu_8_distribute_main.py gpu8卡训练脚本
  ├─resnest.py 经过修改后的resnest网络 
  ├─run_8_distribute_train.py 进行npu 8卡训练的脚本  
  ├─train_8p.sh GPU8卡启动脚本
  ├─test_1p.sh  模型测试脚本
  ├─hccl_8p.json  hccl配置脚本 # 需要根据自己的机器适配
  ├─eavl_ckpt.py  模型推理测试脚本

```
###   **数据集** 

数据集 Cityscapes   
Cityscapes拥有5000张在城市环境中驾驶场景的图像（2975train，500 val,1525test）。它具有19个类别的密集像素标注（97％coverage），其中8个具有实例级分割。Cityscapes数据集，即城市景观数据集，这是一个新的大规模数据集，其中包含一组不同的立体视频序列，记录在50个不同城市的街道场景。城市景观数据集中于对城市街道场景的语义理解图片数据集，该大型数据集包含来自50个不同城市的街道场景中记录的多种立体视频序列，除了20000个弱注释帧以外，还包含5000帧高质量像素级注释。因此，数据集的数量级要比以前的数据集大的多。Cityscapes数据集共有fine和coarse两套评测标准，前者提供5000张精细标注的图像，后者提供5000张精细标注外加20000张粗糙标注的图像。Cityscapes数据集包含2975张图片。包含了街景图片和对应的标签.Cityscapes数据集，包含戴姆勒在内的三家德国单位联合提供，包含50多个城市的立体视觉数据。   

### 训练过程及结果
epoch=500    
batch_size=12    
lr=0.00001  

GPU 精度模型
GPU : obs://huaweiinference/resnest/gpu_ckpt/

NPU 精度模型
NPU : obs://huaweiinference/resnest/npu_ckpt/

###   **train** 

`bash train_gpu_8p.sh`


###  **eval** 

`bash tet_gpu_1p.sh`

###  **参数解释**  


 model_path---------------加载模型的路径（例如 ./model/xception_model.ckpt）不加载预训练模型时设为None即可  
 train_data----------------tfrecord数据集的路径 （例如 ./train_data），只需要将所有的tfrecord文件放入其中 \
 model_save_dir--------------经过fine_turn后的模型保存路径 \
 is_training-----------------是否训练，默认加载模型进行eval，如若需要加载预训练模型进行训练需将该值设为True\
 img_N----------------相应数据集包含图片数量\
 batch_size---------------当do_train 为False时，该值需要能被图片数量整除，以确保最终准确率的准确性，do_train为True时则无该要求\
 epochs--------------------该值只在do_train 为True时有效，表示训练轮次\
 learning_rate------------学习率\

### 说明
没有达到论文精度的原因： 
1. 修改了模型  
2. 没有论文提到的64p环境  


###  **offline_inference** 

[offline_inference readme](./offline_inference/README.md)