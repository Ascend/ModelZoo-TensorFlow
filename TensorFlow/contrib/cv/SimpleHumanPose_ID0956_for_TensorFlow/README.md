## SimpleHumanPose

### 基本信息

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Human Pose Estimation**

**版本（Version）：1.0**

**修改时间（Modified） ：2021.11.28**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Benchmark**

**描述（Description）：基于TensorFlow框架的SimpleHumanPose网络训练及测试代码**

### 概述

SimpleHumanPose模型主要针对人体姿态估计和姿态追踪，提出一种简化的baseline。当前流行的人体姿态估计方法都过于复杂，各种模型在结构上看起来差异性很大，但是性能上又很接近，很难判断究竟是什么在起作用。相比于其他人体姿态估计模型，该模型并没有使用过多的技巧和理论依据。它以简单的技术为基础，通过全面的消融实验进行了验证，提出一个较为简化直观的模型，取得了良好的性能。

- 参考论文：

  https://arxiv.org/abs/1804.06208

- 参考实现：

  https://github.com/mks0601/TF-SimpleHumanPose

- 通过Git获取对应commit_id的代码方法如下：

  ```
  git clone {repository_url}		# 克隆仓库的代码
  cd {repository_name}    		# 切换到模型的代码仓目录
  git checkout {branch}			# 切换到对应分支
  git reset --hard {commit_id}	# 代码设置到对应的commit_id
  cd {code_path}					# 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```
### 默认配置

- 训练超参
  - Base learning rate: 5e-4
  - Weight decay: 1e-5
  - Train epoch: 140
  - Batch size: 32
  - Optimizer: Adam
  - Pre-trained model: ResNet-50

### 训练环境准备

1. 硬件环境准备请参见各硬件产品文档"[驱动和固件安装升级指南](https://support.huawei.com/enterprise/zh/category/ai-computing-platform-pid-1557196528909)"。需要在硬件设备上安装与CANN版本配套的固件与驱动。

2. 宿主机上需要安装Docker并登录[Ascend Hub中心](https://ascendhub.huawei.com/#/detail?name=ascend-tensorflow-arm)获取镜像。

   当前模型支持的镜像列表如[表1](https://gitee.com/ascend/modelzoo/blob/master/built-in/TensorFlow/Official/cv/image_classification/DenseNet121_for_TensorFlow/README.md#zh-cn_topic_0000001074498056_table1519011227314)所示。

   **表 1** 镜像列表

   | *镜像名称*                                                   | *镜像版本* | *配套CANN版本*                                               |
   | ------------------------------------------------------------ | ---------- | ------------------------------------------------------------ |
   | $\circ$ *ARM架构：[ascend-tensorflow-arm](https://ascend.huawei.com/ascendhub/#/detail?name=ascend-tensorflow-arm)* <br>$\circ$ *x86架构：[ascend-tensorflow-x86](https://ascend.huawei.com/ascendhub/#/detail?name=ascend-tensorflow-x86)* | *20.2.0*   | *[20.2](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software)* |

### 脚本和示例代码

```bash
├── README.md                                 //代码说明文档
├── LICENSE                                   //许可证
├── src
|    ├── boot_train.py                        //ModelArts启动训练代码
|    ├── boot_test.py                         //ModelArts启动测试代码
|    ├── pip-requirements.txt                 //配置文件，用于指定依赖包的包名及版本号
│    ├── data                                 //包含数据加载代码
         ├── COCO
             ├── dataset.py  
│    ├── lib                                  //包含2D多人姿态估计系统的核心代码
         ├── nets/
         ├── nms/
         ├── tfflat/
         ├── utils/
         ├── __init__.py
         ├── Makefile                           
│    ├── main                                 //包含train/test网络的代码          
         ├── config.py
         ├── gen_batch.py
         ├── model.py
         ├── test_my.py
         ├── train.py               
```

### 模型训练

- 源码准备

  单击“立即下载”，并选择合适的下载方式下载源码包，并解压到合适的工作路径。

- 数据集准备

  模型使用COCO2017数据集，请用户自行准备好数据集，包含训练集和验证集两部分。另外，还需要准备COCO2017数据集的annotations文件、ResNet-50预训练模型。test时还需要准备human detection结果，并将训练得到的ckpt放在model_dump文件夹下，按照下面的目录结构进行组织。
  
  ```bash
  ├── dataset
  │    ├── annotations
           ├── person_keypoints_train2017.json
           ├── person_keypoints_val2017.json                            
  │    ├── dets    
           ├── human_detection.json                     
  │    ├── imagenet_weights
           ├── resnet_v1_50.ckpt
  │    ├── images            
           ├── train2017/
           ├── val2017/        
  │    ├── model_dump                                          
  ```
- 训练过程
  
  在main/config.py中，您可以更改模型的设置，包括网络主干、batch size大小和输入大小等。训练可以执行命令：

  ```bash
  python main/train.py
  ```
- 测试过程
  
  将准备好的human detection结果放在dataset/dets目录下，将训练好的模型放在dataset/model_dump目录下（论文中使用的是第140个epoch得到的模型）。测试可以执行命令：

  ```bash
  python main/test_my.py
  ```
### 结果

1. 模型精度

   |Methods|AP|AP.5 |AP.75|AP(M)|AP(L)|AR|AR.5|AR.75|AR(M|AR(L)|
   |--|--|--|--|--|--|--|--|--|--|--|
   |256x192_resnet50(NPU)|70.5|89.3|78.0|67.1|76.9|76.3|93.2|83.2|72.0|82.4|
   |256x192_resnet50(GPU)|70.2|89.0|77.6|66.8|76.9|76.0|93.1|82.7|71.7|82.3|

2. 模型性能

   |Methods|time cost / epoch|
   |--|--|
   |256x192_resnet50(NPU)|0.21h/epoch|
   |256x192_resnet50(GPU)|0.27h/epoch|
