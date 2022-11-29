- [基本信息](#基本信息.md)
- [概述](#概述.md)
- [训练环境准备](#训练环境准备.md)
- [快速上手](#快速上手.md)
- [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Object Detection**

**版本（Version）：1.1**

**修改时间（Modified） ：2022.11.29**

**大小（Size）：450KB**

**框架（Framework）：TensorFlow-gpu_1.14.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：文字检测算法**

<h2 id="概述.md">概述</h2>

## 简述<a name="section194554031510"></a>

基于keras和tensorflow实现的使用可微分二值化实现的实时文本检测算法，主要参考了官方实现MhLiao/DB。使用了“可微分二值化”模块，该模块在分割网络中进行自适应像素二值化过程，简化了后处理，提高了文本检测。

- 参考论文：
  
  [https://arxiv.org/abs/1911.08947](Real-time Scene Text Detection with Differentiable Binarization)

- 参考实现：

  https://github.com/xuannianz/DifferentiableBinarization

- 适配昇腾 AI 处理器的实现：
  
  https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/DifferentiableBinarization

- 通过Git获取对应commit\_id的代码方法如下：
  
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    

## 默认配置<a name="section91661242121611"></a>

-   训练超参（单卡）：
    - Batch size: 16
    - steps_per_epoch：200
    - epochs:100
    - dataset_dir: ./datasets/total_text
    - checkpoints_dir: ./checkpoints
    - start_learning_rate: 1e-3
    - beta_1(一阶矩估计的指数衰减率):0.9
    - beta_2(二阶矩估计的指数衰减率):0.999
    - epsilon(模糊因子):None
    - decay=0

    

<h2 id="训练环境准备.md">训练环境准备</h2>

-  硬件环境和运行环境准备请参见《[CANN软件安装指南](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=installation-update)》
-  运行以下命令安装依赖。
```
pip3 install requirements.txt
```
说明：依赖配置文件requirements.txt文件位于模型的根目录

<h2 id="快速上手.md">快速上手</h2>

## 数据集准备<a name="section361114841316"></a>

数据集路径
  datasets/total_text/train_images
  datasets/total_text/train_gts
  datasets/total_text/train_list.txt
  datasets/total_text/test_images
  datasets/total_text/test_gts
  datasets/total_text/test_list.txt


## 模型训练<a name="section715881518135"></a>

- 运行train.py文件
- 开始训练。

    - 启动训练之前，首先要配置程序运行相关环境变量。

      环境变量配置信息参见：

      [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)

    - 单卡训练

        1.首先在脚本train.py中，配置训练数据集路径，模型存储位置，batch_size,请用户根据实际配置，数据集参数如下所示：

             ```

             --dataset_dir=datasets/total_text  --checkpoints_dir=checkpoints  --batch_size=16

             ```
        2.配置start_learning_rate、beta_1、beta_2、epsilon、decay，在train.py中optimizers.Adam方法中进行配置；

        3.启动训练
        
             启动单卡训练 （脚本为train.py） 
        
             ```
             python3 train.py

             ```
## 测试
`python inference.py`

![image1](test/img192.jpg) 
![image2](test/img795.jpg)
![image3](test/img1095.jpg)           


<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码

```
|--README.md                                                      #说明文档									
|--train.py                                                       #训练代码
|--generator.py
|--losses.py
|--model.py								
|--requirements.txt                                               #所需依赖
|--transform.py	   						
|--datasets                                                       #训练需要的数据集
|       |--test_gts
|       |--test_images
|       |--train_gts
|       |--train_images
|       |--train_list.txt
|       |--train.txt
|--test			           	                         
|	|--img192.jpg
|	|--img795.jpg
|	|--img1095.jpg
|--checkpoints                                                    #模型存放位置
```

## 脚本参数<a name="section6669162441511"></a>

```
--dataset_dir=datasets/total_text					
--checkpoints_dir=checkpoints					
--batch_size=16				       
```

## 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡或者多卡训练。

