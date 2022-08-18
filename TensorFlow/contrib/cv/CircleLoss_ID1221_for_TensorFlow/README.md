- [基本信息](#基本信息.md)
- [概述](#概述.md)
- [训练环境准备](#训练环境准备.md)
- [快速上手](#快速上手.md)
- [迁移学习指导](#迁移学习指导.md)
- [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Object Detection**

**版本（Version）：1.1**

**修改时间（Modified） ：2022.8.17**

**大小（Size）：384KB**

**框架（Framework）：TensorFlow_1.15**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的深度度量学习的代理锚损失训练代码**

<h2 id="概述.md">概述</h2>

## 简述<a name="section194554031510"></a>

Circle 损失对于两种基本的深度特征学习方法有一个统一的公式，即使用类级别标签和成对标签进行学习。Circle 损失提供了一种更灵活的优化方法，可以实现更明确的收敛目标。通过实验，我们证明了 Circle loss 在各种深度特征学习任务中的优越性。在人脸识别、人员重新识别以及几个细粒度的图像检索数据集上，所取得的性能与现有技术相当。通过实验，我们证明了 Circle loss 在各种深度特征学习任务中的优越性。在人脸识别、人员重新识别以及几个细粒度的图像检索数据集上，所取得的性能与现有技术相当。通过实验，我们证明了 Circle loss 在各种深度特征学习任务中的优越性。在人脸识别、人员重新识别以及几个细粒度的图像检索数据集上，所取得的性能与现有技术相当

- 参考论文：
  
  [https://arxiv.org/abs/2002.10857][Circle Loss: A Unified Perspective of Pair Similarity Optimization]

- 参考实现：

  https://github.com/zhen8838/Circle-Loss

- 适配昇腾 AI 处理器的实现：
  
  https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/CircleLoss_ID1221_for_TensorFlow

- 通过Git获取对应commit\_id的代码方法如下：
  
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    

## 默认配置<a name="section91661242121611"></a>

-   训练超参（单卡）：
    - Batch size： 30
    - data_url        #数据集
    - train_url       #训练结果                     
    - max_nrof_epochs
    - data_dir        #数据集MS-Celeb-1M_clean_align
    - lfw_dir         #数据集lfw-deepfunneled_align
    - learning_rate  0.1
    - epoch_size 

## 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 否       |
| 混合精度   | 是      |
| 数据并行   | 否     |


## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

拉起脚本中，

```
 ./train_full_1p.sh --help

parameter explain:
    --precision_mode         #precision mode(allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision)
    --data_path              # dataset of training
    --output_path            # output of training
    --train_steps            # max_step for training
    --train_epochs           # max_epoch for training
    --batch_size             # batch size
    -h/--help                show help message
```

混合精度相关代码示例:

 ```
 custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")

 ```

<h2 id="训练环境准备.md">训练环境准备</h2>

-  硬件环境和运行环境准备请参见《[CANN软件安装指南](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=installation-update)》
-  运行以下命令安装依赖。
```
pip3 install requirements.txt
```
说明：依赖配置文件requirements.txt文件位于模型的根目录

<h2 id="快速上手.md">快速上手</h2>

## 数据集准备<a name="section361114841316"></a>

1、模型训练数据集请用户自行获取(obs://cann-id1221/dataset/)。

2、CircleLoss训练的模型及数据集可以参考"简述 -> 参考实现"


## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练。

    - 启动训练之前，首先要配置程序运行相关环境变量。

      环境变量配置信息参见：

      [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

    - 单卡训练


        1. 配置训练参数
        
          首先在脚本test/train_full_1p.sh中，配置batch_size、steps、epochs、data_path等参数，请用户根据实际路径配置data_path，或者在启动训练的命令行中以参数形式下发。
        
             ```
            --data_url=${cur_path}/../dataset 
            --train_url=${output_path} 
            --max_nrof_epochs=500 
            --data_dir=${cur_path}/../dataset/MS-Celeb-1M_clean_align 
            --lfw_dir=${cur_path}/../dataset/lfw-deepfunneled_align 
             ```
        
         2. 启动训练。
        
             ```
             bash test/train_full_1p.sh
             ```
        
         3.精度训练结果。
         ```
            | -   | 论文    | GPU （v100） | NPU   |
            |-----|-------|------------|-------|
            | acc | 0.997 | 0.948      | 0.951 |
        ```


<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码

```
├── README.md                          
├── circle_loss.py						
├── ckpt2pb.py									
├── facenet.py				
├── lfw.py								
├── modelzoo_level.txt 						
├── modelzoo_level.txt						
├── requirements.txt																					
├── modelarts_entry_acc.py                  
├── modelarts_entry_perf.py                 
├── test     
│    ├──train_performance_1p.sh             //训练性能入口
│    ├──train_full_1p.sh                    //训练精度入口，包含准确率评估
```

## 脚本参数<a name="section6669162441511"></a>

```
- Batch size： 30
- data_url        #数据集
- train_url       #训练结果                     
- max_nrof_epochs
- data_dir        #数据集MS-Celeb-1M_clean_align
- lfw_dir         #数据集lfw-deepfunneled_align
- learning_rate  0.1
- epoch_size 

```

## 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡。模型存储路径为${cur_path}/output/$ASCEND_DEVICE_ID，包括训练的log以及checkpoints文件。以1卡训练为例，loss信息在文件${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。