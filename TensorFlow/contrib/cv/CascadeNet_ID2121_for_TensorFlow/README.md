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

**修改时间（Modified） ：2022.8.24**

**大小（Size）：676MB**

**框架（Framework）：TensorFlow_1.15.0**

**模型格式（Model Format）：hdf5**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的用于 MR 图像重建的卷积神经网络和卷积递归神经网络的深度级联训练代码**

<h2 id="概述.md">概述</h2>

## 简述<a name="section194554031510"></a>

ascade模型是论文“A Deep Cascade of Convolutional Neural Networks for Dynamic MR Image Reconstruction”的Tensorflow实现（基于晟腾NPU）。该框架使用卷积神经网络（CNN）的深层级联从欠采样数据中重建2D心脏磁共振（MR）图像的动态序列，以加速数据采集过程，论文提出了Data sharing层，有效引入了核磁影像已知的先验信息，降低了网络的学习难度，有效保证了网络学习的效果。

- 参考论文：
  
  [https://arxiv.org/pdf/1703.00555.pdf](A Deep Cascade of Convolutional Neural Networks for Dynamic MR Image Reconstruction)

- 参考实现：

  https://github.com/js3611/Deep-MRI-Reconstruction   

- 适配昇腾 AI 处理器的实现：
  
  https://gitee.com/ascend/ModelZoo-TensorFlow/blob/master/TensorFlow/contrib/cv/CascadeNet_ID2121_for_TensorFlow

- 通过Git获取对应commit\_id的代码方法如下：
  
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    

## 默认配置<a name="section91661242121611"></a>

-   训练超参（单卡）：
    - batch_size 16
    - num_epoch  5
    - data_url
    - data_train_dir
    - image_size
    - save_ckpt_Dir
    - learning_rate 1e-3

## 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 否       |
| 混合精度   | 是      |
| 数据并行   | 否       |


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

1、用户需自行下载data数据集(方法见https://github.com/js3611/Deep-MRI-Reconstruction)

2、CascadeNet的模型及数据集可以参考"简述 -> 参考实现"


## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练。

    - 启动训练之前，首先要配置程序运行相关环境变量。

      环境变量配置信息参见：

      [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

    - 单卡训练


        1.首先在脚本test/train_performance_1p.sh中, 本模型需要训练训练需要根据安装教程，配置输入与输出的路径。配置训练数据集路径，请用户根据实际路径配置，数据集参数如下所示：

            ```
    	    --data_url ${data_path} 
    	    --batch_size 16 
    	    --image_size  256 
    	    --num_epoch 2 
    	    --data_train_dir ${data_path}/chest_train_acc3.hdf5 
    	    --save_ckpt_Dir ./save_ckpt 
            --data_test_dir ${data_path}/chest_test_acc3.hdf5
            ```

        2.启动训练
        
             启动单卡训练  
        
             ```
             bash train_performance_1p.sh
             ```

        3.full脚本训练如下
             
             ```
             训练脚本

             python3 train.py   

             测试脚本

             python3 test.py 

             ```   
           
        4.性能训练结果(在GPU复现中，由于自行编写的脚本与原论文中不同，评估结果也有一定差异。以下为复现者自行编写后的评估结果)
        
             ```
            |                |  GPU | NPU   |
            |----------------|------|--------|
            | 平均单Step耗时 | ~7.0E-4 s | ~7.7E-4 s |
             ```             
        
    
<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码

```
|--LICENSE
|--README.md                                                      #说明文档									                                         
|--modelzoo_level.txt									
|--requirements.txt                                               #所需依赖                                                 
|--test			           	                          #训练脚本目录
|	|--train_full_1p.sh
|	|--train_performance_1p.sh
|--ckpt_to_pb.py
|--image_to_bin.py                                                     								                                         
|--make_h5data.py									
|--om_precision.py 
|--test.py									
|--train.py
|--view_pb.py 
|--models		           	                          
|	|--model_tf.py
|	|--pre_input.py
```

## 脚本参数<a name="section6669162441511"></a>

```
--batch_size
--image_size
--random_seed
--num_epoch
--checkpoint_period
--data_url
--data_train_dir
--learning_rate
--continue_training
--train_url
--last_checkpoint_dir
--last_checkpoint_dir_name                   
--save_ckpt_Dir
```

## 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡。模型存储路径为${cur_path}/output/$ASCEND_DEVICE_ID，包括训练的log以及checkpoints文件。以1卡训练为例，loss信息在文件${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。