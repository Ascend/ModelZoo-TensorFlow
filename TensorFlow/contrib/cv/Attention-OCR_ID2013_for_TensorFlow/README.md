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

**大小（Size）：165139124KB**

**框架（Framework）：TensorFlow_1.15**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的Attention-OCR自然场景文本检测识别网络训练代码**

<h2 id="概述.md">概述</h2>

## 简述<a name="section194554031510"></a>

Attention-OCR是一个基于卷积神经网络CNN、循环神经网络RNN以及一种新颖的注意机制的自然场景文本检测识别网络

- 参考论文：
  
  [https://arxiv.org/abs/1704.03549][Attention-based Extraction of Structured Information from Street View Imagery]

- 参考实现：

  https://github.com/tensorflow/models/tree/master/research/attention_ocr

- 适配昇腾 AI 处理器的实现：
  
  https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/Attention-OCR_ID2013_for_TensorFlow

- 通过Git获取对应commit\_id的代码方法如下：
  
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    

## 默认配置<a name="section91661242121611"></a>

-   训练超参（单卡）：
    - Batch size： 32
    - dataset_dir
    - train_log_dir                        
    - checkpoint_inception
    - max_number_of_steps
    - log_interval_steps

## 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 是       |
| 混合精度   | 是      |
| 数据并行   | 是      |


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

1、模型训练使用FSNS数据集，数据集请用户自行获取。

2、Attention-OCR训练的模型及数据集可以参考"简述 -> 参考实现"


## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练。

    - 启动训练之前，首先要配置程序运行相关环境变量。

      环境变量配置信息参见：

      [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

    - 单卡训练


        1. 配置训练参数
        
          首先在脚本test/train_full_1p.sh中，配置batch_size、steps、epochs、data_path等参数，请用户根据实际路径配置data_path，或者在启动训练的命令行中以参数形式下发。
        
             执行python3.7 ./NPU_train.py如下参数 
             ```
             --dataset_dir=${data_path}/data/fsns  #数据集路径
             --train_log_dir=${output_path}        #输出路径
             --checkpoint_inception=${ckpt_path} 
             --max_number_of_steps=1000000 
             --log_interval_steps=50000
             ```
             执行python3.7 ./eval.py如下参数
             ```
             --dataset_dir=${data_path}/data/fsns 
             --train_log_dir=${output_path} 
             --Not_on_modelart=False 
             --num_batches=1339
             ```  
        
         2. 启动训练。
        
             ```
             bash test/train_full_1p.sh
             ```
        


<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码

```
├── common_flags.py                          //为训练和测试定义配置
├── data_provider.py						//数据集处理
├── eval.py									//模型评估
├── inception_preprocessing.py				//为Inception网络预处理图像
├──	metrics.py								//模型的质量度量
├──	model.py								//模型构建函数
├── modelzoo_level.txt 						//网络状态描述文件
├── NPU_train.py							//模型NPU训练
├──	sequence_layers.py						//用于字符预测的序列层的各种实现
├── train.py								//模型GPU训练
├── train_testcase.sh						//训练测试用例
├── utils.py								//支持构建Attention-OCR的函数
├── modelarts_entry_acc.py                  //用于在Modelarts上拉起精度测试
├── modelarts_entry_perf.py                 //用于在Modelarts上拉起性能测试
├── test     
│    ├──train_performance_1p.sh             //训练性能入口
│    ├──train_full_1p.sh                    //训练精度入口，包含准确率评估
├── datasets
│    ├──fsns.py                             //读取FSNS数据集的配置
│    ├──fsns_test.py                        //FSNS数据集模块的测试

```

## 脚本参数<a name="section6669162441511"></a>

```
--dataset_dir          #数据集目录
--train_log_dir        #训练日志以及checkpoints存放位置
--checkpoint_inception #用于初始权重的inception位置
--max_number_of_steps  #训练轮数
--log_interval_steps   #保存checkpoints的频率

```

## 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡。模型存储路径为${cur_path}/output/$ASCEND_DEVICE_ID，包括训练的log以及checkpoints文件。以1卡训练为例，loss信息在文件${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。