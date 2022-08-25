- [基本信息](#基本信息.md)
- [概述](#概述.md)
- [训练环境准备](#训练环境准备.md)
- [快速上手](#快速上手.md)
- [迁移学习指导](#迁移学习指导.md)
- [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Image Classification**

**版本（Version）：1.1**

**修改时间（Modified） ：2022.8.25**

**大小（Size）：56MB**

**框架（Framework）：TensorFlow_1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的VisionTransformer图像分类网络训练代码**

<h2 id="概述.md">概述</h2>

## 简述<a name="section194554031510"></a>

当前Transformer模型被大量应用在NLP自然语言处理当中，而在计算机视觉领域，Transformer的注意力机制attention也被广泛应用，比如Se模块，CBAM模块等等注意力模块，这些注意力模块能够帮助提升网络性能。而VisionTransformer展示了不需要依赖CNN的结构，也可以在图像分类任务上达到很好的效果，并且也十分适合用于迁移学习。

- 参考论文：

  [http://xxx.itp.ac.cn/pdf/2010.11929.pdf](An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale)

- 参考实现：

  https://github.com/emla2805/vision-transformer

- 适配昇腾 AI 处理器的实现：
  
  https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/VisionTransformer_ID1217_for_Tensorflow

- 通过Git获取对应commit\_id的代码方法如下：
  
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    

## 默认配置<a name="section91661242121611"></a>

-   训练超参（单卡）：
  - Batch size: 4
  - Learning rate(LR): 0.001
  - Optimizer: Adam
  - Train epoch: 1


## 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 否       |
| 混合精度   | 是       |
| 数据并行   | 否       |


## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

拉起脚本中，传入--precision_mode='allow_mix_precision'

```
 ./train_full_1p.sh --help

parameter explain:
    --precision_mode         precision mode(allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision)
    --over_dump                  if or not over detection, default is False
    --data_dump_flag         data dump flag, default is False
    --data_dump_step             data dump step, default is 10
    --profiling                  if or not profiling for performance debug, default is False
    --data_path                  source data of training
    -h/--help                    show help message
```

相关代码示例:

```
config = tf.ConfigProto()
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True  # 在昇腾AI处理器执行训练
custom_op.parameter_map["precision_mode"].s=tf.compat.as_bytes("allow_mix_precision")

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

1、用户自行准备好数据集（数据集链接obs://cann-id1217/dataset/）

2、VisionTransformer训练的模型及数据集可以参考"简述 -> 参考实现"


## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练。

    - 启动训练之前，首先要配置程序运行相关环境变量。

      环境变量配置信息参见：

      [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

    - 单卡训练

          1. 配置训练参数
        
             首先在脚本train_full_1p.sh中，配置训练数据集路径和checkpoint保存路径，请用户根据实际路径配置，示例如下所示：
        
             ```
             # 路径参数初始化
              --data_dir=${data_path} 
              --model_dir=${output_path} 
              --num_cells=6 
              --image_size=224 
              --num_epochs=35 
              --train_batch_size=64 
              --eval_batch_size=64 
              --lr=2.56 
              --lr_decay_value=0.88 
              --lr_warmup_epochs=0.35 
              --mode=train_and_eval 
              --iterations_per_loop=1251  
             ```
        
          2. 启动训练（脚本为./test/train_full_1p.sh） 
        
             ```
             bash train_full_1p.sh --data_path
             ```
          3. 训练精度结果

             ```
            |       | GPU   | NPU   |
            |-------|-------|-------|
            | ACC | 0.8709 | 0.871 |
             ```

<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码

```
|--LICENSE
|--README.md									
|--fusion_switch.cfg
|--modelarts_entry_acc.py
|--modelarts_entry_perf.py
|--modelzoo_level.txt 									
|--requirements.txt		   						
|--vit_allpipeline_fusion_accelerate.py
|--vit_allpipeline_performance.py
|--test			           						
|	|--train_full_1p.sh
|	|--train_performance_1p.sh
|--vit_keras
|       |--layers.py
|       |--utils.py
|       |--vit.py
```

## 脚本参数<a name="section6669162441511"></a>

```
--data_path
--output_path
--learning_rate
--batch_size
```

## 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡网络训练。模型存储路径为${cur_path}/output/$ASCEND_DEVICE_ID，包括训练的log以及checkpoints文件。以单卡训练为例，loss信息在文件${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。