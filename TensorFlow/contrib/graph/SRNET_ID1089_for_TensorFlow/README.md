- [基本信息](#基本信息.md)
- [概述](#概述.md)
- [训练环境准备](#训练环境准备.md)
- [快速上手](#快速上手.md)
- [迁移学习指导](#迁移学习指导.md)
- [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Image Generation**

**版本（Version）：1.1**

**修改时间（Modified） ：2022.8.22**

**大小（Size）：78.5MB**

**框架（Framework）：TensorFlow_1.15**

**模型格式（Model Format）：ckpt**

**精度（Precision）：FP32**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的图像生成网络训练代码**

<h2 id="概述.md">概述</h2>

## 简述<a name="section194554031510"></a>

SRNet，由三个模块组成：文本转换模块、背景修复模块和融合模块。文本转换模块更改的文本内容将源图像转换为目标文本，同时保留原始文本样式。背景修补模块擦除原始文本。并使用适当的纹理填充文本区域。融合模块合并来自前两个模块的信息，以及生成编辑后的文本图像。

- 参考论文：
  
  [https://dl.acm.org/doi/pdf/10.1145/3343031.3350929](Editing Text in the Wild)

- 参考实现：

  https://github.com/youdao-ai/SRNet

- 适配昇腾 AI 处理器的实现：
  
  https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/graph/SRNET_ID1089_for_TensorFlow

- 通过Git获取对应commit\_id的代码方法如下：
  
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    

## 默认配置<a name="section91661242121611"></a>

-   训练超参（单卡）：
    - batch-size: 8
    - learning_rate = 1e-4
    - decay_steps = 10000
    - max_iter = 20000
    - save_ckpt_interval = 10000
    - gen_example_interval = 1000

## 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 否       |
| 混合精度   | 否      |
| 数据并行   | 否       |


## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

<h2 id="训练环境准备.md">训练环境准备</h2>

-  硬件环境和运行环境准备请参见《[CANN软件安装指南](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=installation-update)》
-  运行以下命令安装依赖。
```
pip3 install requirements.txt
```
说明：依赖配置文件requirements.txt文件位于模型的根目录

<h2 id="快速上手.md">快速上手</h2>

## 数据集准备<a name="section361114841316"></a>

1、用户需自行生成数据集（方法见https://github.com/youdao-ai/SRNet-Datagen）

2、SRNET的模型及数据集可以参考"简述 -> 参考实现"

## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练。

    - 启动训练之前，首先要配置程序运行相关环境变量。

      环境变量配置信息参见：

      [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

    - 单卡训练


        1.首先在脚本test/train_full_1p.sh中, 训练需要根据安装教程，配置输入与输出的路径。配置训练数据集路径，请用户根据实际路径配置，数据集参数如下所示：

            ```
            --data_dir=${data_path}/v1 
            --output_dir=${output_path}
          
            ```

        2.启动训练
        
             启动单卡训练  
        
             ```
             bash train_full_1p.sh
             ```
        3.精度训练结果
          在原文之中，对生成的图片质量进行评估的主要标准为峰值信噪比(PSNR)和结构相似性(SSIM)。两张图像之间的PSNR和SSIM越高,说明越相似。使用训练得到的结果，我们生成 
          了1000张图片，并计算了它们和对应真值之间PSNR和SSIM，最终得到的结果如下
        
             ```
            |         |   PSNR   | SSIM   |
            | :-----: | :-----: | :-----: | 
            | NPU     | 17.31   | 1.75    | 
             ```             
    

<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码

```
|--LICENSE
|--README.md                                                     									
|--examples			           	                          
|	|--labels
|	|--results                                         
|--modelarts_entry_perf.py
|--modelzoo_level.txt
|--predict.py
|--cfg.py
|--datagen.py
|--loss.py
|--model.py
|--train.py
|--utils.py									
|--requirements.txt                                                                                               
|--test			           	                          
|	|--train_full_1p.sh
|	|--train_performance_1p.sh
```

## 脚本参数<a name="section6669162441511"></a>

```
--learning_rate = 1e-4 # default 1e-3
--decay_rate = 0.9
--decay_steps = 10000
--staircase = False
--beta1 = 0.9 # default 0.9
--beta2 = 0.999 # default 0.999
--test_max_iter = 500
--max_iter = 20000
--show_loss_interval = 50
--write_log_interval = 50
--save_ckpt_interval = 10000
--gen_example_interval = 1000
--pretrained_ckpt_path = None
--train_name = None # used for name examples and tensorboard logdirs, set None to use time
```

## 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡。模型存储路径为${cur_path}/output/$ASCEND_DEVICE_ID，包括训练的log以及checkpoints文件。以1卡训练为例，loss信息在文件${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。