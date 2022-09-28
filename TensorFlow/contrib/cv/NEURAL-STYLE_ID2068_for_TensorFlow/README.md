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

**大小（Size）：131KB**

**框架（Framework）：TensorFlow_1.15**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架神经样式处理网络训练代码**

<h2 id="概述.md">概述</h2>

## 简述<a name="section194554031510"></a>

在美术，特别是绘画中，人类已经掌握了创造独特的技能，通过在图像的内容和风格之间构成复杂的相互作用，视觉体验。到目前为止，这个过程的算法基础是未知的，不存在具有类似能力的人工系统。然而，在视觉感知的其他关键领域，如物体和人脸识别近人的表现最近被一类生物学上的被称为深度神经网络的启发视觉模型。我们介绍一个基于深度神经网络的人工系统，可创建艺术图像具有高感知质量。该系统使用神经表示来分离和重组任意图像的内容和风格，提供神经用于创建艺术图像的算法。此外，鉴于性能优化的人工神经网络和生物视觉，我们的工作提供了一条通往人类如何创造和感知艺术形象的算法理解的道路。

- 参考论文：
  
  [https://arxiv.org/pdf/1508.06576v2.pdf](A Neural Algorithm of Artistic Style)

- 参考实现：

  https://github.com/anishathalye/neural-style

- 适配昇腾 AI 处理器的实现：
  
  https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/NEURAL-STYLE_ID2068_for_TensorFlow

- 通过Git获取对应commit\_id的代码方法如下：
  
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    

## 默认配置<a name="section91661242121611"></a>

-   训练超参（单卡）：
    - CONTENT_WEIGHT = 5e0
    - CONTENT_WEIGHT_BLEND = 1
    - STYLE_WEIGHT = 5e2
    - TV_WEIGHT = 1e2
    - STYLE_LAYER_WEIGHT_EXP = 1
    - LEARNING_RATE = 1e1
    - BETA1 = 0.9
    - BETA2 = 0.999
    - EPSILON = 1e-08
    - STYLE_SCALE = 1.0
    - ITERATIONS = 1000
    - VGG_PATH = 'imagenet-vgg-verydeep-19.mat'
    - POOLING = 'max'


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

1、数据集链接自行获取https://github.com/anishathalye/neural-style/tree/master/examples

2、NEURAL-STYLE训练的模型及数据集可以参考"简述 -> 参考实现"


## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练。

    - 启动训练之前，首先要配置程序运行相关环境变量。

      环境变量配置信息参见：

      [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)

    - 单卡训练


        1.首先在脚本test/train_full_1p.sh中, 训练需要根据安装教程，配置输入与输出的路径。配置训练数据集路径，请用户根据实际路径配置，数据集参数如下所示：

             ```

            --content=${data_path}/1-content.jpg 
            --styles=${data_path}/1-style.jpg 
            --output=${output_path}/11.jpg 
            --network=${data_path}/imagenet-vgg-verydeep-19.mat 
            --iterations=${train_steps} 
            --print-iterations 1 
            --progress-write 
            --progress-plot

            ```

        2.启动训练
        
             启动单卡训练  
        
             ```
             bash train_full_1p.sh
             ```
           


<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码

```
|--LICENSE
|--README.md                                                      #说明文档									
|--neural_style.py                                            
|--modelarts_entry_acc.py
|--modelarts_entry_perf.py
|--modelzoo_level.txt									
|--requirements.txt                                               #所需依赖
|--stylize.py
|--vgg.py                                                   
|--test			           	                          #训练脚本目录
|	|--train_full_1p.sh
|	|--train_performance_1p.sh
```

## 脚本参数<a name="section6669162441511"></a>

```
--content <content file> 
--styles <style file> 
--output <output file>
--checkpoint-output  
--checkpoint-iterations  #保存检查点图像。
--iterations             #更改迭代的次数(默认为1000)。500-2000 次迭代似乎会产生不错的结果。
--content-weight 
--style-weight
--learning-rate 
--style-layer-weight-exp #调整样式转换的“抽象”程度。
--content-weight-blend   #指定内容传输层的系数。默认值1.0，风格转换尝试保留更细致的内容细节。值应该在[0.0,1.0]。
--pooling                #允许去选择使用平均池化层还是最大池化层，原始 VGG 使用最大池化，但风格迁移论文建议将其替换为平均池化。
--preserve colors        #保留内容图颜色选项
```

## 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡，8卡网络训练。模型存储路径为${cur_path}/output/$ASCEND_DEVICE_ID，包括训练的log以及checkpoints文件。以1卡训练为例，loss信息在文件${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。