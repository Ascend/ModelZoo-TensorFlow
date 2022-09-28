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

**修改时间（Modified） ：2022.8.26**

**大小（Size）：7691MB**

**框架（Framework）：TensorFlow_1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：FP32**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的风格迁移训练代码**

<h2 id="概述.md">概述</h2>

## 简述<a name="section194554031510"></a>

UGATIT提出了一种新的无监督图像到图像翻译方法，该方法以端到端的方式结合了一个新的注意力模块和一个新的可学习的归一化函数。注意力模块根据辅助分类器获得的注意力图，引导我们的模型关注区分源域和目标域的更重要区域。与以前的基于注意力的方法不同，我们的模型可以翻译需要整体变化的图像和需要大形状变化的图像。此外，我们新的AdaLIN（自适应层实例标准化）函数帮助我们的注意力引导模型根据数据集灵活控制学习参数在形状和纹理中的变化量。实验结果表明，与现有的固定网络结构和超参数模型相比，该方法具有优越性。

- 参考论文：

  [https://arxiv.org/abs/1907.10830](U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation)

- 参考实现：

  https://github.com/taki0112/UGATIT

- 适配昇腾 AI 处理器的实现：
  
  https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/UGATIT_ID0722_for_TensorFlow

- 通过Git获取对应commit\_id的代码方法如下：
  
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    

## 默认配置<a name="section91661242121611"></a>

-   训练超参（单卡）：

      - data_path
      - epoch  10001
      - iteration
      - batch_size
      - save_freq
      - lr 0.0001
      - output_path
    

## 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 否       |
| 混合精度   | 否       |
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

 UGATIT模型默认不开启混合精度

<h2 id="训练环境准备.md">训练环境准备</h2>

-  硬件环境和运行环境准备请参见《[CANN软件安装指南](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=installation-update)》
-  运行以下命令安装依赖。
```
pip3 install requirements.txt
```
说明：依赖配置文件requirements.txt文件位于模型的根目录

<h2 id="快速上手.md">快速上手</h2>

## 数据集准备<a name="section361114841316"></a>

1、模型训练使用selfie2anime数据集，数据集请用户自行获取

2、数据集放入相应模型目录下，在训练脚本中指定数据集路径，可正常使用

3、UGATIT训练的模型及数据集可以参考"简述 -> 参考实现"


## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练。

    - 启动训练之前，首先要配置程序运行相关环境变量。

      环境变量配置信息参见：

      [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)

    - 单卡训练

          1. 配置训练参数
        
             首先在脚本train_full_1p.sh中，配置训练数据集路径和checkpoint保存路径，请用户根据实际路径配置data_path，epoch，output_path示例如下所示：
        
             ```
             # 路径参数初始化
                --data_path="${data_path}/dataset/selfie2anime"  
                --output_path="./output"
                --epoch=${train_epochs}
                --batch_size=${batch_size}
             ```
        
          2. 启动训练（脚本为./test/train_full_1p.sh） 
        
             ```
             bash train_full_1p.sh --data_path
             ```

          3. 训练精度结果

            |精度指标项|GPU实测|NPU实测|
            |---|---|---|
            |d_loss|110.8|108.7|
           


<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码

```
├── README.md                                 // 代码说明文档
├── main.py                                  // 网络训练/推理入口
├── ops.py                                   //网络层定义
├── UGATIT.py                                //网络定义
├── utils.py                                 
├── requirements.txt                          //依赖列表
├── LICENSE                                   
├── output                                  //输出文件夹
│    ├──checkpoint                          //模型输出
│    ├──logdir                              //日志输出
│    ├──results                             //结果输出
│    └──sample
├── dataset
│    └── selfie2anime
│        ├── trainA
│        ├── trainB
│        ├── testA
│        ├── testB
│        └── init_model
├── test
│    ├── train_performance_1p.sh             //单卡训练验证性能启动脚本
│    └── train_full_1p.sh                    //单卡全量训练启动脚本
```

## 脚本参数<a name="section6669162441511"></a>

```
--num_gpus
--phase
--light
--data_path
--epoch
--iteration
--batch_size
--print_freq
--save_freq
--decay_flag
--decay_epoch
--lr
--GP_ld
--adv_weight
--cycle_weight
--identity_weight
--cam_weight
--gan_type
--smoothing
--ch
--n_res
--n_dis
--img_size
--img_ch
--augment_flag
--output_path

```

## 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡训练。模型存储路径为${cur_path}/output/$ASCEND_DEVICE_ID，包括训练的log以及checkpoints文件。以单卡训练为例，loss信息在文件${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。