- [基本信息](#基本信息.md)
- [概述](#概述.md)
- [训练环境准备](#训练环境准备.md)
- [快速上手](#快速上手.md)
- [迁移学习指导](#迁移学习指导.md)
- [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Image Restorationg**

**版本（Version）：1.1**

**修改时间（Modified） ：2022.08.24**

**大小（Size）：25MB**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：FP32**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的用于图像恢复的深度均值偏移先验代码** 

<h2 id="概述.md">概述</h2>

## 简述<a name="section194554031510"></a>

在本文中，作者介绍了一个自然图像先验，它直接表示自然图像分布的高斯平滑版本。 作者将先验包含在图像恢复的公式中，作为贝叶斯估计器，这允许解决噪声盲图像恢复问题。 实验表明先验梯度对应于自然图像分布上的均值偏移向量。 此外，作者使用去噪自编码器学习均值偏移向量场，并将其用于梯度下降方法以执行贝叶斯风险最小化。 论文展示了噪声盲去模糊、超分辨率和去马赛克的竞争结果。

- 参考论文：
  
  [https://arxiv.org/abs/1709.03749](Deep Mean-Shift Priors for Image Restoration)

- 参考实现：

  https://github.com/siavashBigdeli/DMSP-tensorflow   

- 适配昇腾 AI 处理器的实现：
  
  https://gitee.com/ascend/ModelZoo-TensorFlow/blob/master/TensorFlow/contrib/cv/DMSP_ID1290_for_Tensorflow

- 通过Git获取对应commit\_id的代码方法如下：
  
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    

## 默认配置<a name="section91661242121611"></a>

-   训练超参（单卡）：
  - Batch size: 1
  - Gaussian noise levels = 11
  - Learning rate(LR): 0.01
  - momentum : 0.9
  - Optimizer: SGD with Momentum
  - Train epoch: 300 iterations

## 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 否       |
| 混合精度   | 否      |
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

1、请用户自行准备好数据集，包含训练集和测试集两部分，数据集包括BSDS300，包含train和test两部分(obs://neu-error-log/dmsp)

2、DMSP的模型及数据集可以参考"简述 -> 参考实现"


## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练。

    - 启动训练之前，首先要配置程序运行相关环境变量。

      环境变量配置信息参见：

      [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

    - 单卡训练


        1.首先在脚本test/train_performance_1p.sh中, 本模型需要训练训练需要根据安装教程，配置输入与输出的路径。配置训练数据集路径，请用户根据实际路径配置，数据集参数如下所示：

            ```
            --data_url=${data_path}/BSDS300/images/train
            ```

        2.启动训练
        
             启动单卡训练  
        
             ```
             bash train_performance_1p.sh
             ```

        3.performance脚本训练如下
             
             ```
             训练脚本

             python ./src/demo_DMSP.py  

             ```           
        
    
<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码

```
├──LICENSE
├──README.md                                                      #说明文档									                                         
├──modelzoo_level.txt									
├──requirements.txt                                               #所需依赖                                                 
├──test			           	                          #训练脚本目录
|	├──train_full_1p.sh
|	├──train_performance_1p.sh
├── src
│    ├──BSDS300/                              //数据集
│    ├──config.py                            //训练定义
│    ├──DAE.py                               //模型定义
│    ├──DAE_model.py                         //重载模型
│    ├──demo_DMSP.py                         //主程序
│    ├──DMSPDeblur.py                        //先验去噪
│    ├──network.py                           //其他功能函数
│    ├──ops.py                               //算子定义
├── scripts
│    ├──train_dmsp.sh                        
```

## 脚本参数<a name="section6669162441511"></a>

```
--batch_size
--data_url
--num_iter
--denoiser
--sigma_dae
--mu
--alpha
```

## 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡。模型存储路径为${cur_path}/output/$ASCEND_DEVICE_ID，包括训练的log以及checkpoints文件。以1卡训练为例，loss信息在文件${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。