- [基本信息](#基本信息.md)
- [概述](#概述.md)
- [训练环境准备](#训练环境准备.md)
- [快速上手](#快速上手.md)
- [迁移学习指导](#迁移学习指导.md)
- [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Image Synthesis**

**版本（Version）：1.1**

**修改时间（Modified） ：2021.12.30**

**大小（Size）：36M**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Research**

**描述（Description）：基于TensorFlow框架的DCGAN训练代码**

<h2 id="概述.md">概述</h2>

    GAN生成对抗网络，该框架可以教会一个深度学习模型来捕捉训练数据分布，并生成具有同分布的相同数据。GAN最早由lan Goodfellow在2014年首次提出。
    GAN由两个不同的模型组成，一个是生成模型generator，一个是鉴别模型discriminator。其中，generator的作用是产生fake image使其封隔与训练图像相似; discriminator 
    的作用是来判断这个fake image与真正的image是否相同。训练过程中，generator通过产生越来越好的fake image，来不断试图去打败discriminator；同时discriminator也是 
    如此。这个游戏的平衡Equilibrium是当生成器生成看起来像是直接来自训练数据的完美赝品时，判别器总是猜测生成器输出为真或假的概率为50%。
  
- 参考论文：
    
   https://arxiv.org/abs/1511.06434

- 参考实现：

   https://github.com/carpedm20/DCGAN-tensorflow

- 适配昇腾 AI 处理器的实现：    
    
   https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/built-in/cv/image_synthesis/DCGAN_ID2196_for_TensorFlow

- 通过Git获取对应commit\_id的代码方法如下：
    
      git clone {repository_url}    # 克隆仓库的代码
      cd {repository_name}    # 切换到模型的代码仓目录
      git checkout  {branch}    # 切换到对应分支
      git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
      cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换

## 默认配置

- 网络结构

    - 生成模型generator
    - 鉴别模型discriminator

- 主要训练超参（单卡）：
    - epoch: 25
    - learning_rate: 0.0002
    - batch_size: 64

## 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 否       |
| 混合精度   | 是       |
| 数据并行   | 是       |

## 混合精度训练

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度


```
  run_config=tf.ConfigProto();
  custom_op = run_config.graph_options.rewrite_options.custom_optimizers.add();
  custom_op.name = 'NpuOptimizer'
  custom_op.parameter_map["use_off_line"].b = True
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

1. 用户需自行准备训练数据集，如celebA，可参考github源获取。

2. 训练数据集目录结构如下：
   
   ```
   train_data/
   └── data
       └── celebA

   ```

## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练    
   
    1. 启动训练之前，首先要配置程序运行相关环境变量。

       环境变量配置信息参见：

          [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)
    
    2. 单卡训练 

        2.1 配置train_full_1p.sh脚本中`data_path`（脚本路径DCGAN_ID2196_for_TensorFlow/test/train_full_1p.sh）,请用户根据实际路径配置，数据集参数如下所示：

            --data_path=${data_path}
            
        2.2 1p指令如下:

            bash train_full_1p.sh --data_path=train_data

- 验证
    
       ```
       NA

        ```

<h2 id="迁移学习指导.md">迁移学习指导</h2>

- 数据集准备。

    1.  获取数据。
        请参见“快速上手”中的数据集准备。
    
- 模型训练。

    参考“模型训练”中训练步骤。

- 模型评估。

    参考“模型训练”中验证步骤。

<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码

```
DCGAN_ID2196_for_TensorFlow/
├── assets
│   ├── d__hist.png
│   ├── d_hist.png
│   ├── mnist1.png
│   ├── mnist2.png
│   └── mnist3.png
├── average.png
├── DCGAN.png
├── download.py
├── LICENSE
├── main.py
├── model.py
├── modelzoo_level.txt
├── ops.py
├── README.md
├── requirements.txt
├── test
│   ├── train_full_1p.sh
│   └── train_performance_1p.sh
├── utils.py
└── web
    ├── app.py
    ├── css
    │   ├── fakeLoader.css
    │   ├── font-awesome.min.css
    │   └── main.css
    ├── fonts
    │   ├── FontAwesome.otf
    │   ├── fontawesome-webfont.eot
    │   ├── fontawesome-webfont.svg
    │   ├── fontawesome-webfont.ttf
    │   ├── fontawesome-webfont.woff
    │   └── slick.woff
    └── index.html

```

## 脚本参数<a name="section6669162441511"></a>

```
--epoch          训练epoch设置
--dataset        训练训练集类型   
                 
```
## 训练过程<a name="section1589455252218"></a>

1. 通过“模型训练”中的训练指令启动单卡训练。
2. 将训练脚本（train_full_1p.sh）中的data_path设置为训练数据集的路径。具体的流程参见“模型训练”的示例。
3. 模型存储路径为“curpath/output/ASCEND_DEVICE_ID”，包括训练的log文件。
4. 以多卡训练为例，loss信息在文件curpath/output/{ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。

## 推理/验证过程<a name="section1465595372416"></a>

```
 NA

```