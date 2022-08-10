-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [快速上手](#快速上手.md)
-   [高级参考](#高级参考.md)

<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：** 深度学习 

**版本（Version）：1.0**

**修改时间（Modified） ：2022.08.08**

**大小（Size）：4.58MB**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt、pbtxt、meta**

**精度（Precision）：**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Research**

**描述（Description）：基于TensorFlow框架验证提出的concrete random variables – continuous relaxations of discrete random
variables** 

<h2 id="概述.md">概述</h2>

引入了具体的随机变量——离散随机变量的连续松弛。具体分布是一个新的分布族，具有闭合形式密度和简单的重新参数化。
- 参考论文：
    [https://arxiv.org/abs/1611.00712

- 参考实现：

    [Rebar-Tensorflow](https://github.com/tensorflow/models/tree/master/research/rebar) 

- 适配昇腾 AI 处理器的实现：
    [https://gitee.com/yuanshuo111/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/Rebar_ID2016_for_TensorFlow]
  

  ​    


- 通过Git获取对应commit\_id的代码方法如下：

  ```
  git clone {repository_url}    # 克隆仓库的代码
  cd {repository_name}    # 切换到模型的代码仓目录
  git checkout  {branch}    # 切换到对应分支
  git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
  cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```

## 默认配置<a name="section91661242121611"></a>

- 训练超参

  - Batch size: 24
  - Learning rate:0.0003
  - model: SBNDynamicRebar
  - n_layer:2
  - task: sbn


## 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 否     |
| 混合精度   | 否       |
| 并行数据   | 是       |



<h2 id="训练环境准备.md">训练环境准备</h2>

1. 硬件环境准备请参见各硬件产品文档"[驱动和固件安装升级指南]( https://support.huawei.com/enterprise/zh/category/ai-computing-platform-pid-1557196528909)"。需要在硬件设备上安装与CANN版本配套的固件与驱动。

2. 

- 数据集准备



1, 模型训练使用MNIST数据集，数据集请用户自行获取，也可通过如下命令行获取。

```bash
$ python download.py --dataset MNIST

2, 放入模型目录下，在训练脚本中指定数据集路径，可正常使用。

## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。

- 启动训练之前，首先要配置程序运行相关环境变量。

  环境变量配置信息参见：

     [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

- 单卡训练 

  1. 在Pycharm当中使用Modelarts插件进行配置。

     在Pycharm当中使用Modelarts插件进行配置，具体配置如下所示:
     
     Ai engine: Ascend-Powered-Engine   tensorflow_1.15-cann_5.0.2-py_37-euler_2.8.3-aarch64
     Boot file path设置为: D:\REBAR\rebarnpu\rebar_npu_20220103120007\acc.py
     Code Directory设置为: D:\REBAR\rebarnpu\rebar_npu_20220103120007
     OBS Path设置为对应项目的工作目录，此项目为：/rebar-ysnpu/train_dir/
     Data Path in OBS设置为OBS当中存放数据的目录,此项目为：/rebar-ysnpu/data/
     其中.代表当前工作目录。
     ```
  
     启动训练，在Modelarts当中单击Apply and Run即可进行训练。
 
  2. 在TestUser01裸机上进行训练：

    
    
    环境准备：export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64/:$LD_LIBRARY_PATH
             source /usr/local/Ascend/ascend-toolkit/set_env.sh
             export LD_LIBRARY_PATH=/usr/local/Ascend/nnae/5.0.4.alpha002/fwkacllib/lib64/:$LD_LIBRARY_PATH
             source   /usr/local/Ascend/tfplugin/set_env.sh
             source /usr/local/Ascend/nnae/set_env.sh
 
    训练步骤：cd Pycode/rebar_npu   python3.7 rebar_train.py --hparams="model=SBNDynamicRebar,learning_rate=0.001,n_layer=2,task=sbn"

  
- 验证。

  本代码验证和测试阶段都在训练脚本中。

- 数据集准备。

  数据集要求如下：

  1. 获取数据。

     如果要使用自己的数据集，需要将数据集放到data_url对应目录下。参考代码中的数据集存放路径如下：

     - 训练集：'/data/MNIST'
     - 测试集：'/data/MNIST'
     
  2. 准确标注类别标签的数据集。
  
  3. 数据集每个类别所占比例大致相同。

- 模型训练。

  参考“模型训练”中训练步骤。

- 模型评估。

  参考“模型训练”中验证步骤。

<h2 id="训练结果.md">训练结果</h2>

- 精度结果比对:

GPU型号：Tesla V100-SXM2-16GB
NPU型号：昇腾910

  训练：
|精度指标项|论文发布|GPU实测|NPU实测|
|---|---|---|---|
|密度估计|102.3|98.99532|102.55647|
  测试：
|精度指标项|论文发布|GPU实测|NPU实测|
|---|---|---|---|
|密度估计|102.1|101.83852|103.31268|


## 脚本参数<a name="section6669162441511"></a>

```
--data_path              数据集路径，默认：path/data
--batch_size             每个NPU的batch size，默认：24
--learing_rata           初始学习率，默认：0.0003
--steps                  2000000
```



## 训练过程<a name="section1589455252218"></a>


1. 通过“模型训练”中的训练指令启动网络训练。

2. 参考脚本的模型存储路径为:/home/TestUser01/Pycode/rebar_npu/root/rebar/data/output (裸机TestUser01）

3. NPU训练过程部分打屏信息如下:
'''
Step 2000610: [-102.55647       0.66552424    1.            0.962311      0.9606392
    0.94565284    0.9456387     1.            1.            1.
    1.        ]
-5.152364730834961

Test 2000610: [ -98.39359    -103.31268       0.7060462     1.            0.96359265
    0.9600566     0.94702345    0.9468977     1.            1.
    1.            1.        ]
...
4,GPU训练过程部分打屏信息如下：
'''
Step 2000640: [-98.99532      4.835244     1.0000001    0.96947265   0.96826416
   0.950561     0.95059884   1.           1.0000014    1.
   0.9999999 ]
-6.6486406326293945

Test 2000640: [ -96.69081    -101.83852       4.675375      1.0000001     0.9665527
    0.9636474     0.95117027    0.9502216     1.            1.0000014
    1.            0.99999994]
'''
## 数据集地址
OBS地址：
obs://rebar-ysnpu/data/

官方网址：
http://yann.lecun.com/exdb/mnist/ 



