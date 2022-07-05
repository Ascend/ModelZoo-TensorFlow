-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [快速上手](#快速上手.md)
-   [训练结果](#训练结果.md)
-   [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Visual Odometry** 

**版本（Version）：1.0**

**修改时间（Modified） ：2022.7.5**

**大小（Size）：104KB**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的卷积长短期记忆实现视觉里程计训练代码** 

<h2 id="概述.md">概述</h2>


ConvLSTM最早由香港科技大学的团队提出，解决序列图片的时空预测问题。本网络的ConvLSTM结构用于处理车载摄像头序列图片，实现一个视觉里程计。

- 参考论文：

    https://arxiv.org/abs/1506.04214
    https://arxiv.org/abs/1709.08429

- 参考实现：

    https://github.com/giserh/ConvLSTM-2
    https://github.com/Kallaf/Visual-Odometry/blob/master/VisualOdometry.ipynb 

- 适配昇腾 AI 处理器的实现：
    
        
  https://gitee.com/ascend/ModelZoo-TensorFlow/new/master/TensorFlow/contrib/cv/ConvLSTM_ID2358_for_TensorFlow
        


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

  - Batch size： 32
  - Train epoch: 240
  - learing_rata: 0.0001


## 支持特性<a name="section1899153513554"></a>

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 否    |
| 混合精度  | 是    |
| 并行数据  | 是    |

## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

脚本已开启混合精度，设置precision_mode参数的脚本参考如下。

  ```
  custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
  ```

<h2 id="训练环境准备.md">训练环境准备</h2>

1.  硬件环境准备请参见各硬件产品文档"[驱动和固件安装升级指南]( https://support.huawei.com/enterprise/zh/category/ai-computing-platform-pid-1557196528909)"。需要在硬件设备上安装与CANN版本配套的固件与驱动。
2.  所需环境依赖：ubuntu16.04+python3.7.5+tensorflow-gpu1.15.0+opencv+opencv-contrib3.4.2.17+evo1.12.0

<h2 id="快速上手.md">快速上手</h2>

- 数据集准备
1. 模型训练使用KITTI Visual Odometry/SLAM benchmark数据集，数据集请用户自行获取。

## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。

- 启动训练之前，首先要配置程序运行相关环境变量。

  环境变量配置信息参见：

     [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

- 单卡训练 

  1. 配置训练参数。

     训练参数已经默认在脚本中设置，需要在启动训练时指定数据集路径和输出路径

     ```
    parser.add_argument('--datapath')
    parser.add_argument('--outputpath')
    parser.add_argument('--bsize', default=32)
    parser.add_argument('--trajectory_length', default=4)
    parser.add_argument('--lr', default=0.0001)
    parser.add_argument('--train_iter', default=240)
    parser.add_argument('--time_steps', default=1)
     ```

  2. 启动训练。

     启动单卡训练  

     ```
     python Truemain.py --datapath '你的数据集路径' --outputpath '你的输出路径'
     ```

<h2 id="训练结果.md">训练结果</h2>

- 精度结果比对

|精度指标项|论文发布|GPU实测|NPU实测|
|---|---|---|---|
|ACC|xxx|yyy|zzz|

- 性能结果比对  

|性能指标项|论文发布|GPU实测|NPU实测|
|---|---|---|---|
|FPS|XXX|4.5sec/step|3.4sec/step|


<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码<a name="section08421615141513"></a>

```
├── Truemain.py                               //网络训练与测试代码
├── README.md                                 //代码说明文档
├── cell.py                                   //convlstm核
├── test
├── requirements.txt                          //训练python依赖列表
│    ├──modelarts_entry_acc.py                //modelarts训练验证精度脚本
│    ├──modelarts_entry_perf.py               //modelarts训练验证性能脚本

```

## 脚本参数<a name="section6669162441511"></a>

```
--data_path              数据集路径
--batch_size             每个NPU的batch size，默认：32
--learing_rata           初始学习率，默认：0.0001
--steps                  数据集图片跨越步数，默认：1
--train_iter             训练epcoh数量，默认：240
```

## 训练过程<a name="section1589455252218"></a>

1.  通过“模型训练”中的训练指令启动单卡卡训练。

2.  数据集、参考脚本的模型存储路径为用户定义。


