- [基本信息](#基本信息.md)
- [概述](#概述.md)
- [训练环境准备](#训练环境准备.md)
- [快速上手](#快速上手.md)
- [迁移学习指导](#迁移学习指导.md)
- [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）： Instance Segmentation**

**版本（Version）：1.1**

**修改时间（Modified） ：2022.02.17**

**大小（Size）：85M**

**框架（Framework）：TensorFlow_2.4.1**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：用于分割的点云系列的深度学习网络**

<h2 id="概述.md">概述</h2>

    点云是一种很重要的几何数据结构，因为它不规则的形式，很多研究都是将它转换成规则的3D体素或者collections,这样的话就会造成大量不必要的数据也会产生一些相应的问题。
    点云最接近原始传感数据，它是坐标的，所以它可以轻易地转换成其他三种数据格式(mesh,体素，深度图)，也可以由其他格式轻易地转成点云
    本文设计了一种新型的网络结构直接作用于点云数据上，并且很好的利用了点云数据的随机排列不变性。PointNet是一个统一的框架可以适用于目标分类，part segmentation, 
    scene semantic parsing.它的性能比肩于甚至好于当前最好的结果，这个网络对于输入扰动和损坏具有很强的健壮性。
  
- 参考论文：
    
   https://arxiv.org/pdf/1612.00593.pdf

- 参考实现：

   https://github.com/keras-team/keras-io/blob/master/examples/vision

- 适配昇腾 AI 处理器的实现：    
    
   https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow2/built-in/cv/image_segmentation/pointnet_segmentation_ID2532_for_TensorFlow2.X

- 通过Git获取对应commit\_id的代码方法如下：
    
      git clone {repository_url}    # 克隆仓库的代码
      cd {repository_name}    # 切换到模型的代码仓目录
      git checkout  {branch}    # 切换到对应分支
      git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
      cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换

## 默认配置

- 主要训练超参（单卡）：
    - batch_size: 32
    - epochs: 60
    - lr: 0.001

## 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 否       |
| 混合精度   | 是       |
| 数据并行   | 否       |

## 混合精度训练

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度


```
  ############维测参数##############
  precision_mode="allow_mix_precision"
  #维持参数，以下不需要修改
  over_dump=False
  if [[ $over_dump == True ]];then
    over_dump_path=$cur_path/test/overflow_dump #此处cur_path为代码根目录
    mkdir -p ${over_dump_path}
  fi
  data_dump_flag=False
  data_dump_step="10"
  profiling=False
  use_mixlist=False
  mixlist_file="ops_info.json"
  fusion_off_flag=False
  fusion_off_file="fusion_switch.cfg"
  ############维测参数##############
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

1. 用户需自行准备 PartAnnotation训练数据集。

2. PartAnnotation数据集目录结构如下：
   
   ```
   train_data/
   └── PartAnnotation
       ├── 02691156
       └── metadata.json

   ```

## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练    
   
    1. 启动训练之前，首先要配置程序运行相关环境变量。

       环境变量配置信息参见：

          [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)
    
    2. 单卡训练 

        2.1 配置train_full_1p.sh脚本中`data_path`（脚本路径pointnet_segmentation_ID2532_for_TensorFlow2.X/test/train_full_1p.sh）,请用户根据实际路径配置，数据集参数如下所示：

            --data_path=train_data
            
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
pointnet_segmentation_ID2532_for_TensorFlow2.X/
├── LICENSE
├── pointnet_segmentation.py
├── ReadME.md
├── requirements.txt
└── test
    ├── train_full_1p.sh
    ├── train_performance_1p_dynamic_eval.sh
    └── train_performance_1p_static_eval.sh

```

## 脚本参数<a name="section6669162441511"></a>

```
--data_path      训练数据集路径
--epochs         训练epoch设置
--batch_size     训练bs设置
--eval_static    训练输入固定shape
                      
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
