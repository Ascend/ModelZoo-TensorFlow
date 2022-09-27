- [基本信息](#基本信息.md)
- [概述](#概述.md)
- [训练环境准备](#训练环境准备.md)
- [快速上手](#快速上手.md)
- [迁移学习指导](#迁移学习指导.md)
- [高级参考](#高级参考.md)

<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Instance Segmentation**

**版本（Version）：1.1**

**修改时间（Modified） ：2022.04.11**

**大小（Size）：43M**

**框架（Framework）：TensorFlow_2.6.2**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow2.X框架的3D点云采样的图像分类和分割网络训练代码**


<h2 id="概述.md">概述</h2>

## 简述<a name="section194554031510"></a>

点云（point cloud）是一种非常重要的几何数据结构。由于点云的无规律性（irregular format），大部分研究者将点云转换为规律的3D体素网格（3D voxel grids）或者一组不同视角的2D图像。这种转换数据的方式，增加了数据的规模，同时也会带来一系列问题。PointNet是一种可以直接处理点云的神经网络，并且考虑了输入点云序列不变性的特征。PointNet提供了统一的应用架构，可以用于分类（classification），块分割（part segmentation），语义理解（semantic parsing）。尽管网络很简单，但是非常有效。从实验结果上看，它超越了经典的方法，至少也达到同样的水平。理论上，我们进行了分析，包括网络学习了什么，以及当数据被一定程度的干扰后，网络为什么能保持稳定。


  - 参考论文：

    https://arxiv.org/abs/1612.00593(https://arxiv.org/abs/1612.00593)

  - 参考实现：
    https://github.com/keras-team/keras-io/blob/master/examples/vision/pointnet.py(https://github.com/keras-team/keras-io/blob/master/examples/vision/pointnet.py)


  - 适配昇腾 AI 处理器的实现：
    https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow2/built-in/keras_sample/cv/PointNet_ID2913_for_TensorFlow2.X

  - 通过Git获取对应commit\_id的代码方法如下：
    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    ```




## 默认配置<a name="section91661242121611"></a>


-   网络结构
    -  设计最大池化层（对称函数），用于聚合所有点的特征信息
    -  计算全局点云特征向量后，通过将全局特征与每个点特征连接起来，将全局特征反馈给每个点特征。然后我们在合并的点特征的基础上提取新的每点特征——这时，每点特征都能识别局部和全局信息
    -  通过一个小网络(T-net)来预测一个仿射变换矩阵，并直接将这个变换应用到输入点的坐标上。小网络与大网络相似，由点独立特征提取、最大池化和全连接层等基本模块组成。

-   训练超参（单卡）：
    -   Batch size: 32
    -   learning_rate：0.0015
    -   num_point：2048
    -   Train epoch: 250


## 支持特性<a name="section1899153513554"></a>

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 否    |
| 混合精度  | 是    |
| 数据并行  | 否    |

## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>
相关代码示例。

```
  config_proto = tf.ConfigProto(allow_soft_placement=True)
  custom_op = config_proto.graph_options.rewrite_options.custom_optimizers.add()
  custom_op.name = 'NpuOptimizer'
  custom_op.parameter_map["use_off_line"].b = True
  custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
  config_proto.graph_options.rewrite_options.remapping = RewriterConfig.OFF
  session_config = npu_config_proto(config_proto=config_proto)
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

1. 模型训练使用modelnet40_ply_hdf5_2048数据集，即ModelNet40模型训练出的点云数据（HDF5文件类型）。每个点云包含从形状表面均匀采样的 2048 个点。每个云都是零均值并归一化为一个单位球体。
2. 安装 h5py。该代码已在 Ubuntu 14.04 上使用 Python 2.7、TensorFlow 1.0.1、CUDA 8.0 和 cuDNN 5.1 进行了测试。
```
sudo apt-get install libhdf5-dev 
sudo pip install h5py
```
3.log默认情况下，日志文件和网络参数将保存到文件夹中。HDF5 文件中ModelNet40模型的点云将自动下载 (416MB) 到数据文件夹。

## 模型训练<a name="section715881518135"></a>
- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练。
  
    1. 启动训练之前，首先要配置程序运行相关环境变量。

       环境变量配置信息参见：

          [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)
    

    2. 单卡训练
       
        2.1 设置单卡训练参数（脚本位于PointNet_ID2913_for_TensorFlow2.X/test/train_full_1p.sh），示例如下。
            
        
        ```
        batch_size=32
        #训练step
        train_epochs=250
        #学习率
        learning_rate=0.0015
        ```
        
        
        
        2.2 单卡训练指令（PointNet_ID2913_for_TensorFlow2.X/test） 

        ```
        于终端中运行export ASCEND_DEVICE_ID=0 (0~7)以指定单卡训练时使用的卡
        bash train_full_1p.sh --data_path=xx
        数据集应为h5类型，配置data_path时需指定为data这一层，例：--data_path=/home/data
        ├─data
           ├─ply_data_test0.h5*
           ├─ply_data_test_0_id2file.json*
           ├─ply_data_test1.h5*
           ├─ply_data_test_1_id2file.json*
           ├─ply_data_train0.h5*
           ├─ply_data_train_0_id2file.json*
           ├─ply_data_train1.h5*
           ├─ply_data_train_1_id2file.json*
           ├─ply_data_train2.h5*
           ├─ply_data_train_2_id2file.json*
           ├─ply_data_train3.h5*
           ├─ply_data_train_3_id2file.json*
           ├─ply_data_train4.h5*
           ├─ply_data_train_4_id2file.json*
           ├─shape_names.txt*
           ├─test_files.txt*
           ├─train_files.txt*
         
        ```

<h2 id="迁移学习指导.md">迁移学习指导</h2>

- 数据集准备。

    1.  获取数据。
        请参见“快速上手”中的数据集准备
    
- 模型训练

    请参考“快速上手”章节

<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码<a name="section08421615141513"></a>

    ├── README.md                                //说明文档
    ├── requirements.txt                         //依赖
    ├── modelzoo_level.txt                       //状态文件
    ├── provider.py                              //数据集处理脚本
    ├── train.py                                 //网络训练脚本
    ├── models                                   //网络结构定义脚本
         |—— pointnet_cls.py
         |—— pointnet_cls_basic.py
         |—— pointnet_seg.py
         |—— transform_nets.py
    ├── test
    |    |—— train_full_1p.sh                    //单卡训练脚本
    |    |—— train_performance_1p.sh             //单卡训练脚本
    ...
 
## 脚本参数<a name="section6669162441511"></a>

```
batch_size                                       训练batch_size
learning_rate                                    初始学习率
max_epochs                                       最大训练epoch数
num_point                                        每个点云包含从形状表面均匀采样的点数
precision_mode                                   default="allow_mix_precision", type=str,help='the path to save over dump data'
over_dump                                        type=ast.literal_eval,help='if or not over detection, default is False'
data_dump_flag                                   type=ast.literal_eval,help='data dump flag, default is False'
data_dump_step                                   data dump step, default is 10
profiling                                        type=ast.literal_eval help='if or not profiling for performance debug, default is False'
profiling_dump_path                              type=str, help='the path to save profiling data'
over_dump_path                                   type=str, help='the path to save over dump data'
data_dump_path                                   type=str, help='the path to save dump data'
use_mixlist                                      type=ast.literal_eval,help='use_mixlist flag, default is False'
fusion_off_flag                                  type=ast.literal_eval,help='fusion_off flag, default is False'
mixlist_file                                     type=str,help='mixlist file name, default is ops_info.json'
fusion_off_file                                  type=str,help='fusion_off file name, default is fusion_switch.cfg'
auto_tune                                        help='auto_tune flag, default is False'
```

## 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡训练。
将训练脚本（train_full_1p.sh）中的data_path设置为训练数据集的路径。具体的流程参见“模型训练”的示例。
