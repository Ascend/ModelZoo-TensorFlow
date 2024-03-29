- [基本信息](#基本信息.md)
- [概述](#概述.md)
- [训练环境准备](#训练环境准备.md)
- [快速上手](#快速上手.md)
- [迁移学习指导](#迁移学习指导.md)
- [高级参考](#高级参考.md)

<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Super Resolution**

**版本（Version）：1.1**

**修改时间（Modified） ：2022.04.21**

**大小（Size）：712K**

**框架（Framework）：TensorFlow_2.6.2**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Research**

**描述（Description）：基于TensorFlow2.X框架的图像超分辨率重建网络训练代码**


<h2 id="概述.md">概述</h2>

## 简述<a name="section194554031510"></a>

    由Shi在2016年提出的ESPCN（Efficient Sub-Pixel CNN）是一种在给定低分辨率版本的情况下重建图像的高分辨率版本的网络模型。它利用高效的“亚像素卷积”层，学习一组图像放大滤波器。


  - 参考论文：

    https://arxiv.org/abs/1609.05158(https://arxiv.org/abs/1609.05158)

  - 参考实现：
    https://github.com/keras-team/keras-io/blob/master/examples/vision/super_resolution_sub_pixel.py


  - 适配昇腾 AI 处理器的实现：

    https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow2/built-in/keras_sample/cv/super_resolution_sub_pixel_ID2541_for_TensorFlow2.X

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
    -  Interpolation,用简单的三次样条插值进行初步的上采样，然后进行学习非线性映射
    -  deconvolution,在最后的上采样层，通过学习最后的deconvolution layer。但deconvolution本质上是可以看做一种特殊的卷积，理论上后面要通过stack filters才能使得性能有更大的提升。
    -  亚像素卷积sub-pixel Layer，跟常规的卷积层相比,其输出的特征通道数为r^2，其中r为缩放倍数

-   训练超参（单卡）：
    -   Batch size: 8
    -   crop_size: 300
    -   upscale_factor: 3
    -   Train epoch: 100


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

1. 模型训练使用BSDS500数据集，该数据集包含200张训练图，100张验证图，200张测试图；所有真值用.mat文件保存，包含segmentation和boundaries，每张图片对应真值有五个，为5个标注的真值，训练时真值可采用平均值或者用来扩充数据，评测代码中会依次对这五个真值都做对比。
2. 创建训练和验证数据集image_dataset_from_directory。
3. 重新缩放图像以获取 [0, 1] 范围内的值。
4. 裁剪和调整图像大小，将图像从 RGB 颜色空间转换为 YUV 颜色空间。

## 模型训练<a name="section715881518135"></a>
- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练。
  
    1. 启动训练之前，首先要配置程序运行相关环境变量。

       环境变量配置信息参见：

          [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)
    

    2. 单卡训练
       
        2.1 设置单卡训练参数（脚本位于super_resolution_sub_pixel_ID2541_for_TensorFlow2.X/test/train_full_1p.sh），示例如下。
            
        
        ```
        batch_size=8
        #训练step
        train_epochs=100
        #训练epochs
        ```
        
        
        
        2.2 单卡训练指令（super_resolution_sub_pixel_ID2541_for_TensorFlow2.X/test） 

        ```
        于终端中运行export ASCEND_DEVICE_ID=0 (0~7)以指定单卡训练时使用的卡
        bash train_full_1p.sh --data_path=xx
        数据集应为h5类型，配置data_path时需指定为datasets这一层，例：--data_path=/home/datasets
        ├─datasets
           ├─BSR
             ├─bench
             ├─BSDS500
             ├─documentation
   
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
    ├── super_resolution_sub_pixel.py            //网络结构定义脚本
    ├── test
    |    |—— train_full_1p.sh                    //单卡训练脚本
    |    |—— train_performance_1p.sh             //单卡训练脚本

 
## 脚本参数<a name="section6669162441511"></a>

```
batch_size                                       训练batch_size
epochs                                           训练epoch数
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
