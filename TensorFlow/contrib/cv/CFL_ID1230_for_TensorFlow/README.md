# 基本信息

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Computer Vision**

**版本（Version）：1.1**

**修改时间（Modified）：2022.03.09**

**大小（Size）：117KB**

**框架（Framework）：Tensorflow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Research**

**描述（Description）：基于Tensorflow框架对360°全景图片实现3D布局恢复的测试代码**

# 模型概述

CFL模型是CFL: End-to-End Layout Recovery from 360 Images论文的Tensorflow实现，该论文的核心思想是使用StdConvs模型和EquiConvs模型分别在360°全景图片上实现3D布局恢复，并生成边图和角图。需要注意的是，此脚本是使用了StdConvs模型。

- 参考论文

  [Corners for Layout: End-to-End Layout Recovery from 360 Images (cfernandezlab.github.io)](https://cfernandezlab.github.io/CFL/)

- 参考实现

  [GitHub - cfernandezlab/CFL: Tensorflow implementation of our end-to-end model to recover 3D layouts. Also with equirectangular convolutions!](https://github.com/cfernandezlab/CFL)

# 默认配置

- 测试数据预处理（以SUN360测试集为例，仅作为用户参考示例）
  - 图像的输入尺寸：128×256
  - 图像的输入格式：jpg
- 测试超参
  - Batch size：16
  - Test epoch：1
  - Test step：72

# 支持特性

| 特性列表 | 是否支持 |
| :------: | :------: |
|  分布式  |    否    |
| 混合精度 |    是    |

# 混合精度

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

脚本已默认开启混合精度，设置precision_mode参数的脚本参考如下。

```python
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
```

# 环境准备

- 硬件环境准备请参见各硬件产品文档"[驱动和固件安装升级指南](https://gitee.com/link?target=https%3A%2F%2Fsupport.huawei.com%2Fenterprise%2Fzh%2Fcategory%2Fai-computing-platform-pid-1557196528909)"，需要在硬件设备上安装与CANN版本配套的固件与驱动。
- 宿主机上需要安装Docker并登录[Ascend Hub中心](https://gitee.com/link?target=https%3A%2F%2Fascendhub.huawei.com%2F%23%2Fdetail%3Fname%3Dascend-tensorflow-arm)获取镜像。
- 安装必要的python依赖'pip install -r requirements.txt'


# 快速上手

模型测试之前的准备工作：模型使用SUN360数据集和CFL模型训练得到的ckpt文件（见参考实现），数据集和ckpt文件请用户自行获取。

# 模型测试

- 单击“立即下载”，并选择合适的下载方式下载源码包。

- 启动测试之前，首先要配置程序运行相关环境变量。环境变量配置信息参见：[Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/%E5%85%B6%E4%BB%96%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)

- 单卡测试

  - 配置参数

    首先在脚本test/train_full_1p.sh中，配置data_path、output_path等参数，请用户根据实际路径配置data_path和output_path，或者在启动测试的命令行中以参数形式下发。

    ```python
    batch_size=16
    data_path=./data_weights
    output_path=./output
    ```

  - 启动测试

    启动单卡测试（脚本为test/train_full_1p.sh）

    `bash test/train_full_1p.sh --data_path=./data_weights --output_path=./output`

# 测试结果

- 精度结果对比

  - EDGES

    | 精度指标项 | 论文发布 | GPU实测 | NPU实测 |
    | :--------: | :------: | :-----: | :-----: |
    |    IoU     |  0.575   |  0.588  |  0.583  |
    |  Accuracy  |  0.931   |  0.933  |  0.931  |
    | Precision  |  0.789   |  0.782  |  0.818  |
    |   Recall   |  0.667   |  0.691  |  0.661  |
    |  f1 score  |  0.722   |  0.733  |  0.730  |

  - CORNERS

    | 精度指标项 | 论文发布 | GPU实测 | NPU实测 |
    | :--------: | :------: | :-----: | :-----: |
    |    IoU     |  0.460   |  0.465  |  0.457  |
    |  Accuracy  |  0.974   |  0.974  |  0.974  |
    | Precision  |  0.887   |  0.872  |  0.885  |
    |   Recall   |  0.488   |  0.498  |  0.484  |
    |  f1 score  |  0.627   |  0.632  |  0.624  |



# 高级参考

##### 文件说明

```python
|--Models
 	 |--__init__.py           //网络初始化
     |--CFL_StdConvs.py       //网络构建
     |--network.py            //网络结构
|--test
     |--train_full_1p.sh      //单卡全量启动脚本
|--License                    //声明
|--README.md                  //代码说明文档
|--config.py                  //参数设置文件
|--modelarts_entry_acc.py     //拉起测试文件
|--modelzoo_level.txt         //网络进度
|--requirements.txt           //python依赖列表
|--test_CFL.py                //网络测试代码
|--output                     //测试结果存放路径
|--data_weights               //数据集和ckpt文件存放路径
     |--Datasets
          |--SUN360
               |--test
                    |--CM_gt
                         |--pano_0b9db1eaf8b73158dd047b8f810cf0cc_CM.jpg
                         ...
                         |--pano_azzfywvfwnlpcl_CM.jpg
                    |--EM_gt
                         |--pano_0b9db1eaf8b73158dd047b8f810cf0cc_EM.jpg
                         ...
                         |--pano_azzfywvfwnlpcl_EM.jpg
                    |--RGB
                         |--pano_0b9db1eaf8b73158dd047b8f810cf0cc.jpg
                         ...
                         |--pano_azzfywvfwnlpcl.jpg
```

##### 脚本参数

```python
--batch_size        每个NPU的batch size,默认:16
--data_path         数据集路径,默认:./data_weights
--output_path       结果输出路径,默认:./output
```
