# STNet_ID2360_for_TensorFlow

## 目录
-   [基本信息](#基本信息)
-   [概述](#概述)
-   [训练环境准备](#训练环境准备)
-   [快速上手](#快速上手)
-   [高级参考](#高级参考)

## 基本信息
-   发布者（Publisher）：Huawei
-   应用领域（Application Domain）： Computer Vision
-   版本（Version）：1.1
-   修改时间（Modified） ：2021.12.04
-   大小（Size）：888K
-   框架（Framework）：TensorFlow 1.15.0
-   模型格式（Model Format）：ckpt
-   精度（Precision）：Mixed
-   处理器（Processor）：昇腾910
-   应用级别（Categories）：Official
-   描述（Description）：使用Tansformer实现的空间变换网络

<h2 id="概述">概述</h2>


- 参考论文：

    https://arxiv.org/abs/1506.02025

- 参考实现：

    https://github.com/daviddao/spatial-transformer-tensorflow

- 适配昇腾 AI 处理器的实现：
    
        
  https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/STNet_ID2360_for_TensorFlow

- 通过Git获取对应commit\_id的代码方法如下：
    
    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    ```

## 默认配置<a name="section91661242121611"></a>

- 训练数据集预处理（以MNIST训练集为例，仅作为用户参考示例）：

    - 图像的输入尺寸为40*40

- 训练超参

  - Batch size: 100
  - embedding dim
  - Train epoch: 500
  - Optimizer: NPULossScaleOptimizer

    
## 支持特性<a name="section1899153513554"></a>

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 否    |
| 混合精度  | 是    |
| 并行数据  | 否    |

## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。




<h2 id="快速上手">快速上手</h2>

- 数据集准备

    数据集使用MNIST手写数据集，obs链接：
    obs://stnet-id2360/dataset/

## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。

- 启动训练之前，首先要配置程序运行相关环境变量。
### NPU
  环境变量配置信息参见：

     [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)

- 单卡训练 

  1. 配置训练参数。

     首先在脚本test/train_full_1p.sh中，配置batch_size等参数

     ```
        batch_size=100
     ```

  2. 启动训练。

     启动单卡训练 （./test/train_full_1p.sh）

     ```
     bash train_full_1p.sh --data_path=./dataset/ --output_path=./output
     ```
<h2 id="训练结果.md">训练结果</h2>

- 精度结果比对

|精度指标项|论文发布|GPU实测|NPU实测|
|---|---|---|---|
|ACC|0.961|0.962|0.937|

- 性能结果比对  

|性能指标项|GPU实测|NPU实测|
|---|---|---|
|FPS|0.78|1.94|
<h2 id="高级参考">高级参考</h2>

## 脚本和示例代码<a name="section08421615141513"></a>

```
├── LICENSE
├── README.md
├── inference.py
├── modelarts_entry_acc.py
├── modelarts_entry_perf.py
├── modelzoo_level.txt
├── ops_info.json
├── requirements.txt
├── stnet.py                                 //网络模型
├── test
│   ├── train_full_1p.sh               //单卡全量训练启动脚本
│   └── train_performance_1p.sh        //单卡训练验证性能启动脚本
├── tf_utils.py
└── train.py


```


