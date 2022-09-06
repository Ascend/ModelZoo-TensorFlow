- [基本信息](#基本信息.md)
- [概述](#概述.md)
- [训练环境准备](#训练环境准备.md)
- [快速上手](#快速上手.md)
- [迁移学习指导](#迁移学习指导.md)
- [高级参考](#高级参考.md)

<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Image Classification**

**版本（Version）：1.1**

**修改时间（Modified） ：2022.7.22**

**大小（Size）**_**：324KB**

**框架（Framework）：TensorFlow 2.6**

**模型格式（Model Format）：ckpt**

**精度（Precision）：FP32**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Research**

**描述（Description）：本项目的目标是使用TensorFlow 2.0训练最先进的人脸识别器。所选架构是Ascend 910的改进版本，使用的损失函数是ArcFace**

<h2 id="概述.md">概述</h2>

- 使用keras.optimizers.SGD优化器，在MS1M-ArcFace数据进行图像分类训练。

- 参考论文：

    [https://arxiv.org/pdf/1801.07698.pdf](https://arxiv.org/pdf/1801.07698.pdf)

- 参考实现：

    [https://github.com/deepinsight/insightface](https://github.com/deepinsight/insightface)

- 适配昇腾 AI 处理器的实现：

    [https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow2/built-in/cv/image_classification/face_recognition_TF2_ID3613_for_TensorFlow2.X](https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow2/built-in/cv/image_classification/face_recognition_TF2_ID3613_for_TensorFlow2.X)

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

-   训练超参（单卡）：
    -   Batch size: 16
    -   Train epochs: 5


## 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 否       |
| 混合精度   | 是       |
| 数据并行   | 否       |


## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>
相关代码示例。

```
npu_device.global_options().precision_mode = args.precision_mode
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

1. 数据集请用户自行获取。

## 模型训练<a name="section715881518135"></a>
- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练。

    1. 启动训练之前，首先要配置程序运行相关环境变量。

       环境变量配置信息参见：

          [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)


    2. 单卡训练

        2.1 设置单卡训练参数（脚本位于 train_full_1p.sh），示例如下。

        ```
        #Batch Size
        batch_size=16
        #训练epoch
        train_epochs=5
        ```

        2.2 单卡训练指令（脚本位于train_full_1p.sh）

        ```
        于终端中运行export ASCEND_DEVICE_ID=0 (0~7)以指定单卡训练时使用的卡
        bash train_full_1p.sh --data_path=./data
        数据集应有如下结构（数据切分可能不同）
        dataset
        └── MS1M-ArcFace
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
    ├── train.py                                 //主脚本
    ├── LICENSE
    ├── arcface.py.py                            //依赖函数
    ├── evaluation.py                            //依赖函数
    ├── resnet.py                           	 //依赖函数
    ├── resnet_batchRenorm.py                    //依赖函数
    ├── resnet_groupNorm.py                      //依赖函数
    ├── train_multigpu.py                        //依赖函数
    ├── test
    |    |—— train_full_1p.sh                    //单卡全量训练脚本
    |    |—— train_performance_1p.sh             //单卡性能测试脚本

## 脚本参数<a name="section6669162441511"></a>

```
--data_path=${data_path}                          //训练数据路径
--train_epochs=${train_epochs}                    //训练epochs
```

## 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡训练。
将训练脚本（train_full_1p.sh）中的data_path设置为训练数据集的路径。具体的流程参见“模型训练”的示例。