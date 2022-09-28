-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [快速上手](#快速上手.md)
-   [训练结果](#训练结果.md)
-   [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：3D Object Reconstruction** 

**版本（Version）：1.2**

**修改时间（Modified） ：2022.2.11**

**大小（Size）：110KB**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的物体三维重建网络训练代码** 

<h2 id="概述.md">概述</h2>

	A Novel Hybrid Ensemble Approach For 3D Object Reconstruction from Multi-View Monocular RGB images for Robotic Simulations. 

- 参考论文：

    https://arxiv.org/pdf/1901.11153.pdf

- 参考实现：

    https://github.com/Ajithbalakrishnan/3D-Object-Reconstruction-from-Multi-View-Monocular-RGB-images

- 适配昇腾 AI 处理器的实现：
  
  
  https://toscode.gitee.com/ascend/modelzoo/pulls/5278
        

## 默认配置<a name="section91661242121611"></a>

- 训练数据集预处理（以ShapeNet训练集为例，仅作为用户参考示例）：

  - 图像的输入尺寸为137*137
  - 图像输入格式：png

- 测试数据集预处理（以ShapeNet验证集为例，仅作为用户参考示例）

  - 图像的输入尺寸为137*137
  - 图像输入格式：png

- 数据集获取：

    http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz
    http://cvgl.stanford.edu/data2/ShapeNetVox32.tgz

- 训练超参

  - Batch size： 2
  - Train epoch: 50
  - Train step: 80900


## 支持特性<a name="section1899153513554"></a>

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 否    |
| 混合精度  | 是    |
| 并行数据  | 是    |

## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

脚本已默认开启混合精度。

<h2 id="训练环境准备.md">训练环境准备</h2>

1.  硬件环境准备请参见各硬件产品文档"[驱动和固件安装升级指南]( https://support.huawei.com/enterprise/zh/category/ai-computing-platform-pid-1557196528909)"。需要在硬件设备上安装与CANN版本配套的固件与驱动。
2.  代码使用的华为镜像为ascend-share/5.0.4.alpha005_tensorflow-ascend910-cp37-euleros2.8-aarch64-training:1.15.0-21.0.2_1224。



<h2 id="快速上手.md">快速上手</h2>

- 数据集准备
1. 模型训练使用ShapeNet数据集，数据集请自行获取。

## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。

- 启动训练之前，首先要配置程序运行相关环境变量。

  环境变量配置信息参见：

     [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)

- 单卡训练 

  1. 配置训练参数。

     首先在脚本main_AttSets.py中，配置batch_size、epochs、total_mv等参数，请用户根据实际路径配置data_path，或者在启动训练的命令行中以参数形式下发。

     ```
      batch_size=2
      total_mv=24
      epochs=50
     ```

  2. 启动训练。

     启动单卡训练 （脚本为Pix2Vox_ID1284_for_TensorFlow/test/train_full_1p.sh） 

     ```
     bash train_full_1p.sh 
     ```

<h2 id="训练结果.md">训练结果</h2>

- 精度结果比对

|精度指标项|论文发布|GPU实测|NPU实测|
|---|---|---|---|
|ACC|NA|0.8184|0.8289|

- 性能结果比对  

|性能指标项|论文发布|GPU实测|NPU实测|
|---|---|---|---|
|FPS|NA|34.12|14.15|


<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码<a name="section08421615141513"></a>

```
├── main_AttSets.py                                  //网络训练代码
├── demo_AttSets.py                                  //网络测试代码
├── binvox_rw.py                                  //体素模型预处理代码
├── export_obj.py                                  //模型加载代码
├── tools.py                                      //自定义工具包
├── voxel.py                                       //体素模型预处理代码
├── npu_train.sh                                       //npu训练脚本
├── README.md                                 //代码说明文档
├── requirements.tx                             //训练python依赖列表
├── test
│    ├──train_full_1p.sh                    //单卡全量训练启动脚本
```

## 脚本参数<a name="section6669162441511"></a>

```
--data_path              数据集路径
--batch_size             每个NPU的batch size，默认：2
--att_lr、ref_lr           学习率，默认：0.0002/0.0001/0.00005
--epochs                 训练epcoh数量，默认：50
```

## 训练过程<a name="section1589455252218"></a>

1.  通过“模型训练”中的训练指令启动单卡卡训练。

2.  参考脚本的模型存储路径为./Model/train_mod。

