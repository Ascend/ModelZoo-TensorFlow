-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [快速上手](#快速上手.md)
-   [训练结果](#训练结果.md)
-   [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Computer Vision**

**修改时间（Modified） ：2022.11.6**

**框架（Framework）：TensorFlow 1.15.0**

**描述（Description）：基于TensorFlow框架对高清图片重建相应的超分辨率图片的训练代码** 

<h2 id="概述.md">概述</h2>

```
SRFBN是采取反馈连接来提高重建超分辨率图片效果的网络模型
```
- 参考论文：

    https://arxiv.org/abs/1903.09814v2

- 参考实现：

    https://github.com/turboLIU/SRFBN-tensorflow/blob/master/train.py

## 默认配置<a name="section91661242121611"></a>

- 训练数据集预处理：

  - 图像的输入尺寸为64*64
- 测试数据集预处理：

  - 图像的输入尺寸为64*64
- 训练超参

  - Batch size： 1
  - Train epoch: 1000


<h2 id="快速上手.md">快速上手</h2>

- 数据集准备
1. 模型训练使用DIV2K数据集。

## 模型训练<a name="section715881518135"></a>

- 单卡训练 

  1. 配置训练参数。

     首先在脚本test/train_performance_1p.sh中，配置batch_size、epochs、data_path等参数，请用户根据实际路径配置data_path，或者在启动训练的命令行中以参数形式下发。

     ```
      batch_size=1
      epochs=1000
      data_path="../DIV2K/DIV2K_train_HR"
     ```
     
  2. 启动训练。

     启动单卡训练 （脚本为SRFBN_for_TensorFlow/test/train_performance_1p.sh） 

     ```
     bash train_performance_1p.sh --data_path=../DIV2K/DIV2K_train_HR
     ```

<h2 id="训练结果.md">训练结果</h2>

- 精度结果比对

| 精度指标项 | GPU实测     | NPU实测     |
| ---------- | ----------- | ----------- |
| PSNR       | 6.706763287 | 5.831956861 |

- 性能结果比对  

| 性能指标项 | GPU实测        | NPU实测        |
| ---------- | -------------- | -------------- |
| FPS        | 3.358950029841 | 4.976489075014 |


<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码<a name="section08421615141513"></a>

```
├── Basic_Model.py                 //基本模型代码                
├── README.md                      //代码说明文档
├── config.py                      //模型配置代码
├── PreProcess.py                  //数据预处理代码
├── psnr_ssim.py                   //图像质量评估代码
├── requirements.txt               //训练python依赖列表
├── SRFBN_model.py                 //SRFBN网络模型代码
├── test.py                        //测试代码
├── traditional_blur.py            //图像模糊处理代码
├── train.py                       //训练代码
├── test 
│    ├──train_performance_1p.sh              //单卡训练验证性能启动脚本

```

## 脚本参数<a name="section6669162441511"></a>

```
--data_path              数据集路径，默认：path/data
--batch_size             每个NPU的batch size，默认：1
--epochs                 训练epcoh数量，默认：1000
```

## 训练过程<a name="section1589455252218"></a>

1.  通过“模型训练”中的训练指令启动单卡训练。