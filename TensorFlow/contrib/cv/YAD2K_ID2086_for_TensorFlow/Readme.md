# YAD2K(yolov2) 介绍

## 基本信息

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：** 目标检测

**版本（Version）：1.1**

**修改时间（Modified） ：2021.11.2**2

**大小（Size）：74M**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：.h5**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Research**

**描述（Description）：基于TF1.15+keras2.1.3的yolov2复现**

##  概述

Redmon和Farhadi（2017）提出了YOLO9000（也被成为YOLOv2）。与YOLO相比，YOLOv2做出了如下的改进，包括batch normalization，使用高分辨率训练图像、维度聚类（K-means）以及锚框（anchor box）等。backbone使用 Darknet19，对于网络输出其预测的是相对于当前网格的偏移量而不是预测边界框的坐标。

论文中 yolov2 以 40FPS 速度在 VOC 2007 Test集 上实现了78.6mAP。

本项目 GPU上以 69FPS 速度在 VOC 2007 Test集上 以score_threshold=0.05 与iou_threshold=0.6 复现 72.51%mAP 

+ 参考论文：

  [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242)

  




## 默认配置

+ 训练超参
  + Batch size：10
  + Optimizer：Adam
  + Train epoch：100
+ 测试超参
  + eval  score_threshold=0.05
  + eval  iou_threshold=0.6



##  支持特性

| 特效列表 | 是否支持 |
| -------- | -------- |
| 混合精度 | 是       |

### 混合精度训练

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。



## Rquirements

+ Keras 2.3.1
+ TensorFlow 1.15.0
+ opencv-python 4.5.4.58
+ pillow 8.3.2
+ numpy 1.16.2
+ hdf5 1.12.1
+ h5py 2.10.0
+ math
+ shutil
+ time
+ matplotlib
+ os
+ moxing
+ json
+ glob
+ argparse



## 模型训练与评测步骤

### For train

+ 在modelArts上训练，入口地址为 boot_modelarts.py 文件。需将 For test 中的 用于test的部分注释。

+ 通过入口地址文件 执行 训练脚本 run.1p.sh。 

  ```python
      # 用于训练
      bash_header = os.path.join(code_dir, 'run_1p.sh')  # 训练脚本文件
      bash_command = 'bash %s %s' % (bash_header, arg_url)
      print("bash command:", bash_command)
      os.system(bash_command)
  ```

+ 在脚本文件中，执行train代码。 训练使用预训练，训练数据集为2007+2012 trainval数据，其中12520条作为训练集，4000条作为验证集。

  ```sh
  python3.7 ${code_dir}/train.py \
          --dataset=${data_dir} \
          --result=${result_dir} \
          --obs_dir=${obs_url}
  ```

+ 训练的每个阶段的代码保存在 obs 的 result 文件夹中，格式为h5。

### For Test（Detect and Evaluate）

+ 在modelArts上执行测试，入口地址统一 为 boot_modelarts.py 文件。 做test时，需要注释掉 boot_modelarts.py 文件中的 执行训练脚本代码（如上代码）。 将test代码，还原如下：

  ```python
      # 用于看看测试结果 已经预测生成AP等
      detect_bash = os.path.join(code_dir,'detect_1p.sh') #测试脚本文件
      detect_command = 'bash %s %s' % (detect_bash, arg_url)
      print("detect command:",detect_command)
      os.system(detect_command)
  ```

+ 通过入口地址文件 执行 训练脚本 detect_1p.sh。 

  + 若你想查看效果图 请将下面的 代码 还原。 注释后者

    ```sh
    #数据查看
    python3.7 ${code_dir}/detect.py \
            --dataset=${data_dir} \
            --result=${result_dir} \
            --obs_dir=${obs_url}
    ```

  + 若你想做test评估 请将下面的 代码 还原。注释前者

    ```sh
    # 生成 文件夹 等
    python3.7 ${code_dir}/get_info.py \
            --dataset=${data_dir} \
            --result=${result_dir} \
            --obs_dir=${obs_url}
    
    # 画出ap曲线等
    python3.7 ${code_dir}/get_map.py \
            --dataset=${data_dir} \
            --result=${result_dir} \
            --obs_dir=${obs_url}
    ```

### Detect 的效果

detect的效果如下图所示：

![result_000004](C:\Users\kaikai\Desktop\有用的东西\result_000004.jpg)

![result_000166](C:\Users\kaikai\Desktop\有用的东西\result_000166.jpg)

![result_000069](C:\Users\kaikai\Desktop\有用的东西\result_000069.jpg)

### Detect 的效果

+ get_info.py  会在 obs 里得到 accuracy 文件。 其中包含 验证的效果txt detections，以及真实标签txt groundtruths 

+ get_map.py 会根据上面的文件，做出 AP、F1、Precision、Recall 的图像 输出在 obs 中的mapresults 文件中



论文中 作者只提供了 mAP 作为 其模型性能的衡量标准。 

作者在VOC 2007+2012的trainval数据上进行训练，并在 VOC2007 test集上做测试 得到各类的AP值，其mAP为78.3。

该项目同样在VOC 2007+2012的trainval数据上训练，使用25%的数据作为验证集，在VOC2007 test集上做测试。选用收敛后验证集loss最小的作为测试模型，其在test集上的mAP为72.51。

该项目与原论文mAP差10。原作者在论文里并没有描述 其 设置的分类置信度门槛值和IOU门槛值是多少。但其实这对于mAP结果是有影响的。

该项目设置的门槛值为 score_threshold=0.05,  iou_threshold=0.6。


## GPU与NPU 精度与性能比对
- 精度结果比对

|精度指标项|论文发布|GPU实测|NPU实测|
|---|---|---|---|
|mAP|78.6|72.51|72.51|

- 性能结果比对  

|性能指标项|论文发布|GPU实测|NPU实测|
|---|---|---|---|
|FPS|-----|69.14|1.63|

## 测试用例

+ 在modelArts上训练，入口地址为 boot_modelarts.py 文件。需将 For train 和For test 的部分都注释掉。

+ 通过入口地址文件 执行 训练脚本 train_testcase.sh。 

  ```python
      # 用于 自测试用例
      self_batch = os.path.join(code_dir, 'train_testcase.sh')  # 训练 自测试用例 脚本文件
      self_command = 'bash %s %s' % (self_batch, arg_url)
      print("self_command:", self_command)
      os.system(self_command)
  ```

+ 在测试用例脚本文件中，执行整个train和evaluate的过程。 训练使用预训练，训练数据集为部分训练数据，evaluate为部分测试数据。

  ```sh
  python3.7 ${code_dir}/train_testcase.py \
          --dataset=${data_dir} \
          --result=${result_dir} \
          --obs_dir=${obs_url}
  ```

+ 同样测试用例训练的每个阶段的 h5格式权重模型保存在 obs 的 result 文件夹中。后台log 可以看到打印的精度。

## 迁移学习指导

+ 数据集准备

  + 使用VOC2007+2012的数据。 可以使用 data_process 中的 voc_annotation.py文件进行数据处理。其会根据标签（test、train、val）将xml转化为对应的测试、训练、验证文本文件。（其为图片地址+图片所含所有框的框坐标+框宽高+物体类别序号）,注意图片地址要对应自己的地址。整体格式如下图所示：

    ![数据格式](http://kyle-pic.oss-cn-hangzhou.aliyuncs.com/img/datageshi.png)

  + 如果要迁移学习到其他的数据格式，需要自行的制作如上图的格式文本文件。需要注意图片的路径，要自行制作训练集和验证集两部分。

+ 模型参数修改

  + 修改data_process文件下config.py的内容。迁移自己的数据集，需要修改classes数组的内容。理论上需要重新设计anchors，这部分需要迁移者在自己的数据上进行K-means聚类。

+ 加载预训练模型

  + 使用自己的预训练参数文件。（注意代码使用的是.h5的参数文件）

  + modelArts上训练，需要配置obs路径（预训练参数文件需要放在obs内）

    ```python
    premodel = config1.dataset+'/ckpt.h5'
    ```

    

## 高级参考

### 脚本和示例代码（省略测试用例文件）

```
├── data_process
│    ├──check_data.py                         //数据txt文本内容测试，查看内容是否正确
│    ├──config.py                    		  //网络参数配置
│    ├──data_loader.py                        //构建dataset，方便在训练过程中读取数据
│    ├──tool.py                               //文件读取工具包
│    ├──voc_annotation.py                     //所需数据集拆分成训练和测试集两部分
├── nets
│    ├──v2net.py                              //yolov2网络模型
│    ├──yololoss.sh                           //网络输出编码以及损失函数设计
├── logs									  //训练的模型保存地址
├── test_img								  //可以使用里面的图片做一个简单的小detect
├── boot_modelarts.py                         //模型的入口地址
├── ckpt.h5                                   //网络预训练参数模型
├── detect.py                                 //模型预测相关函数
├── detect_1p.sh                              //Detect and Evaluate 脚本文件
├── draw_loss.py                              //绘制训练loss，先把训练结果手动保存在了 												  		train_log.txt中
├── get_info.py                               //生成预测和实际的分类和预测框结果文件
├── get_map.py                                //读取上者的结果文件，生成AP、																	Precison、Recall和F1的图像
├── help_modelarts.py                         //obs与modelArts文件读取和存储函数
├── run_1p.sh                                 //训练 脚本文件
├── train.py                                  //网络训练代码
├── train_log.txt                             //手动保存的训练结果
├── README.md                                 //代码说明文档
```



## 训练过程

训练的参数 可以手动在

```
Epoch 38/100
1/1252 [..............................] - ETA: 6:48 - loss: 1.6335
2/1252 [..............................] - ETA: 6:50 - loss: 4.7053
3/1252 [..............................] - ETA: 6:45 - loss: 3.4914
4/1252 [..............................] - ETA: 6:41 - loss: 3.5281
5/1252 [..............................] - ETA: 6:39 - loss: 3.2594
6/1252 [..............................] - ETA: 6:38 - loss: 3.3860
7/1252 [..............................] - ETA: 6:34 - loss: 3.3317
.......
1243/1252 [============================>.] - ETA: 2s - loss: 2.8483
1244/1252 [============================>.] - ETA: 2s - loss: 2.8478
1245/1252 [============================>.] - ETA: 2s - loss: 2.8472
1246/1252 [============================>.] - ETA: 1s - loss: 2.8471
1247/1252 [============================>.] - ETA: 1s - loss: 2.8461
1248/1252 [============================>.] - ETA: 1s - loss: 2.8457
1249/1252 [============================>.] - ETA: 0s - loss: 2.8450
1250/1252 [============================>.] - ETA: 0s - loss: 2.8450
1251/1252 [============================>.] - ETA: 0s - loss: 2.8462
1252/1252 [==============================] - 383s 306ms/step - loss: 2.8452 - val_loss: 18.0393
.......
```

