-   [基本信息]
-   [概述]
-   [训练环境准备]
-   [训练过程]
-   [精度指标]

<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：?Missing pattern**

**版本（Version）：1.1**

**修改时间（Modified） ：2021.12.23**

**大小（Size）：389K**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的车道线检测训练代码** 

<h2 id="概述.md">概述</h2>

本算法是一种快速车道检测的算法，能够处理多数车道和车道变换。本算法将车道检测问题转为实例分割问题，从而每个车道线各自形成一个实例以实现实现端到端训练。在分割车道线用于拟合车道之前，采用一个已学习好的透视变换，在图像上做这种调整，与固定的鸟瞰图做对比以确保在道路平面变化下的车道线拟合的鲁棒性。本算法在tuSimple数据集中验证过且取得较有优势的结果。

- 参考论文：

[D. Neven, B. D. Brabandere, S. Georgoulis, M. Proesmans and L. V. Gool, "Towards End-to-End Lane Detection: an Instance Segmentation Approach," 2018 IEEE Intelligent Vehicles Symposium (IV), 2018, pp. 286-291, doi: 10.1109/IVS.2018.8500547.] 

- 参考实现：

    

- 适配昇腾 AI 处理器的实现：
  
  [https://gitee.com/jingzhongrenxc/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/MNN-LANENET_ID1251_for_TensorFlow](https://gitee.com/jingzhongrenxc/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/MNN-LANENET_ID1251_for_TensorFlow)      


- 通过Git获取对应commit\_id的代码方法如下：
  
    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    ```

## 默认配置<a name="section91661242121611"></a>

- 训练数据集（tuSimple数据集，仅作为用户参考示例）：
  -数据集获取地址：https://link.zhihu.com/?target=https%3A//github.com/TuSimple/tusimple-benchmark/issues/3

- 训练超参

    - SNAPSHOT_EPOCH: 8
    - BATCH_SIZE: 16
    - VAL_BATCH_SIZE: 16
    - EPOCH_NUMS: 905
    - WARM_UP:
        - ENABLE: True
        - EPOCH_NUMS: 8
    - LR: 0.001
    - LR_POLYNOMIAL_POWER: 0.9
    - MOMENTUM: 0.9
    - WEIGHT_DECAY: 0.0005
    - MOVING_AVE_DECAY: 0.9995

## 支持特性<a name="section1899153513554"></a>

| 特性列表  | 是否支持 |
|-------    |------    |
| 混合精度  |  是      |


## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

脚本已默认关闭混合精度，因为开启之后性能无提升。

  ```custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")



     100%|██████████| 192/192 [02:00<00:00,  1.59it/s]
  
  ```


<h2 id="训练环境准备.md">训练环境准备</h2>

硬件环境：Ascend: 1*Ascend 910(32GB) | ARM: 24 核 96GB
运行环境：ascend-share/5.1.rc2.alpha005_tensorflow-ascend910-cp37-euleros2.8-aarch64-training:1.15.0-21.0.2_0602

    
  
<h2 id="训练过程.md">训练过程</h2>

- 数据集准备。

  数据集要求如下：

 请参阅 data/training_data_example 文件夹结构。 并且需要生成一个train.txt和一个val.txt来记录用于训练模型的数据。
训练样本由三部分组成，二进制分割标签文件、实例分割标签文件和原始图像。 二进制分割使用 255 表示车道字段，其余部分使用 0。 该实例使用不同的像素值来表示不同的车道场，其余的为 0。所有的训练图像将根据配置文件缩放到相同的比例。
使用下脚本生成 tensorflow 记录文件：
python tools/make_tusimple_tfrecords.py 

-脚本和示例代码
├── config                              
    ├── tusimple_lanenet.yaml          //配置文件

├── data_provider          //准备数据
    ├── lanenet_data_feed_pipline.py 
    ├── tf_io_pipline_tools.py

├── lanenet_model          //构建模型
    ├── lanenet.py
    ├── lanenet_back_end.py
    ├── lanenet_discriminative_loss.py
    ├── lanenet_front_end.py
    ├── lanenet_postprocess.py

├── local_utils                                   
    ├── config_utils

        ├── parse_config_utils.py
    ├── log_util

        ├── init_logger.py

├── mnn_project 
    ├── freeze_lanenet_model.py

├── semantic_segmentation_zoo
    ├── bisenet_v2.py
    ├── cnn_basenet.py
    ├── vgg16_based_fcn.py

├── tools
    ├── evaluate_lanenet_on_tusimple.py
    ├── evaluate_model_utils.py
    ├── generate_tusimple_dataset.py
    ├── make_tusimple_tfrecords.py
    ├── test_lanenet.py
    ├── train_lanenet_tusimple.py          //入口

├── trainner 
    ├── tusimple_lanenet_multi_gpu_trainner.py
    ├── tusimple_lanenet_single_gpu_trainner.py

 
- 模型训练。

   使用pycharm在ModelArts训练启动文件为：/tools/train_lanenet_tusimple.py




## 训练过程<a name="section1589455252218"></a>

启动单卡训练

1.  训练脚本log中包括如下信息。

```
  0%|          | 0/192 [00:00<?, ?it/s]2022-06-16 22:11:40.757045: I tf_adapter/kernels/geop_npu.cc:817] The model has been compiled on the Ascend AI processor, current graph id is: 11

train loss: 34.08809, b_loss: 1.56111, i_loss: 28.89311:   0%|          | 0/192 [03:17<?, ?it/s]
train loss: 34.08809, b_loss: 1.56111, i_loss: 28.89311:   1%|          | 1/192 [03:17<10:28:28, 197.42s/it]
train loss: 33.28113, b_loss: 1.53935, i_loss: 28.10791:   1%|          | 1/192 [03:18<10:28:28, 197.42s/it]
train loss: 33.28113, b_loss: 1.53935, i_loss: 28.10791:   1%|          | 2/192 [03:18<7:18:14, 138.39s/it] 
train loss: 32.75388, b_loss: 1.61211, i_loss: 27.50790:   1%|          | 2/192 [03:18<7:18:14, 138.39s/it]
train loss: 32.75388, b_loss: 1.61211, i_loss: 27.50790:   2%|▏         | 3/192 [03:18<5:05:44, 97.06s/it] 
train loss: 33.34206, b_loss: 1.55278, i_loss: 28.15541:   2%|▏         | 3/192 [03:19<5:05:44, 97.06s/it]
train loss: 33.34206, b_loss: 1.55278, i_loss: 28.15541:   2%|▏         | 4/192 [03:19<3:33:27, 68.13s/it]
train loss: 34.02236, b_loss: 1.59200, i_loss: 28.79649:   2%|▏         | 4/192 [03:19<3:33:27, 68.13s/it]
train loss: 34.02236, b_loss: 1.59200, i_loss: 28.79649:   3%|▎         | 5/192 [03:19<2:29:12, 47.88s/it]
train loss: 33.55459, b_loss: 1.55814, i_loss: 28.36258:   3%|▎         | 5/192 [03:20<2:29:12, 47.88s/it]
train loss: 33.55459, b_loss: 1.55814, i_loss: 28.36258:   3%|▎         | 6/192 [03:20<1:44:28, 33.70s/it]
train loss: 34.67654, b_loss: 1.56015, i_loss: 29.48252:   3%|▎         | 6/192 [03:21<1:44:28, 33.70s/it]
train loss: 34.67654, b_loss: 1.56015, i_loss: 29.48252:   4%|▎         | 7/192 [03:21<1:13:19, 23.78s/it]
train loss: 33.96736, b_loss: 1.56802, i_loss: 28.76547:   4%|▎         | 7/192 [03:21<1:13:19, 23.78s/it]
train loss: 33.96736, b_loss: 1.56802, i_loss: 28.76547:   4%|▍         | 8/192 [03:21<51:36, 16.83s/it]  
train loss: 33.22740, b_loss: 1.53942, i_loss: 28.05412:   4%|▍         | 8/192 [03:22<51:36, 16.83s/it]
train loss: 33.22740, b_loss: 1.53942, i_loss: 28.05412:   5%|▍         | 9/192 [03:22<36:30, 11.97s/it]
train loss: 35.09825, b_loss: 1.53474, i_loss: 29.92964:   5%|▍         | 9/192 [03:23<36:30, 11.97s/it]
train loss: 35.09825, b_loss: 1.53474, i_loss: 29.92964:   5%|▌         | 10/192 [03:23<25:58,  8.56s/it]
train loss: 34.73063, b_loss: 1.53588, i_loss: 29.56088:   5%|▌         | 10/192 [03:23<25:58,  8.56s/it]
train loss: 34.73063, b_loss: 1.53588, i_loss: 29.56088:   6%|▌         | 11/192 [03:23<18:38,  6.18s/it]
train loss: 33.60005, b_loss: 1.60795, i_loss: 28.35823:   6%|▌         | 11/192 [03:24<18:38,  6.18s/it]
train loss: 33.60005, b_loss: 1.60795, i_loss: 28.35823:   6%|▋         | 12/192 [03:24<13:31,  4.51s/it]
train loss: 32.77514, b_loss: 1.59790, i_loss: 27.54338:   6%|▋         | 12/192 [03:24<13:31,  4.51s/it]
train loss: 32.77514, b_loss: 1.59790, i_loss: 27.54338:   7%|▋         | 13/192 [03:24<09:58,  3.35s/it]
train loss: 33.52126, b_loss: 1.57790, i_loss: 28.30949:   7%|▋         | 13/192 [03:25<09:58,  3.35s/it]
train loss: 33.52126, b_loss: 1.57790, i_loss: 28.30949:   7%|▋         | 14/192 [03:25<07:30,  2.53s/it]
train loss: 32.66967, b_loss: 1.52915, i_loss: 27.50666:   7%|▋         | 14/192 [03:26<07:30,  2.53s/it]
train loss: 32.66967, b_loss: 1.52915, i_loss: 27.50666:   8%|▊         | 15/192 [03:26<05:45,  1.95s/it]
train loss: 34.08065, b_loss: 1.56651, i_loss: 28.88028:   8%|▊         | 15/192 [03:26<05:45,  1.95s/it]
train loss: 34.08065, b_loss: 1.56651, i_loss: 28.88028:   8%|▊         | 16/192 [03:26<04:34,  1.56s/it]
...
train loss: 27.36981, b_loss: 1.53718, i_loss: 22.19878: 100%|██████████| 192/192 [05:20<00:00,  1.67s/it]
2022-06-16 22:16:42.990 | INFO     | trainner.tusimple_lanenet_single_gpu_trainner:train:366 - => Epoch: 1 Time: 2022-06-16 22:16:42 Train loss: 31.00657 ...

<h2 id="精度指标.md">精度指标</h2>
训练总Loss

| gpu   | npu  |原论文 |
|-------|------|------|
|1.9305 |0.991 |      | 
原文训练轮数过多(40k轮)，只迭代905轮  
