-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [快速上手](#快速上手.md)
-   [迁移学习指导](#迁移学习指导.md)
-   [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Computer Version**

**版本（Version）：1.1**

**修改时间（Modified） ：2022.1.21**

**大小（Size）：928kb**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Research**

**描述（Description）：基于TensorFlow框架的GeoNet网络训练代码** 

<h2 id="概述.md">概述</h2>

GeoNet是一种可以联合学习单目深度、光流和相机姿态的无监督学习框架，这三者通过三维场景几何特性耦合在一起，以端到端的方式进行联合学习。从每个单独模块的预测中提取几何关系，然后将其合并为图像重构损失。

- 参考论文：

    [Yin Z , Shi J . GeoNet: Unsupervised Learning of Dense Depth, Optical Flow and Camera Pose[C]// 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2018.](https://arxiv.org/abs/1803.02276v1) 

- 参考实现：https://github.com/yzcjtr/GeoNet

    

- 适配昇腾 AI 处理器的实现：
  
  [https://gitee.com/huimin-yu/modelzoo/tree/master/contrib/TensorFlow/Research/cv/GeoNet_ID2357_for_TensorFlow](https://gitee.com/huimin-yu/modelzoo/tree/master/contrib/TensorFlow/Research/cv/GeoNet_ID2357_for_TensorFlow)      

## 默认配置<a name="section91661242121611"></a>

- 训练超参

  --mode=train_rigid --learning_rate=0.00001 --seq_length=3 --batch_size=4 --max_steps=210100


## 支持特性<a name="section1899153513554"></a>

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 是    |
| 混合精度  | 是    |
| 并行数据  | 是    |

## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

脚本已默认开启混合精度，设置precision_mode参数的脚本参考如下。

  ```
 global_config = tf.ConfigProto(log_device_placement=False)  
custom_op = global_config.graph_options.rewrite_options.custom_optimizers.add()  
custom_op.name = "NpuOptimizer"  
custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")  
  ```


<h2 id="训练环境准备.md">训练环境准备</h2>

1.  NPU环境
 ```
 NPU: 1*Ascend 910
 CPU: 24*vCPUs 550GB
  ```
2.  运行环境
 ```
swr.cn-north-4.myhuaweicloud.com/ascend-share/5.0.4.alpha005_tensorflow-ascend910-cp37-euleros2.8-aarch64-training:1.15.0-21.0.2_0107
  ```
3.  第三方依赖

 ```
 imageio
 scipy
 opencv-python
  ```

- 数据集准备
1. 模型训练使用kitti_odometry数据集，已上传至obs中，obs路径如下：obs://cann-id2357/dataset/stageone/

## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。

- 启动训练之前，首先要配置程序运行相关环境变量。

  环境变量配置信息参见：

     [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

- 训练命令

     ```
     python geonet_main.py --mode=train_rigid --dataset_dir=/home/yhm/stageone/train/ --checkpoint_dir=/home/yhm/output/ckpt/ --learning_rate=0.00001 --seq_length=3 --batch_size=4 --max_steps=210100 
     ```
其中，/home/yhm/output/ckpt/是事先创建的存放ckpt文件夹。

- 测试命令

     ```
     python geonet_main.py --mode=test_pose --dataset_dir=/home/yhm/stageone/test/ --init_ckpt_file=/home/yhm/output/ckpt/model-185000 --batch_size=1 --seq_length=3 --pose_test_seq=9 --output_dir=/home/yhm/output/pose/
    ```
/home/yhm/output/ckpt/model-185000为用于测试的模型文件，若restore其它iter的ckpt文件，修改后缀为model-xxxxxx即可。测试结果输出存到/home/yhm/output/pose/下。
 - 验证指令

      ```
      python kitti_eval/eval_pose.py --gtruth_dir=/home/yhm/stageone/eval/09_snippets/ --pred_dir=/home/yhm/output/pose/ 
      ```

<h2 id="迁移学习指导.md">迁移学习指导</h2>

- 数据集准备。

  获取数据。请参见“快速上手”中的数据集准备

- 模型训练。

   参考“模型训练”中训练步骤。

- 模型测试。

    参考“模型训练”中测试步骤。
     
-  模型评估。
    
    参考“模型训练”中验证步骤。

<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码<a name="section08421615141513"></a>

1.脚本参数
```
 --data_path=/home/yhm --output_path=/home/yhm/output
```
运行性能脚本时，执行下面的命令：
```
 bash ./test/train_performance_1p.sh --data_path=/home/yhm --output_path=/home/yhm/output
```
运行精度脚本时，执行下面的命令：
```
 bash ./test/train_full_1p.sh --data_path=/home/yhm --output_path=/home/yhm/output
```
2.  参考脚本的模型日志路径为./test/output/0/，训练脚本log中包括如下信息。

```
Iteration: [    100] | Time: 2.8347s/iter | Loss: 3.112
Iteration: [    200] | Time: 0.1000s/iter | Loss: 2.696
Iteration: [    300] | Time: 0.0999s/iter | Loss: 2.420
Iteration: [    400] | Time: 0.0999s/iter | Loss: 2.276
Iteration: [    500] | Time: 0.1007s/iter | Loss: 2.782
Iteration: [    600] | Time: 0.1003s/iter | Loss: 2.549
Iteration: [    700] | Time: 0.1006s/iter | Loss: 2.566
Iteration: [    800] | Time: 0.1005s/iter | Loss: 2.481
Iteration: [    900] | Time: 0.1005s/iter | Loss: 2.505
Iteration: [   1000] | Time: 0.1003s/iter | Loss: 2.293
Iteration: [   1100] | Time: 0.1009s/iter | Loss: 2.707
Iteration: [   1200] | Time: 0.1004s/iter | Loss: 2.616
Iteration: [   1300] | Time: 0.1003s/iter | Loss: 2.283
Iteration: [   1400] | Time: 0.1005s/iter | Loss: 2.569
Iteration: [   1500] | Time: 0.1002s/iter | Loss: 2.488
```

## 推理/验证过程<a name="section1465595372416"></a>

1.  脚本中已封装了测试和验证执行命令，执行一次脚本，即可进行训练、测试、及验证，获得最终精度。需要注意的是，这样得到的精度为限定模型的精度，如train_full_1p.sh中得到的是model-185000精度结果。

2.  若需要获得不同iteration的精度结果，可以执行测试命令的py文件。修改--init_ckpt_file=/home/yhm/output/ckpt/model-xxxxxx即可，因为运行脚本后，所有ckpt文件都存在home/yhm/output/ckpt/目录下。同时，需要修改测试结果输出的位置为/home/yhm/output/pose_xxxxxx/。

3.  最后执行验证的py文件可以获得不同iteration的精度结果，测试结果输出的位置为/home/yhm/output/pose_xxxxxx/，将其与真值进行对比，得到ATE mean和std。

4.  训练结果

![输入图片说明](NPU_VS_GPU.png)