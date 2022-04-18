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

**大小（Size）：3.2M**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的ADAGAN网络模式缺失训练代码** 

<h2 id="概述.md">概述</h2>

在gan网络每次迭代中，通过在重加权样本上运行GAN算法，向混合模型中添加一个新组件，许多潜在的较弱的个体预测器被聚集起来，形成一个强大的复合预测器，收敛速度增快。 

- 参考论文：

    [Ilya Tolstikhin,Sylvain Gelly,Olivier Bousquet,Carl-Johann Simon-Gabriel,and Bernhard Sch?lkopf.2017.AdaGAN:boosting generative models.In Proceedings of the 31st International Conference on Neural Information Processing Systems.5430–5439.] 

- 参考实现：

    

- 适配昇腾 AI 处理器的实现：
  
  [https://gitee.com/jingzhongrenxc/modelzoo/blob/master/contrib/TensorFlow/Research/cv/adagan/](https://gitee.com/jingzhongrenxc/modelzoo/blob/master/contrib/TensorFlow/Research/cv/adagan/)      


- 通过Git获取对应commit\_id的代码方法如下：
  
    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    ```

## 默认配置<a name="section91661242121611"></a>

- 训练数据集（GMM数据集，仅作为用户参考示例）：
  -数据集自主生成，无需获取
  -gmm_modes_num：10

- 训练超参

  - Batch size: 64
  - Train epoch: 15
  - Iteration（adagan_steps_total）: 10
  - mode（gmm_modes_num）：10

## 支持特性<a name="section1899153513554"></a>

| 特性列表  | 是否支持 |
|-------    |------    |
| 混合精度  |  是      |


## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

脚本已默认关闭混合精度，因为开启之后性能下降。

  ```custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")



     Epoch 15/15: 100%|██████████████████████████| 1000/1000 [00:34<00:00, 29.30it/s]
  
  ```


<h2 id="训练环境准备.md">训练环境准备</h2>

硬件环境：Ascend: 1*Ascend 910(32GB) | ARM: 24 核 96GB

运行环境：ascend-share/5.0.3.alpha003_tensorflow-ascend910-cp37-euleros2.8-aarch64-training:1.15.0-21.0.2_0923

    
  
<h2 id="训练过程.md">训练过程</h2>

- 数据集准备。

  数据集要求如下：

  无需获取，自动生成GMM数据集

-脚本和示例代码

├── adagan_gmm.py                              //主代码

├── README.md                                  //代码说明文档

├── adagan.py                                  //权重更新及采样

├── gan.py                                     //gan网络模型

├── metrics.md                                 //plot图
 
- 模型训练。

   使用pycharm在ModelArts训练启动文件为：adagan_gmm.py






- 启动单卡训练

 模型存储路径为results，训练脚本log中包括如下信息。

```
Epoch  1/15:   0%|                                     | 0/1000 [00:00<?, ?it/s]2021-12-24 13:51:01.626671: I /home/jenkins/agent/workspace/Compile_GraphEngine_Centos_ARM/tensorflow/tf_adapter/kernels/geop_npu.cc:694] The model has been compiled on the Ascend AI processor, current graph id is:11
2021-12-24 13:51:20.821929: I /home/jenkins/agent/workspace/Compile_GraphEngine_Centos_ARM/tensorflow/tf_adapter/kernels/geop_npu.cc:694] The model has been compiled on the Ascend AI processor, current graph id is:21
2021-12-24 13:51:22.428759: I /home/jenkins/agent/workspace/Compile_GraphEngine_Centos_ARM/tensorflow/tf_adapter/kernels/geop_npu.cc:694] The model has been compiled on the Ascend AI processor, current graph id is:31
2021-12-24 13:51:32.796679: I /home/jenkins/agent/workspace/Compile_GraphEngine_Centos_ARM/tensorflow/tf_adapter/kernels/geop_npu.cc:694] The model has been compiled on the Ascend AI processor, current graph id is:41

```
Epoch 15/15:  92%|████████████████████████▋  | 915/1000 [00:19<00:01, 47.29it/s]

2021-12-13 22:50:05.202624: I /home/jenkins/agent/workspace/Compile_GraphEngine_Centos_ARM/tensorflow/tf_adapter/kernels/geop_npu.cc:694] The model has been compiled on the Ascend AI processor, current graph id is:81
INFO:root:Evaluating: log_p=-4.282, C=1.000

<h2 id="精度指标.md">精度指标</h2>
C覆盖率

| gpu   | npu  |原论文 |
|-------|------|-------|
|   1   |  1   |   1   | 

<h2 id="性能指标.md">性能指标</h2>

|     gpu   |     npu     |
|-----------|-------------|
|92.4 (it/s)|125.51 (it/s)| 
