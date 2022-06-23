-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [快速上手](#快速上手.md)
-   [高级参考](#高级参考.md)
  <h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）： Optical Character Recognition**

**修改时间（Modified） ：2022.6.14**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的Attention-OCR自然场景文本检测识别网络训练代码** 

<h2 id="概述.md">概述</h2>

Attention-OCR是一个基于卷积神经网络CNN、循环神经网络RNN以及一种新颖的注意机制的自然场景文本检测识别网络。

- 参考论文：

    ["Attention-based Extraction of Structured Information from Street View
    Imagery"](https://arxiv.org/abs/1704.03549)


- 参考实现：[models/research/attention_ocr at master · tensorflow/models · GitHub](https://github.com/tensorflow/models/tree/master/research/attention_ocr)


- 适配昇腾 AI 处理器的实现：

  [TensorFlow/contrib/cv/Attention-OCR_ID2013_for_TensorFlow · Ypo6opoc/ModelZoo-TensorFlow - 码云 - 开源中国 (gitee.com)](https://gitee.com/ypo6opoc/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/Attention-OCR_ID2013_for_TensorFlow)

  ​



## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

脚本已默认开启混合精度，设置precision_mode参数的脚本参考如下。

  ```
    config = tf.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭
    config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF  # 必须显式关闭
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
  ```

## Loss Scale<a name="section168064817164"></a>

在混合精度计算中使用float16数据格式数据动态范围降低，造成梯度计算出现浮点溢出，会导致部分参数更新失败。为了保证部分模型训练在混合精度训练过程中收敛，需要配置Loss Scale的方法。

Loss Scale方法通过在前向计算所得的loss乘以loss scale系数S，起到在反向梯度计算过程中达到放大梯度的作用，从而最大程度规避浮点计算中较小梯度值无法用FP16表达而出现的溢出问题。在参数梯度聚合之后以及优化器更新参数之前，将聚合后的参数梯度值除以loss scale系数S还原。

动态Loss Scale通过在训练过程中检查梯度中浮点计算异常状态，自动动态选取loss scale系数S以适应训练过程中梯度变化，从而解决人工选取loss scale系数S和训练过程中自适应调整的问题。

在具体实现中，昇腾910 AI处理器由于浮点计算特性不同，在计算过程中的浮点异常检查等部分与GPU存在差异。

## 开启动态Loss Scale<a name="section20779114113713"></a>

设置动态Loss Scale的脚本参考如下。

```
    loss_scale_manager = tf.contrib.mixed_precision.ExponentialUpdateLossScaleManager(init_loss_scale=2**32, incr_every_n_steps=1000, decr_every_n_nan_or_inf=2, decr_ratio=0.5)
    optimizer = NPULossScaleOptimizer(optimizer, loss_scale_manager)
```

<h2 id="训练环境准备.md">训练环境准备</h2>

1.  硬件环境准备请参见各硬件产品文档"[驱动和固件安装升级指南]( https://support.huawei.com/enterprise/zh/category/ai-computing-platform-pid-1557196528909)"。需要在硬件设备上安装与CANN版本配套的固件与驱动。
2.  宿主机上需要安装Docker并登录[Ascend Hub中心](https://ascendhub.huawei.com/#/detail?name=ascend-tensorflow-arm)获取镜像。



<h2 id="快速上手.md">快速上手</h2>

- 数据集准备
1. 模型训练使用FSNS数据集，数据集请用户自行获取。
2. 数据集处理后，放入模型目录下，在训练脚本中指定数据集路径，可正常使用。

* 使用Inception_v3对网络进行初始化

  ```
  wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
  tar xf inception_v3_2016_08_28.tar.gz
  python train.py --checkpoint_inception=./inception_v3.ckpt
  ```



## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。

- 启动训练之前，首先要配置程序运行相关环境变量。

  环境变量配置信息参见：

     [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

- 单卡训练 

  1. 配置训练参数。

     首先在modelarts_entry_acc.py中，配置训练数据集路径、训练日志以及checkpoints存放位置和Inception保存路径，请用户根据实际路径配置，示例如下所示：

     ```
     # 路径参数初始化
     --data_url="/home/ma-user/modelarts/inputs/data_url_0"
     --train_url="/home/ma-user/modelarts/outputs/train_url_0/"
     --ckpt_path="/home/ma-user/modelarts/inputs/inception_v3.ckpt"
     ```

  2. 启动训练。（该脚本在训练结束后会自动进行模型精度的验证）

     ```
     python3 modelarts_entry_acc.py
     ```


- 验证。

  1. 配置验证参数。

     首先在eval.py中，配置训练数据集路径和checkpoint保存路径，请用户根据实际路径配置，示例如下所示：

     ```
     # 路径参数初始化
     --data_url="/home/ma-user/modelarts/inputs/data_url_0"
     --train_url="/home/ma-user/modelarts/outputs/train_url_0/"
     ```

  2. 启动验证。

     ```
     python3 eval.py
     ```




## 性能和精度<a name="section715881518135"></a>

- NPU环境下的性能：0.13 sec/step
- NPU环境下的精度：经过400K个step的训练后精度达到 82.66% 经过1000K个step训练（在NPU上训练了34个小时）后的精度为83.50%
- GPU环境（一张RTX 2080Ti）下的性能：0.32 sec/step 
- GPU环境（一张RTX 2080Ti）下的精度：经过400K个step的训练后精度达到 82.9375%







<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码<a name="section08421615141513"></a>

```
├── common_flags.py                          //为训练和测试定义配置
├── data_provider.py						//数据集处理
├── eval.py									//模型评估
├── inception_preprocessing.py				//为Inception网络预处理图像
├──	metrics.py								//模型的质量度量
├──	model.py								//模型构建函数
├── modelzoo_level.txt 						//网络状态描述文件
├── NPU_train.py							//模型NPU训练
├──	sequence_layers.py						//用于字符预测的序列层的各种实现
├── train.py								//模型GPU训练
├── train_testcase.sh						//训练测试用例
├── utils.py								//支持构建Attention-OCR的函数
├── modelarts_entry_acc.py                  //用于在Modelarts上拉起精度测试
├── modelarts_entry_perf.py                 //用于在Modelarts上拉起性能测试
├── test     
│    ├──train_performance_1p.sh             //训练性能入口
│    ├──train_full_1p.sh                    //训练精度入口，包含准确率评估
├── datasets
│    ├──fsns.py                             //读取FSNS数据集的配置
│    ├──fsns_test.py                        //FSNS数据集模块的测试

```


##脚本参数<a name="section6669162441511"></a>

```
--dataset_dir          数据集目录
--train_log_dir        训练日志以及checkpoints存放位置
--checkpoint_inception 用于初始权重的inception位置
--max_number_of_steps  训练轮数
--log_interval_steps   保存checkpoints的频率

```

## 训练过程<a name="section1589455252218"></a>

1.  通过“模型训练”中的训练指令启动单卡训练。

2.  训练脚本log中包括如下信息。

```
global step 999970: loss = 30.2755 (0.114 sec/step)
global step 999971: loss = 30.2009 (0.123 sec/step)
global step 999972: loss = 30.1321 (0.125 sec/step)
global step 999973: loss = 30.1833 (0.125 sec/step)
global step 999974: loss = 30.6518 (0.125 sec/step)
global step 999975: loss = 30.7026 (0.127 sec/step)
global step 999976: loss = 30.2010 (0.126 sec/step)
global step 999977: loss = 30.2577 (0.128 sec/step)
global step 999978: loss = 30.4012 (0.118 sec/step)
global step 999979: loss = 30.1842 (0.122 sec/step)
global step 999980: loss = 30.2944 (0.124 sec/step)
global step 999981: loss = 30.3081 (0.117 sec/step)
global step 999982: loss = 30.2698 (0.112 sec/step)
global step 999983: loss = 30.0736 (0.113 sec/step)
global step 999984: loss = 30.0913 (0.112 sec/step)
global step 999985: loss = 30.1208 (0.112 sec/step)
global step 999986: loss = 30.0885 (0.122 sec/step)
global step 999987: loss = 30.1067 (0.129 sec/step)
global step 999988: loss = 30.3190 (0.128 sec/step)
global step 999989: loss = 30.2904 (0.129 sec/step)
global step 999990: loss = 30.2282 (0.128 sec/step)
global step 999991: loss = 30.2426 (0.122 sec/step)
global step 999992: loss = 30.1282 (0.111 sec/step)
global step 999993: loss = 30.3605 (0.112 sec/step)
global step 999994: loss = 30.3907 (0.121 sec/step)
global step 999995: loss = 30.4900 (0.121 sec/step)
global step 999996: loss = 30.2729 (0.114 sec/step)
global step 999997: loss = 30.0970 (0.116 sec/step)
global step 999998: loss = 30.2434 (0.115 sec/step)
global step 999999: loss = 30.3903 (0.116 sec/step)
global step 1000000: loss = 30.1427 (0.113 sec/step)
```

## 推理/验证过程<a name="section1465595372416"></a>

1.  通过“模型训练”中的验证指令启动验证。

2.  当前只能针对该工程训练出的checkpoint进行推理测试。

3.  测试结束后会打印测试集的SequenceAccuracy和CharacterAccuracy，如下所示。

```
INFO:tensorflow:Starting evaluation at 2022-06-15-12:12:03
I0615 20:12:03.015775 140615084922688 evaluation.py:450] Starting evaluation at 2022-06-15-12:12:03
INFO:tensorflow:Evaluation [133/1339]
I0615 20:14:19.040591 140615084922688 evaluation.py:167] Evaluation [133/1339]
INFO:tensorflow:Evaluation [266/1339]
I0615 20:16:32.740562 140615084922688 evaluation.py:167] Evaluation [266/1339]
INFO:tensorflow:Evaluation [399/1339]
I0615 20:18:46.108737 140615084922688 evaluation.py:167] Evaluation [399/1339]
INFO:tensorflow:Evaluation [532/1339]
I0615 20:20:57.364342 140615084922688 evaluation.py:167] Evaluation [532/1339]
INFO:tensorflow:Evaluation [665/1339]
I0615 20:23:08.015209 140615084922688 evaluation.py:167] Evaluation [665/1339]
INFO:tensorflow:Evaluation [798/1339]
I0615 20:25:18.427117 140615084922688 evaluation.py:167] Evaluation [798/1339]
INFO:tensorflow:Evaluation [931/1339]
I0615 20:27:30.358431 140615084922688 evaluation.py:167] Evaluation [931/1339]
INFO:tensorflow:Evaluation [1064/1339]
I0615 20:29:42.058201 140615084922688 evaluation.py:167] Evaluation [1064/1339]
INFO:tensorflow:Evaluation [1197/1339]
I0615 20:31:52.447575 140615084922688 evaluation.py:167] Evaluation [1197/1339]
INFO:tensorflow:Evaluation [1330/1339]
I0615 20:34:02.774163 140615084922688 evaluation.py:167] Evaluation [1330/1339]
INFO:tensorflow:Evaluation [1339/1339]
I0615 20:34:11.470245 140615084922688 evaluation.py:167] Evaluation [1339/1339]
INFO:tensorflow:Finished evaluation at 2022-06-15-12:34:11
I0615 20:34:11.471059 140615084922688 evaluation.py:456] Finished evaluation at 2022-06-15-12:34:11
eval/CharacterAccuracy[0.966915]
eval/SequenceAccuracy[0.826643]
```
