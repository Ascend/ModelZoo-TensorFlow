# HRNet_for_TensorFlow

## 目录
* [基本信息](#基本信息)
* [概述](#概述)
* [训练环境准备](#训练环境准备)
* [快速上手](#快速上手)
* [高级参考](#高级参考)


## 基本信息

**发布者（Publisher）：Huawei**
**应用领域（Application Domain）：Image Classification
**版本（Version）：1.1
**修改时间（Modified） ：2021.07.19
**大小（Size）：5M
**框架（Framework）：TensorFlow 1.15.0
**模型格式（Model Format）：ckpt
**精度（Precision）：Mixed
**处理器（Processor）：昇腾910
**应用级别（Categories）：Official
**描述（Description）：基于tensorflow实现ImageNet分类的高分辨率表示。网络结构和训练超参数与官方 pytorch 实现保持相同

## 概述

基于tensorflow实现ImageNet分类的高分辨率表示。网络结构和训练超参数与官方 pytorch 实现保持相同

- 参考论文：

    https://arxiv.org/pdf/1908.07919.pdf

- 参考实现：

    https://github.com/yuanyuanli85/tf-hrnet

- 适配昇腾 AI 处理器的实现：

    https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/built-in/cv/image_classification/HRNet_ID0139_for_TensorFlow


- 通过Git获取对应commit\_id的代码方法如下：
  
    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    ```

#### 默认配置<a name="section91661242121611"></a>

- 训练超参（单卡）

  - Batch size: 32
  - Momentum: 0.9
  - LR scheduler: piecewise constant strategy
  - Learning rate(LR): [0.0125]
  - Optimizer: Momentum
  - Weight l2 scale: 0.0001
  - Train_steps:40036 (nb_smpls_train* nb_epoches * nb_epochs_rat / batch_size=1281167 * 1 * 1 / 32)

#### 支持特性<a name="section1899153513554"></a>

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 是    |
| 混合精度  | 是    |
| 并行数据  | 是    |

#### 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

#### 开启混合精度<a name="section20779114113713"></a>

脚本已默认开启混合精度，设置precision_mode参数的脚本参考如下。

  ```
    config = tf.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    if FLAGS.over_dump:
        custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes(FLAGS.over_dump_path)
        custom_op.parameter_map["enable_dump_debug"].b = True
        custom_op.parameter_map["dump_debug_mode"].s = tf.compat.as_bytes("all")
    if FLAGS.data_dump:
        custom_op.parameter_map["enable_dump"].b = True
        custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes(FLAGS.data_dump_path)
        custom_op.parameter_map["dump_step"].s = tf.compat.as_bytes(str(FLAGS.data_dump_step))
        custom_op.parameter_map["dump_mode"].s = tf.compat.as_bytes("all")
        custom_op.parameter_map["enable_exception_dump"].i = 1
        custom_op.parameter_map["op_debug_level"].i = 2
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes(FLAGS.precision_mode)
  ```


## 训练环境准备

1.  硬件环境准备请参见各硬件产品文档"[驱动和固件安装升级指南]( https://support.huawei.com/enterprise/zh/category/ai-computing-platform-pid-1557196528909)"。需要在硬件设备上安装与CANN版本配套的固件与驱动。
2.  宿主机上需要安装Docker并登录[Ascend Hub中心](https://ascendhub.huawei.com/#/detail?name=ascend-tensorflow-arm)获取镜像。

    当前模型支持的镜像列表如[表1](#zh-cn_topic_0000001074498056_table1519011227314)所示。

    **表 1** 镜像列表

    <a name="zh-cn_topic_0000001074498056_table1519011227314"></a>
    <table><thead align="left"><tr id="zh-cn_topic_0000001074498056_row0190152218319"><th class="cellrowborder" valign="top" width="47.32%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001074498056_p1419132211315"><a name="zh-cn_topic_0000001074498056_p1419132211315"></a><a name="zh-cn_topic_0000001074498056_p1419132211315"></a><em id="i1522884921219"><a name="i1522884921219"></a><a name="i1522884921219"></a>镜像名称</em></p>
    </th>
    <th class="cellrowborder" valign="top" width="25.52%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001074498056_p75071327115313"><a name="zh-cn_topic_0000001074498056_p75071327115313"></a><a name="zh-cn_topic_0000001074498056_p75071327115313"></a><em id="i1522994919122"><a name="i1522994919122"></a><a name="i1522994919122"></a>镜像版本</em></p>
    </th>
    <th class="cellrowborder" valign="top" width="27.16%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001074498056_p1024411406234"><a name="zh-cn_topic_0000001074498056_p1024411406234"></a><a name="zh-cn_topic_0000001074498056_p1024411406234"></a><em id="i723012493123"><a name="i723012493123"></a><a name="i723012493123"></a>配套CANN版本</em></p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="zh-cn_topic_0000001074498056_row71915221134"><td class="cellrowborder" valign="top" width="47.32%" headers="mcps1.2.4.1.1 "><a name="zh-cn_topic_0000001074498056_ul81691515131910"></a><a name="zh-cn_topic_0000001074498056_ul81691515131910"></a><ul id="zh-cn_topic_0000001074498056_ul81691515131910"><li><em id="i82326495129"><a name="i82326495129"></a><a name="i82326495129"></a>ARM架构：<a href="https://ascend.huawei.com/ascendhub/#/detail?name=ascend-tensorflow-arm" target="_blank" rel="noopener noreferrer">ascend-tensorflow-arm</a></em></li><li><em id="i18233184918125"><a name="i18233184918125"></a><a name="i18233184918125"></a>x86架构：<a href="https://ascend.huawei.com/ascendhub/#/detail?name=ascend-tensorflow-x86" target="_blank" rel="noopener noreferrer">ascend-tensorflow-x86</a></em></li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="25.52%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001074498056_p1450714271532"><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><em id="i72359495125"><a name="i72359495125"></a><a name="i72359495125"></a>21.0.2</em></p>
    </td>
    <td class="cellrowborder" valign="top" width="27.16%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001074498056_p18244640152312"><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><em id="i162363492129"><a name="i162363492129"></a><a name="i162363492129"></a><a href="https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software" target="_blank" rel="noopener noreferrer">5.0.2</a></em></p>
    </td>
    </tr>
    </tbody>
    </table>


## 快速上手

#### 数据集准备<a name="section361114841316"></a>

- 模型训练使用ImageNet2012 TF record格式训练

  tfrecord文件制作参照以下路径：

  readme：https://github.com/tensorflow/models/blob/master/research/slim/README.md

  脚本：https://github.com/tensorflow/models/tree/master/research/slim/datasets/download_and_convert_imagenet.sh

#### 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。

- 启动训练之前，首先要配置程序运行相关环境变量。

  环境变量配置信息参见：

     [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)

- 单卡训练 

  1. 配置训练参数。

     ```
      脚本位于  HRNet_ID0139_for_TensorFlow/test/train_full_1p.sh  ，可配置的参数如下：
       --batch_size=32                        //训练的batch size
       --nb_smpls_train=1281167               //训练的steps依赖这个变量，实际训练steps=nb_smpls_train/batch_size
       --data_path="./data/imagenet_TF"       //自行指定数据集路径，必传
     ```
  
  2. 启动训练。
  
     ```
     脚本位于  HRNet_ID0139_for_TensorFlow/test/train_full_1p.sh  ，示例如下：
     bash train_full_1p.sh
     ```


- 8卡训练

  1. 配置训练参数。

     首先检查 HRNet_ID0139_for_TensorFlow/cfgs 目录下是否有存在8卡IP的json配置文件“8p.json”，脚本位于 HRNet_ID0139_for_TensorFlow/test/train_full_8p.sh  ，可配置的参数如下：

     ```
    --batch_size=32                        // 训练的batch size
      --nb_smpls_train=1281167               //训练的steps依赖这个变量，实际训练steps=nb_smpls_train/batch_size
       --data_path="./data/imagenet_TF"       //自行指定数据集路径，必传
     ```
  
  2. 启动训练
  
     脚本位于  HRNet_ID0139_for_TensorFlow/test/train_full_8p.sh  ，示例如下：
  
     ```
     bash train_full_8p.sh
     ```
  


- 模型评估
  train/trainer.py的第202、209行打开，既可以在训练到一定steps时进行evaluate评估
  

## 高级参考

#### 核心脚本和示例代码<a name="section08421615141513"></a>

```
├── README.md                            //代码说明文档
├── scripts                              //GPU训练脚本
│    ├──run_horovod.sh
├── datasets                             //数据预处理
│    ├──abstract_dataset.py
│    ├──ilsvrc12_dataset.py
├── net                                  //模型结构
│    ├──front.py
│    ├──head.py
│    ├──hr_module.py
│    ├──layers.py
│    ├──model.py
│    ├──stage.py
│    ├──utils.py
├── trainer                              //模型训练主接口
│    ├──trainer.py
│    ├──utils.py
├── utils_new                            //共通接口定义
│    ├──config.py
│    ├──imagenet_preprocessing.py
│    ├──misc_utils.py
│    ├──multi_gpu_wrapper.py
├── top                                  //训练入口文件夹
│    ├──train.py                         //训练脚本入口
├── cfgs
│    ├──8p.json                          //8p配置文件
│    ├──w18_s4.cfg                       //w18模型配置文件
│    ├──w30_s4.cfg                       //w30模型配置文件
├── test
│    ├──train_full_1p.sh                 //单卡运行启动脚本(train_steps=100)
│    ├──train_full_8p.sh                 //8卡执行脚本(train_steps=40036)
│    ├──train_performance_1p.sh          //单卡运行启动脚本(train_steps=100)
│    ├──train_performance_8p.sh          //8卡执行脚本(train_steps=40036)
│    ├──env.sh                           //环境变量配置文件

```

#### 脚本参数<a name="section6669162441511"></a>

```
--data_path                   数据集路径
--nb_smpls_train              训练step数
--model_path                  模型ckpt保路径
--net_cfg                     网络配置文件路径net_cfg
--enbl_multi_gpu              是否支持多路
--precision_mode              精度模式，默认allow_fp32_to_fp16
--over_dump                   溢出检测flag，默认false
--data_dump_flag              数据dump flg，默认false
--data_dump_step              第n步进行数据dump，默认为0
--batch_size                  训练的batch size
--train_steps                 训练的steps
```

#### 训练过程<a name="section1589455252218"></a>

1.  通过 “模型训练” 中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡、8卡网络训练。

2.  训练脚本log中包括如下信息（仅作参考）：

```
精度：0.28 （top1）
性能：324.63fps（单p）
```

#### 推理/验证过程<a name="section1465595372416"></a>

1.  通过 “模型训练” 中的指令启动测试。

2.  当前只能针对该工程训练出的checkpoint进行推理测试。
