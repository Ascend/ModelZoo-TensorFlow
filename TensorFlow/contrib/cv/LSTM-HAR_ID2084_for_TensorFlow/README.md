## 基本信息

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：CV** 

**版本（Version）：1.0**

**修改时间（Modified） ：2022.1.13**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Research**

**描述（Description）：基于TensorFlow框架的LSTM人类活动识别网络训练代码**

## 概述

LSTM-HAR是使用堆叠残余双向LSTM单元(RNN)来识别六种运动状态（即行走、上楼梯、下楼梯、坐、站、躺）的模型，其使用A Public Domain Dataset for Human Activity Recognition Using Smartphones作为数据集。

* 参考论文：

   [Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/pdf/1609.08144.pdf)

* 参考项目：

  https://github.com/guillaume-chevalier/HAR-stacked-residual-bidir-LSTMs

* 获取代码

  ```
  git clone https://gitee.com/ascend/modelzoo.git    # 克隆仓库的代码
  cd modelzoo/contrib/TensorFlow/Research/cv/LSTM-HAR_ID2084_for_TensorFlow    # 切换到模型的代码仓目录
  ```

# 默认配置

* 训练超参
  * Batch size: 100
  * Learning rate(LR): 0.001
  * Optimizer: AdamOptimizer
  * Train epoch: 250

##  支持特性

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 否       |
| 混合精度   | 是       |
| 并行数据   | 否       |

##  混合精度训练

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度

脚本已默认开启混合精度，设置precision_mode参数的脚本参考如下。

  ```
  custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
  custom_op.name = 'NpuOptimizer'
  custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes(str(args.precision_mode))
  ```

<h2 id="训练环境准备.md">训练环境准备</h2>

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
    <td class="cellrowborder" valign="top" width="25.52%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001074498056_p1450714271532"><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><em id="i72359495125"><a name="i72359495125"></a><a name="i72359495125"></a>20.2.0</em></p>
    </td>
    <td class="cellrowborder" valign="top" width="27.16%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001074498056_p18244640152312"><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><em id="i162363492129"><a name="i162363492129"></a><a name="i162363492129"></a><a href="https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software" target="_blank" rel="noopener noreferrer">20.2</a></em></p>
    </td>
    </tr>
    </tbody>
    </table>


<h2 id="快速上手.md">快速上手</h2>

- 数据集准备
1. 模型训练使用原作者提供的A Public Domain Dataset for Human Activity Recognition Using Smartphones数据集，数据集请用户自行获取。OBS中数据集的地址为obs://cann-id2084/dataset/

## 模型训练

- 单击“立即下载”，并选择合适的下载方式下载源码包。

- 启动训练之前，首先要配置程序运行相关环境变量。

  环境变量配置信息参见：

     [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

- 单卡训练 

  1. 配置训练参数。

     首先在脚本test/train_full_1p.sh中，配置batch_size、steps、epochs、data_path等参数，请用户根据实际路径配置data_path，或者在启动训练的命令行中以参数形式下发。

     ```
      batch_size=100
      steps=100
      epochs=5
      data_path="../UCI HAR Dataset"
     ```

  2. 启动训练。

     启动单卡训练 （脚本为LSTM-HAR_ID2084_for_TensorFlow/test/train_full_1p.sh） 

     ```
     bash train_full_1p.sh --data_path=../UCI HAR Dataset
     ```
# 训练结果

- 论文精度：94%  GPU复现代码测试集精度：90.19%  NPU迁移代码测试集精度：90.23%

- GPU训练使用华为Tesla V100环境，训练结果：obs://cann-id2084/gpu

- NPU迁移后运行的作业日志：obs://cann-id2084/npu/MA-new-LSTM-HAR_ID2084_for_TensorFlow-01-24-19-15/

<h2 id="训练结果.md">训练结果</h2>

- 精度结果比对

|精度指标项|论文发布|GPU实测|NPU实测|
|---|---|---|---|
|ACC|0.9400|0.9019|0.9023|

- 性能结果比对  

|性能指标项|论文发布|GPU实测|NPU实测|
|---|---|---|---|
|FPS|-----|898.13|767.22|

<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码

```
├── README.md                         //代码说明文档
├── config_dataset_HAR_6_classes.py   //网络训练与测试代码
├── lstm_architecture.py              //模型构建相关函数文件
├── requirements.txt                  //训练python依赖列表
├── modelzoo_level.txt
├── test
│    ├──train_performance_1p.sh       //单卡训练验证性能启动脚本
│    ├──train_full_1p.sh              //单卡全量训练启动脚本                
```

## 训练过程

1.  通过“模型训练”中的训练指令启动单卡卡训练。

2.  参考脚本的模型存储路径为./output/mode.ckpt。
