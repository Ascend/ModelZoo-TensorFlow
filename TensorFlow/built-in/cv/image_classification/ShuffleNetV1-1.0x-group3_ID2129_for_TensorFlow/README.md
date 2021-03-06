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

**修改时间（Modified） ：2021.09.23**

**大小（Size）：1126M**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的高效CNN架构设计网络shufflenetv1**

<h2 id="概述.md">概述</h2>

​         目前，神经网络架构设计主要以计算复杂度的\emph{indirect} 度量，即FLOPs 为指导。然而，\emph{direct} 指标（例如速度）还取决于其他因素，例如内存访问成本和平台特性。因此，这项工作建议评估目标平台上的直接指标，而不仅仅是考虑 FLOP。基于一系列受控实验，这项工作推导出了几个实用的\ emph {指南}，用于有效的网络设计。ShuffleNetV1提出了channel shuffle操作，使得网络可以尽情地使用分组卷积来加速。

- 参考论文：
  https://arxiv.org/pdf/1707.01083.pdf

- 参考实现：

  https://github.com/weiSupreme/shufflenetv2-tensorflow

- 适配昇腾 AI 处理器的实现：
  https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/built-in/cv/image_classification/ShuffleNetV1-1.0x-group3_ID2129_for_TensorFlow

- 通过Git获取对应commit\_id的代码方法如下：

  ```
  git clone {repository_url}    # 克隆仓库的代码
  cd {repository_name}    # 切换到模型的代码仓目录
  git checkout  {branch}    # 切换到对应分支
  git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
  cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```

## 默认配置<a name="section91661242121611"></a>

-   训练数据集预处理（以ImageNet2012的Train数据集为例，仅作为用户参考示例）：
    -   图像的输入尺寸为224\*224
    -   图像输入格式：TFRecord 随机裁剪图像尺寸
    -   随机水平翻转图像
    -   根据ImageNet2012数据集通用的平均值和标准偏差对输入图像进行归一化
-   测试数据集预处理（以ImageNet2012的Validation数据集为例，仅作为用户参考示例）：
    -   图像的输入尺寸为224*224 （将图像缩放到256 * 256，然后在中央区域裁剪图像）
    -   图像输入格式：TFRecord 根据ImageNet2012数据集通用的平均值和标准偏差对输入图像进行归一化

-   训练超参（单卡）：
    -   Batch size: 32    Weight decay: 0.0001 Label smoothing: 0.1 Train epoch: 150
    -   Momentum: 0.9
    -   LR scheduler: cosine
    -   Learning rate\(LR\): 0.01
    -   Optimizer: MomentumOptimizer
    -   Weight decay: 0.0001
    -   Label smoothing: 0.1
    -   Train epoch: 150


## 支持特性<a name="section1899153513554"></a>

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 是    |
| 混合精度  | 是    |
| 数据并行  | 是    |

## 混合精度训练<a name="section168064817164"></a>

混合精度训练昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

设置precision_mode参数的脚本参考如下。 

```
run_config = NPURunConfig( model_dir=flags_obj.model_dir, 
                            session_config=session_config, 
                            keep_checkpoint_max=5, 
                            save_checkpoints_steps=5000, 
                            enable_data_pre_proc=True, 
                            iterations_per_loop=iterations_per_loop, 
                            log_step_count_steps=iterations_per_loop, 
                            precision_mode='allow_mix_precision', 
                            hcom_parallel=True 
                        )
```


<h2 id="训练环境准备.md">训练环境准备</h2>

-  硬件环境和运行环境准备请参见《[CANN软件安装指南](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=installation-update)》
-  运行以下命令安装依赖。
```
pip3 install requirements.txt
```
说明：依赖配置文件requirements.txt文件位于模型的根目录

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
1. 模型训练使用ImageNet2012数据集，数据集请用户自行获取。

2. 数据集训练前需要做预处理操作，请用户参考[Tensorflow-Slim](https://github.com/tensorflow/models/tree/master/research/slim),将数据集封装为tfrecord格式。

3. 数据集处理后，放入模型目录下，在训练脚本中指定数据集路径，可正常使用。

## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练。

   环境变量配置信息参见：

    [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)
-  单卡训练
   
```
 以数据目录为./data为例:
  cd test
  bash train_full_1p.sh  --data_path=../data（全量）
  bash train_performance_1p.sh --data_path=../data（功能、性能测试）

```




<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码<a name="section08421615141513"></a>


```
ShuffleNetV1-1.0x-group3_ID2129_for_TensorFlow/
├── architecture.py
├── architecture_object_detection.py
├── deploy.py
├── deploy_train.py
├── inference_with_trained_model.py
├── input_pipeline.py
├── input_pipeline_color.py
├── model.py
├── model_d.py
├── modifyClassNum.py
├── modelzoo_level.txt
├── predict.py
├── requirements.txt
├── resnet_model_fn.py
├── resnet_v1.py
├── shufflenet.py
├── shufflenet_model_fn.py
├── train.py
├── trainWithImages.py
├── train_shufflenet.py
├── vis.py
├── README.md
└── test
    ├── train_full_1p.sh
    └── train_performance_1p.sh
```


## 脚本参数<a name="section6669162441511"></a>


```
--data_dir                                     数据集路径


```


## 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡训练。
将训练脚本（train_full_1p.sh）中的data_path设置为训练数据集的路径。具体的流程参见“模型训练”的示例。
