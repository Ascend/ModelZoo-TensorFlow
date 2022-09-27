<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：** Image Classification 

**版本（Version）：1.1**

**修改时间（Modified） ：2021.11.12**

**大小（Size）：20M**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Research**

**描述（Description）：基于TensorFlow框架的EfficientNet图像分类网络训练代码** 

<h2 id="概述.md">概述</h2>

EfficientNets是谷歌大脑的工程师谭明星和首席科学家Quoc V. Le在论文《EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks》中提出。该模型的基础网络架构是通过使用神经网络架构搜索（neural architecture search）设计得到。卷积神经网络模型通常是在已知硬件资源的条件下，进行训练的。当你拥有更好的硬件资源时，可以通过放大网络模型以获得更好的训练结果。为系统的研究模型缩放，谷歌大脑的研究人员针对EfficientNets的基础网络模型提出了一种全新的模型缩放方法，该方法使用简单而高效的复合系数来权衡网络深度、宽度和输入图片分辨率。

- 参考论文：

    [Mingxing Tan, Quoc V. Le. “EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.” arXiv:1905.11946](https://arxiv.org/pdf/arXiv:1905.11946.pdf) 

- 参考实现：

    https://github.com/mingxingtan/efficientnet

- 适配昇腾 AI 处理器的实现：
    
        
    https://gitee.com/ascend/modelzoo/tree/master/contrib/TensorFlow/Research/cv/EfficientNet/EfficientNet_V2_ID1220_for_TensorFlow
        


- 通过Git获取对应commit\_id的代码方法如下：
    
    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    ```

- 精度

|                | 论文   | ascend |
|----------------|------|--------|
| TOP-1 Accuracy | 0.771 | 0.766 |

- 性能

| batchsize  |image_size| GPU （v100）  |  NPU |
|---|---|---|---|
| 50  |224 x 224|  0.2667 s/step | 0.1168 s/step |

注：模型型号为EfficientNetB0，由于imagenet 数据集较大（共有128万张图片）,每训练一个epoch所需的时间为3小时，完整训练一次预估需要300小时左右，时间较长且因调参需要多次反复训练，故决定加载预训练模型，减少模型训练时间。

## 默认配置

- 训练数据集预处理（以ImageNet训练集为例，仅作为用户参考示例）：

  - 图像的输入尺寸为224*224
  - 图像输入格式：TFRecord
  - 随机裁剪图像尺寸
  - 随机水平翻转图像
  - 输入图像进行归一化至[0,1]

- 测试数据集预处理（以ImageNet2012验证集为例，仅作为用户参考示例）

  - 图像的输入尺寸为224*224（在中心裁剪83%的图像，然后缩放至224\*224）
  - 图像输入格式：TFRecord
  - 输入图像进行归一化至[0,1]

- 训练超参

  - Batch size: 50
  - Learning rate(LR): 0.0001
  - Optimizer: SGD
  - Nesterov: True
  - Train epoch: 1
  - Loss: softmax_cross_entropy

## 支持特性

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 |  否   |
| 混合精度  |  否  |
| 并行数据  |  否  |


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
1. 模型训练使用ImageNet数据集，数据集请用户自行获取。

2. 数据集训练前需要做预处理操作，请用户参考[Tensorflow-Slim](https://github.com/tensorflow/models/tree/master/research/slim),将数据集封装为tfrecord格式。

3. 数据集处理后，放入模型目录下，在训练脚本中指定数据集路径，可正常使用。
   

## 模型训练

- 单击“立即下载”，并选择合适的下载方式下载源码包。

- 启动训练之前，首先要配置程序运行相关环境变量。

  环境变量配置信息参见：

     [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)

- 单卡训练 

  1. 配置训练参数。

     首先在脚本train_1p.sh中，配置训练数据集路径，请用户根据实际路径配置，数据集参数如下所示：

     ```
      --TMP_DATA_PATF=./data
     ```

  2. 启动训练。

     启动单卡训练 （脚本为EfficientNet_V2_ID1220_for_TensorFlow/train_1p.sh） 

     ```
     bash train_1p.sh
     ```


- 验证。

    1. 测试的时候，需要修改test.py脚本路径参数（脚本位于EfficientNet_V2_ID1220_for_TensorFlow/test.py），配置测试集所在路径以及checkpoint文件所在路径，请用户根据实际路径进行修改。

          ```
          TMP_DATA_PATH = './data/valid'
          TMP_MODEL_PATH = './model/m.ckpt-0'
          ```

  2. 测试指令（脚本位于EfficientNet_V2_ID1220_for_TensorFlow/test_1p.sh）

      ```
      bash test_1p.sh
      ```

<h2 id="迁移学习指导.md">迁移学习指导</h2>

- 数据集准备。

  数据集要求如下：

  1. 获取数据。

     如果要使用自己的数据集，需要将数据集放到脚本参数TMP_DATA_PAT对应目录下。参考代码中的数据集存放路径如下：

     - 训练集： ./data/train_tf
     - 测试集： ./data/valid

     数据集也可以放在其它目录，则修改对应的脚本入参TMP_DATA_PAT即可。

  2. 准确标注类别标签的数据集。

  3. 数据集每个类别所占比例大致相同。

  4. 参照tfrecord脚本生成train/eval使用的TFRecord文件。

  5. 数据集文件结构，请用户自行制作TFRecord文件，包含训练集和验证集两部分，目录参考“1.获取数据”部分

-   模型训练。

    参考“模型训练”中训练步骤。

-   模型评估。
    
    参考“模型训练”中验证步骤。

<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码

```
Efficientnet_V2
└─ 
  ├─README.md
  ├─data用于存放训练数据集 obs://public-dataset/imagenet/orignal/
  	├─train_tf
  	└─valid_tf
  ├─model 用于训练模型 
  	├─m.ckpt
  	└─...
  ├─weights 用于存放预训练模型 obs://efficientnet-v2/weights/
  	├─model.ckpt
  	└─...
  ├─efficientnet_builder.py
  ├─efficientnet_model.py
  ├─preprocess.py
  ├─train.py
  ├─test.py
  ├─train_1p.sh 模型的启动脚本，
  ├─test_1p.sh 模型的启动测试脚本
```

## 脚本参数

```
--is_training             是否进行训练的标志，默认是True
--epochs                  训练的epoch数，默认是1
--image_num               训练集图片数量，默认是1281167
--batch_size              训练的batch大小，默认是90
--TMP_DATA_PATH           训练集文件的存放路径，默认是./data
--TMP_MODEL_PATF          训练完成的模型存放路径，默认是./model
--TMP_LOG_PATH            训练的日志存放路径，默认是./log
--TMP_WEIGHTS_PATH        预训练模型存放路径，默认是./data1/NRE_Check/wx1056345/ID1220_Efficientnet_V2/weights
```


## 训练过程

1.  通过“模型训练”中的训练指令启动单卡训练。

2.  参考脚本的模型存储路径为./model。


## 推理/验证过程

1.  通过“模型训练”中的测试指令启动测试。

2.  当前只能针对该工程训练出的checkpoint进行推理测试。

3.  推理脚本test.py的参数TMP_MODEL_PATF可以配置为checkpoint所在的文件夹路径

4.  测试结束后会打印验证集的top1 accuracy。
