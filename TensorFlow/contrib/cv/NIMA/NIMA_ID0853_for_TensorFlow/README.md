<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Image Assessment**

**版本（Version）：1.0**

**修改时间（Modified） ：2001.11.12**

**大小（Size）：37M**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：h5**

**处理器（Processor）：昇腾910**

**精度（Precision）：Mixed**

**应用级别（Categories）：Research**

**描述（Description）：基于TensorFlow框架的NIMA图像质量评估训练代码** 

<h2 id="概述.md">概述</h2>

NIMA模型是NIMA: Neural Image Assessment论文的Tensorflow+Keras的实现，基于最新的深度物体识别（object detection）神经网络，能够从直接观感（技术角度）和吸引程度（美学角度）预测人类对图像的评估意见的分布。模型具有预测打分与人类主观打分很相近的优点，因此可用作自动检查图像质量的工具或作为损失函数进一步提高生成图像的质量。 

- 参考论文：

    [Hossein Talebi, Peyman Milanfar. “NIMA: Neural Image Assessment.” arXiv:1709.05424](https://arxiv.org/pdf/1709.05424.pdf) 

- 参考实现：
    https://github.com/titu1994/neural-image-assessment

- 适配昇腾 AI 处理器的实现：
    https://gitee.com/ascend/modelzoo/tree/master/contrib/TensorFlow/Research/cv/NIMA/NIMA_ID0853_for_TensorFlow

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
| SRCC | 0.510 | 0.517  |

- 性能

| batchsize  |image_size| GPU （v100）  |  NPU |
|---|---|---|---|
| 100  | 224 X 224 |0.3368 s/step | 0.1459 s/step |

注：由于AVA_Dataset数据集较大（共有28万张图片）,每训练一个epoch所需的时间为4小时，完整训练一次预估需要400小时左右，时间较长且因调参需要多次反复训练，故决定加载预训练模型，减少模型训练时间。

## 默认配置

- 训练数据集预处理（以AVA_Dataset训练集为例，仅作为用户参考示例）：

  - 图像的输入尺寸为224\*224（将图像缩放到256\*256，然后随机裁剪图像）
  - 图像输入格式：JPEG
  - 随机水平翻转图像
  - 对输入图像归一化至[-1,1]

- 测试数据集预处理（以AVA_Dataset验证集为例，仅作为用户参考示例）

  - 图像的输入尺寸为224\*224（将图像缩放到224\*224）
  - 图像输入格式：JPEG
  - 对输入图像归一化至[-1,1]

- 训练超参

  - Train epoch: 7  
  - Batch size: 100
  - Learning rate(LR): 0.0001
  - Optimizer: SGD
  - Weight Decay: 1e-6
  - Momentum: 0.9
  - Nesterov: True
  - Loss: earth_mover_loss

## 支持特性

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 |  否  |
| 混合精度  |  否   |
| 并行数据  |  否   |



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
1. 模型训练使用AVA Dataset数据集，数据集请用户自行获取。

2. 数据集获取后，放入模型目录下，在训练脚本中指定数据集路径，可正常使用。

## 模型训练

- 单击“立即下载”，并选择合适的下载方式下载源码包。

- 启动训练之前，首先要配置程序运行相关环境变量。

  环境变量配置信息参见：

     [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)

- 单卡训练 

  1. 配置训练参数。

     首先在/util/data_loader.py中，配置训练数据集路径，请用户根据实际路径配置，数据集参数如下所示：

     ```
      base_images_path = r'/data2/NRE_Check/NIMA/NIMA_tf_xiaoqiqiya/AVA_dataset/AVA_dataset/image/images/'
      ava_dataset_path = r'/data2/NRE_Check/NIMA/NIMA_tf_xiaoqiqiya/AVA_dataset/AVA_dataset/AVA.txt'
     ```

  2. 启动训练。

     启动单卡训练 （脚本为NIMA_ID0853_for_TensorFlow/train_1p.sh） 

     ```
     bash train_1p.sh
     ```

- 验证

    1. 测试的时候，需要修改脚本路径参数（脚本位于NIMA_ID0853_for_TensorFlow/eval.py），配置checkpoint文件所在路径，请用户根据实际路径进行修改。

          ```
          TMP_MODEL_PATF = './model/'
          model = keras.models.load_model(TMP_MODEL_PATF+"model_007.h5", custom_objects={'earth_mover_loss': earth_mover_loss})
          ```

  2. 测试指令（脚本位于NIMA_ID0853_for_TensorFlow/test_1p.sh）

      ```
      bash test_1p.sh
      ```

<h2 id="迁移学习指导.md">迁移学习指导</h2>

- 数据集准备。

  数据集要求如下：

  1. 获取数据。

     如果要使用自己的数据集，需要将数据集放到data_loader.py脚本参数base_images_path以及ava_dataset_path对应目录文件下。参考代码中的数据集存放路径如下：

     - 数据文件： '/data2/NRE_Check/NIMA/NIMA_tf_xiaoqiqiya/AVA_dataset/AVA_dataset/image/images/'
     - 标签文件： '/data2/NRE_Check/NIMA/NIMA_tf_xiaoqiqiya/AVA_dataset/AVA_dataset/AVA.txt'

     也可自行制作tfrecoed文件，并更改data_loader.py的数据读取流程。

     数据集也可以放在其它目录，则修改对应的脚本入参base_images_path以及ava_dataset_path即可。

  2. 准确标注类别标签的数据集。

  3. 数据集每个类别所占比例大致相同。

- 模型修改

- 加载预训练模型。 
       模型加载修改，修改文件train.py，修改以下代码行。

        ```
        TMP_WEIGHTS_PATH = './weights'
        base_model = MobileNet((image_size, image_size, 3), alpha=1, include_top=False, pooling='avg', weights=None)
        base_model.load_weights(os.path.join(TMP_WEIGHTS_PATH+"/mobilenet_1_0_224_tf_no_top.h5"), by_name=True)
        ```

-   模型训练。

    参考“模型训练”中训练步骤。

-   模型评估。
    
    参考“模型训练”中验证步骤。

<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码

```
NIMA
└─ 
  ├─README.md
  ├─AVA_dataset 用于存放训练数据集 obs://public-dataset/AVA_dataset/
  	├─image
  	└─...
  ├─model 用于训练模型 
  	├─model_001.h5
  	└─...
  ├─weights 用于存放预训练模型  obs://nima-mobilenet/weights/
  	├─mobilenet_1_0_224_tf_no_top.h5
  	└─...
  ├─utils
    ├─score_utils.py
    ├─data_loader.py
    ├─check_dataset.py
  	└─...
  ├─train_1p.sh 模型的启动脚本，
  ├─test_1p.sh 模型的启动测试脚本
```

## 脚本参数

```
--epochs                 训练epoch次数，默认：7
--batchsize              NPU的batch size，默认：100
--train_size             训练集图片数量，默认：250502
--val_size               验证机图片数量，默认：5000
```

## 训练过程

1.  通过“模型训练”中的训练指令启动单卡。

2.  参考脚本的模型存储路径为./model，训练脚本log中包括如下信息。

```
这里是训练日志
```

## 推理/验证过程

1.  通过“模型训练”中的测试指令启动测试。

2.  当前只能针对该工程训练出的checkpoint进行推理测试。

3.  eval.py的参数TMP_MODEL_PATF可以配置为checkpoint所在的文件夹路径，则会根据该路径下的model_007.h5进行推理，也可在
model = keras.models.load_model(TMP_MODEL_PATF+"model_007.h5", custom_objects={'earth_mover_loss': earth_mover_loss})中更改要进行推理的checkpoint。

4.  测试结束后会打印验证集的SRCC。



