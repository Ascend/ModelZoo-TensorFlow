## 基本信息

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：** Classification

**版本（Version）：1.1**

**修改时间（Modified） ：2022.05.15**

**大小（Size）：**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：**

**精度（Precision）：**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Research**

**描述（Description）：基于TensorFlow框架的VisionTransformer图像分类网络训练代码** 

## 概述 

当前Transformer模型被大量应用在NLP自然语言处理当中，而在计算机视觉领域，Transformer的注意力机制attention也被广泛应用，比如Se模块，CBAM模块等等注意力模块，这些注意力模块能够帮助提升网络性能。而VisionTransformer展示了不需要依赖CNN的结构，也可以在图像分类任务上达到很好的效果，并且也十分适合用于迁移学习。


- 参考论文：

    Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ArXiv, abs/2010.11929.


  - arXiv:2010.11929(http://xxx.itp.ac.cn/pdf/2010.11929.pdf) 

- 参考实现：

    https://github.com/faustomorales/vit-keras 

- 适配昇腾 AI 处理器的实现：
    
        
    https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/BicycleGAN_ID1287_for_TensorFlow
        


- 通过Git获取对应commit\_id的代码方法如下：
    
    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    ```

- 精度

|       | GPU   | NPU   |
|-------|-------|-------|
| ACC | 0.8709 | 0.871 |

- 性能

| batchsize | image_size | GPU （v100） |  NPU |
|-----------|------------|---|---|
| 4         | 384*384  | 0.96s | 0.19s |


## 默认配置

- 训练数据集预处理（以原论文的maps训练集为例，仅作为用户参考示例）：

  - 图像的输入尺寸为384*384
  - 图像输入格式：从cifar压缩文件中读取数据 
```python
    def _preprocess(image, label):
        image = tf.image.resize(image, (image_size, image_size))
        image = (image - 127.5) / 127.5
        return image, label
```

- 测试数据集预处理（以原论文的maps验证集为例，仅作为用户参考示例）

  - 图像的输入尺寸为384*384
  - 图像输入格式：从cifar压缩文件中读取数据 
```python
    def _preprocess(image, label):
        image = tf.image.resize(image, (image_size, image_size))
        image = (image - 127.5) / 127.5
        return image, label
```

- 训练超参

  - Batch size: 4
  - Learning rate(LR): 0.001
  - Optimizer: Adam
  - Train epoch: 1

## 支持特性

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 |  否   |
| 混合精度  |  是  |
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


## 注意事项 
1. 本项目是在104机器裸机上完成的，线上modelart的总是报算子错。  
2. vision transformer 需要加载预训练才能达到好的效果，这是是加载了gpu训练的模型。 

<h2 id="快速上手.md">快速上手</h2>

- 数据集准备
1. 104机器上位置为 `/home/HwHiAiUser/wubo/vit/vit-ckpt/datasets`  另外obs也有  

2. 获得数据集后，放入模型目录下，在训练脚本中指定数据集路径，可正常使用。
   

## 模型训练

- 单击“立即下载”，并选择合适的下载方式下载源码包。

- 启动训练之前，首先要配置程序运行相关环境变量。

  环境变量配置信息参见：

     [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

- 单卡训练 

  1. 配置训练参数。

     首先在脚本test/train_full_1p.sh中，配置训练数据集路径，请用户根据实际路径配置，数据集参数如下所示：

     ```
      --data_path ./dataset
     ```

  2. 启动训练。

     启动单卡训练 （脚本为modelarts_entry_acc.py） 

     ```
     python3 modelarts_entry_acc.py 
     ```

 

<h2 id="迁移学习指导.md">迁移学习指导</h2>

- 数据集准备。

  数据集要求如下：

  1. 获取数据。

     如果要使用自己的数据集， obs://cann-id1217/  需要将数据集放到脚本参数data_path对应目录下。参考代码中的数据集存放路径如下：

     - 训练集： ./dataset/train
     - 测试集： ./dataset/val

     数据集也可以放在其它目录，则修改对应的脚本入参data_path即可。


-   模型训练。

    参考“模型训练”中训练步骤。

-   模型评估。
    
    参考“模型训练”中验证步骤。

<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码

```
VisionTransformer
└─ 
  ├─README.md
  ├─dataset用于存放训练数据集和预训练文件
      |-model/
  	├─train
  	└─val
  ├─precision_tool 用来去掉某一些算子融合策略 
  	└─...
  ├─test 用于测试
      ├─output 用于存放测试结果和日志 
  	└─test_1p.sh
  ├─vit_allpipeline_fusion_accelerate.py 精度测试脚本
  ├─vit_allpipeline_performance.py 性能测试脚本
```

## 脚本参数

```
--learning_rate          学习率,默认是0.001
--batch_size             训练的batch大小，默认是4
--data_path              训练集文件路径
--output_path            日志,模型文件等存放的路径
```


## 训练过程

1.  通过“模型训练”中的训练指令启动单卡训练。

2.  参考脚本的模型存储路径为./output。



## 推理/验证过程



