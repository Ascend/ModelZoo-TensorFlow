-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [快速上手](#快速上手.md)
-   [迁移学习指导](#迁移学习指导.md)

## 基本信息

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）： Image Generation**

**版本（Version）：1.0**

**修改时间（Modified） ：2021.12.10**

**大小（Size）：84M**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：checkpoint、pbtxt、meta**

**精度（Precision）：Normal**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Research**

**描述（Description）：基于TensorFlow框架的Improved-GAN图像生成网络训练代码** 

<h2 id="概述.md">概述</h2>

Improved-GAN是一个经典的图像生成网络，主要特点是采用各层两两相互连接的Dense Block结构。
- 参考论文：

    [Salimans Tim, Goodfellow Ian. “Improved Techniques for Training GANs” arXiv:1606.03498 [cs]](https://arxiv.org/pdf/1606.03498.pdf) 

- 参考实现：

    [SSGAN-Tensorflow](https://github.com/clvrai/SSGAN-Tensorflow) 

- 适配昇腾 AI 处理器的实现：
  
  [https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/Improved-GAN_ID2094_for_Tensorflow](https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/Improved-GAN_ID2094_for_Tensorflow)      


- 通过Git获取对应commit\_id的代码方法如下：
  
    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    ```

## 默认配置<a name="section91661242121611"></a>

- 训练数据集预处理（以MNIST训练集为例，仅作为用户参考示例）：

  - 图像的输入尺寸为128*128
  - 图像输入格式：TFRecord

- 训练超参

  - Batch size: 32
  - Deconv: bilinear
  - Train step: 40000


## 支持特性<a name="section1899153513554"></a>

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 是    |
| 混合精度  | 否    |
| 并行数据  | 是    |

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
1. 模型训练使用MNIST数据集，数据集请用户自行获取，也可通过如下命令行获取。

```bash
$ python download.py --dataset MNIST
```

2. 数据集训练前需要做预处理操作，请用户参考[Tensorflow-Slim](https://github.com/tensorflow/models/tree/master/research/slim),将数据集封装为tfrecord格式。

3. 数据集处理后，放入模型目录下，在训练脚本中指定数据集路径，可正常使用。
   

## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。

- 启动训练之前，首先要配置程序运行相关环境变量。

  环境变量配置信息参见：

     [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)


- 单卡训练 

  1. 配置训练参数。

     在Pycharm当中使用Modelarts插件进行配置，具体配置如下所示：

     ```
     Boot file path设置为: ./trainer.py
     Code Directory设置为: .
     OBS Path设置为对应项目的工作目录，此项目为：/improvedgan/
     Data Path in OBS设置为OBS当中存放数据的目录,此项目为：/improvedgan/datasets
     其中.代表当前工作目录。
     ```

  2. 启动训练。

     在Modelarts当中单击Apply and Run即可进行训练

- 验证。

    tensorboard当中记录了验证的效果，tensorboard启动流程如下：
    ```
    $ tensorboard --logdir={logdir}
    ```


<h2 id="迁移学习指导.md">迁移学习指导</h2>

- 数据集准备。

  数据集要求如下：

  1. 获取数据。

     如果要使用自己的数据集，需要将数据集放到脚本参数data_dir对应目录下。参考代码中的数据集存放路径如下：

     - 训练集： /improvedgan/datasets/datasets/MNIST
     - 测试集： /improvedgan/datasets/datasets/MNIST

     训练数据集和测试数据集以id.txt加以区分。

     数据集也可以放在其它目录，则修改对应的脚本入参data_dir即可。

  2. 准确标注类别标签的数据集。

  3. 数据集每个类别所占比例大致相同。

  4. 参照tfrecord脚本生成train/eval使用的TFRecord文件。



-   模型训练。

    参考“模型训练”中训练步骤。

-   模型评估。
    
    参考“模型训练”中验证步骤。




## 训练过程<a name="section1589455252218"></a>


1. NPU训练过程：

![NPU](../../../../../image2.png)

部分训练日志如下：
```
[37m[1m[2021-12-10 13:42:13,501]  [train step 44871] Supervised loss: 0.10064 D loss: 0.55640 G loss: -0.00838 Accuracy: 1.00000 (0.021 sec/batch, 1546.661 instances/sec) [0m
[37m[1m[2021-12-10 13:42:13,705]  [train step 44881] Supervised loss: 0.09815 D loss: 0.55605 G loss: -0.00844 Accuracy: 1.00000 (0.020 sec/batch, 1612.361 instances/sec) [0m
[37m[1m[2021-12-10 13:42:13,907]  [train step 44891] Supervised loss: 0.08994 D loss: 0.55556 G loss: -0.00842 Accuracy: 1.00000 (0.020 sec/batch, 1613.097 instances/sec) [0m
[36m[1m[2021-12-10 13:42:14,099]  [val   step 44900] Supervised loss: 0.10700 D loss: 0.72175 G loss: 0.20160 Accuracy: 0.93750 (0.011 sec/batch, 2818.516 instances/sec) [0m
```

2. GPU训练过程

![GPU](../../../../../image.png)


## 精度与性能对比：
GPU型号：Tesla V100-SXM2-16GB
NPU型号：昇腾910

### 精度

以下指标选自训练40k step后，通过tensorboard可视化出来的数据：

|  | GPU | NPU |
|-------|------|------|
| Accuracy | 0.99～1.00 | 0.99～1.00



## 性能

训练和测试阶段根据每秒钟处理的图像数量进行测算，使用本地用X86裸机验证的性能如下：

|  | GPU | NPU |
|-------|------|------|
| 训练阶段 |  2701.319 instance/sec   | 2976 instance/sec
| 推理阶段  |   6354.103 instance/sec   | 6663 instance/sec

## 数据集地址
OBS地址：

obs://improvedgan/datasets/


分享链接：

URL:
https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=InWK64J3BOZ4KtX7oysFMH5MP5pkvOaWWkYRxoz8RP/acK6IXkZFwsgwuf3f0bkLs9tRoc3S0/thaNawdkQSrASbN1ZRApk3anvzsiq34svBwHenVhoJ1zuixtk5bPi6lXkGYuNrmlZs680YFMVn89Cy3GkUiNqhvlYsp8CeMn1+1bKoyra5PuJTT/coL5gwVcCcxx3TXmpFNGcfuIKuvFEDgdljlK+15iZpK3RexyWy3kybwT6gzd60xKQAZGQI8oOguY387Ses1d/Rmd44A2hN1C53XhF4CcN7k2SjvbJS2dt8QAaZnO2EVj8zSLMfdp3KXIzVwlslnJKhUV+kA4+okQmSjFOEHynrzyp3SfNVwlLnp1+zhHOqCZlE+fMTnz+rNx4qmDKk6xl/c/ocazuBPB+GU4t+VfrSMuZCqjQo16RaIVwnEDZsV/A7rNIHFLUlsS3/E+RetM3iwhvKZKNIS72OKOV//cZkqL6GQiZE8rcThnCU0rB+zQUFxj+vU+9odXVlkNrcGlkbfU6IXhlUAKlsH6Y16lnYOOeGZre0o1XQg8lks/MP5Ue/D4imiNmd2MOewQ4ZCXdoVFrEF42F9d8v3HA2h3OrptMkv5Q=

提取码:
111111

*有效期至: 2022/12/24 10:51:44 GMT+08:00