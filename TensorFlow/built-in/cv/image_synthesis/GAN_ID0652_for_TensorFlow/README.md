# GAN_for_TensorFlow
## 目录
* [基本信息](#基本信息)
* [概述](#概述)
* [训练环境准备](#训练环境准备)
* [快速上手](#快速上手)
* [高级参考](#高级参考)


## 基本信息
- 发布者（Publisher）：huawei
- 应用领域（Application Domain）：Image Synthesis
- 版本（Version）：1.1
- 修改时间（Modified） ：2021.06.22
- 大小（Size）：1.39M
- 框架（Framework）：TensorFlow 1.15.0
- 模型格式（Model Format）：ckpt
- 精度（Precision）：Mixed
- 处理器（Processor）：昇腾910
- 应用级别（Categories）：Official
- 描述（Description）：基于TensorFlow框架的GAN学习算法训练代码

## 概述
-     生成式对抗网络（GAN, Generative Adversarial Networks ）是一种[深度学习](https://baike.baidu.com/item/深度学习/3729729)[模型](https://baike.baidu.com/item/模型/1741186)，是近年来复杂分布上[无监督学习](https://baike.baidu.com/item/无监督学习/810193)最具前景的方法之一。模型通过框架中（至少）两个模块：生成模型（Generative Model）和判别模型（Discriminative Model）的互相[博弈](https://baike.baidu.com/item/博弈/4669968)学习产生相当好的输出。 

-   参考实现：
    
        ```
        https://github.com/bojone/gan
        ```
    
-   适配昇腾 AI 处理器的实现
    
  
    ````
    ```
    https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/built-in/cv/image_synthesis/GAN_ID0652_for_TensorFlow
    ```
    ````
    
    
    
-   通过Git获取对应commit\_id的代码方法如下：
    
        ```
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
        ```

### 默认配置<a name="section91661242121611"></a>
-   网络结构
    -   初始学习率为1e-4
    -   优化器：RMSprop
    -   单卡batchsize：128
    -   总Epoch数设置为15
    
-   训练数据集预处理（当前代码以MNIST_data/train为例，仅作为用户参考示例）：
    -   图像的输入尺寸为224\*224
    -   随机裁剪图像尺寸
    -   随机水平翻转图像
    -   根据平均值和标准偏差对输入图像进行归一化

-   训练超参（单卡）：
    -   Batch size: 128
    -   Momentum: 0.9
    -   LR scheduler: cosine
    -   Learning rate\(LR\): 1e-4
    -   Weight decay: 0.0001
    -   Label smoothing: 0.1
    -   Train epoch: 15


### 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 否       |
| 混合精度   | 是       |
| 数据并行   | 否       |


### 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

### 开启混合精度<a name="section20779114113713"></a>
相关代码示例

```
sess_config = tf.ConfigProto()
custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["dynamic_input"].b = 1
custom_op.parameter_map["dynamic_graph_execute_mode"].s = tf.compat.as_bytes("lazy_recompile")
custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes(args.precision_mode)
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
### 数据集准备

1.模型训练使用**MNIST_data**数据集，数据集请用户自行获取，或者参考链接https://github.com/bojone/gan中获取

2.参考链接里的数据集训练前需要做预处理操作，将压缩包解压并提取png文件

3.数据集处理后，放入模型目录下，在训练脚本中指定数据集路径，可正常使用

### 模型训练<a name="section715881518135"></a>
- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 启动训练之前，首先要配置程序运行相关环境变量。

​       环境变量配置信息参见：

   [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)

- 单卡训练

  - 配置训练参数

  首先在test/train_full_1p.sh中配置训练数据集路径，请用户根据实际路径配置

  ```
  --data_dir=/opt/npu/mnistdata
  ```

  - 启动单卡训练 （脚本为GAN_ID0652_for_TensorFlow/test/train_full_1p.sh） 

    ```
    bash train_full_1p.sh
    ```

## 高级参考

### 脚本和示例代码<a name="section08421615141513"></a>

    ├── README.md                                //说明文档
    ├── requirements.txt						 //依赖
    ├──test										 
    │    ├──train_full_1p.sh				 //单卡训练脚本
    ├──imle.py                   	         //入口训练脚本


### 脚本参数<a name="section6669162441511"></a>

```
    --data_path                       train data dir, default : path/to/data
    --epochs                          number of train epochs
    --batch_size                      mini-batch size ,default: 128 
    --precision_mode                  precision_mode,default:allow_fp32_to_fp16
```

### 训练过程<a name="section1589455252218"></a>

NA


### 推理/验证过程<a name="section1465595372416"></a>

NA
