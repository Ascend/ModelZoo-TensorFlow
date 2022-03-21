- [基本信息](#基本信息.md)
- [概述](#概述.md)
- [训练环境准备](#训练环境准备.md)
- [快速上手](#快速上手.md)
- [迁移学习指导](#迁移学习指导.md)
- [高级参考](#高级参考.md)  

<h2 id="基本信息.md">基本信息</h2>  
**发布者（Publisher）：Huawei**

**应用领域（Application Domain）： Image Classification**

**版本（Version）：1.1**

**修改时间（Modified） ：2021.12.21**

**大小（Size）：1183kb**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：FP32**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Research**

**描述（Description）：基于TensorFlow框架的DeepFaceLab网络训练代码** 

<h2 id="概述.md">概述</h2>

DeepFaceLab是一个由iperov开发的开源换脸deepfake系统，其提供了一个简单易用的pipeline，使得用户可以方便的进行换脸操作。

- 本文的主要贡献有三个：  
  - 提出了一个包括成熟的pipeline的最先进的框架，目标是获得逼真的人脸交换结果；  
  - 2018的DeepFaceLab开源代码，并且一直在跟进cv领域的进展，为主动和被动防御deepfakes作出积极贡献，它在开源论坛和VFX领域得到了广泛的注意；  
  - 介绍了一些DeepFaceLab里高性能的组件和工具，从此用户可以要求更灵活的DeepFaceLab工作流程，同时及时发现问题。  

- 网络架构：  
![img](./Img4Doc/dfnet.png)  

![img](./Img4Doc/conversion.png)

- 架构特点：  
  - 使用pixelshuffle (depth2space) 进行上采样，而不是反卷积和双线性插值。这样做的好处就是可以消除生成图片边界效应；
  - 解码模块采用残差连接方式，融合更多的特征；  
  - 最后一层的输出，采用sigmoid归一化输出为0--1之间，而不是采用tanh归一化到-1--1之间。

- 相关参考：  
    - 参考论文：[DeepFaceLab: A simple, flexible and extensible face swapping framework](https://arxiv.org/abs/2005.05535)
    - 参考实现：[https://github.com/iperov/DeepFaceLab](https://github.com/iperov/DeepFaceLab)

## 默认配置<a name="section91661242121611"></a>
-   训练超参（单卡）：
    - Batch size: 4 (set as default)
    - num_epochs: 20000 (set as default)
## 支持特性<a name="section1899153513554"></a>

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 否  |
| 混合精度  | 是   |
| 数据并行  | 是   |

## 混合精度训练<a name="section168064817164"></a> 
昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

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
3. NPU环境
```
硬件环境：NPU: 1*Ascend 910 CPU: 24*vCPUs 96GB

运行环境：ascend-share/5.0.4.alpha002_tensorflow-ascend910-cp37-euleros2.8-aarch64-training:1.15.0-21.0.2_1125
```
4. 第三方依赖
```
tqdm
numpy==1.19.3
numexpr
h5py==2.10.0
opencv-python==4.1.0.25
scikit-image
scipy==1.4.1
colorama
pyqt5
tf2onnx
```
<h2 id="快速上手.md">快速上手</h2>
- 数据集准备  
训练数据集为FaceForensics++数据集中经S3FD人脸特征提取之后的人脸，已经上传至obs中，obs路径如下
```
- obs path: /dflobs/
- data path in obs: /dflobs/dataset/
```
若需要训练其他的人脸，则需要参考【概述】中【相关参考】【参考实现】的linux版本，分别运行[4_data_src_extract_faces_S3FD.sh]以及[5_data_dst_extract_faces_S3FD.sh]脚本，获得相应人脸的训练数据集。

## 模型训练<a name="section715881518135"></a>  
- 单击“立即下载”，并选择合适的下载方式下载源码包。

- 启动训练之前，首先要配置程序运行相关环境变量。

  环境变量配置信息参见：

     [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)
- 配置训练参数
用户可以在工程目录下train.py文件中修改参数（目标迭代次数：target_iters），默认值已设为20000次。

- 启动训练(单卡)
    - 命令行方式训练：bash npu_train.sh
    - 采用ModelArts进行训练：
    数据和代码准备完成后，您可以创建一个训练作业，选用Ascend-Powered-Engine→tensorflow1.15引擎，基于本地的modelarts_entry.py训练脚本，并最终生成一个可用的模型。
    具体步骤为：  
```
a. 在PyCharm工具栏中，选择"ModelArts > Edit Training Job Configuration"。
b. 在弹出的对话框中，按照如下示例配置训练参数。
   * "Job Name": 自动生成，首次提交训练作业时，该名称也可以自己指定。
   * "AI Engine”: 选择"Ascend-Powered-Engine"，版本为"tensorflow1.15"。
   * "Boot File Path": 选择本地的训练脚本"modelarts_entry.py"。
   * "Code Directory": 选择训练脚本所在的目录。
   * "Image Path": ascend-share/5.0.4.alpha002_tensorflow-ascend910-cp37-euleros2.8-aarch64-training:1.15.0-21.0.2_1125
   * "OBS Path": obs路径，/dflobs/
   * "Data Path in OBS":  obs中训练数据集的路径，/dflobs/dataset/
   * "Specifications": 选择NPU规格，NPU: 1*Ascend 910 CPU: 24*vCPUs 96GB
   * "Compute Nodes": 选择1
c. 选择Apply and Run
```
&ensp; ModelArts训练作业配置如下图所示：  
![img](./Img4Doc/modelarts_cfg.png)

<h2 id="迁移学习指导.md">迁移学习指导</h2>
- 数据集准备。

    1.  获取数据。
        请参见“快速上手”中的数据集准备。
-   模型训练。

    参考“模型训练”中训练步骤。

<h2 id="高级参考.md">高级参考</h2>
## 部分代码结构说明<a name="section08421615141513"></a>
```
├── modelarts_entry.py         // modelarts启动脚本
├── npu_train.sh               // 单卡训练启动脚本                    
├── ssim.py                    // 计算训练指标精度            
├── train.py                   // 网络训练与测试代码
```

## 脚本参数<a name="section6669162441511"></a>
```
--training_data_src            //  训练所有的src数据集（src指提供脸的人）
--training_data_dst            //  训练所用的dst数据集（dst指要换脸的人）
--input_dir                    //  换脸所需要的dst人脸文件夹
--output_dir                   //  换脸结果存放文件夹
--output_mask_dir              //  换脸的mask存放文件夹
--model_dir                    //  训练生成的ckpt模型文件夹
```
## 训练过程及结果<a name="section1589455252218"></a>
默认训练20000次后，可在obs/训练作业/output/model文件夹下得到相应的网络模型，在训练日志中得到相应的SSIM指标参数。  

| 迁移模型    | 训练次数 | NPU精度 |GPU精度 |
| :---------- | ----- | ------ | ------ |
| DeepFaceLab | 20000 | 0.6907±0.0259|0.6556±0.0354|

训练log截图:  

NPU： Ascend910

![img](./Img4Doc/npu_log.png)  

GPU：  Tesla V100  

![img](./Img4Doc/GPU.png)  




