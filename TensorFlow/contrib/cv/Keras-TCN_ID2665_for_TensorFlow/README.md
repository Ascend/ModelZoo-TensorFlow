-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [快速上手](#快速上手.md)
-   [训练结果](#训练结果.md)
-   [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Computer Vision** 

**修改时间（Modified） ：2022.03.31**

**大小（Size）：1.18 MB**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：h5**

**精度（Precision）：fp32**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Research**

**描述（Description）：基于TensorFlow框架验证时域卷积网络长期依赖性的训练代码** 

<h2 id="概述.md">概述</h2>


在序列建模中我们通常使用循环和递归结构，但最近的研究表明，特殊的卷积结构在一些任务中表现得更好。时域卷积网络（TCN）主要包括两个部分：一维的全卷积网络和因果卷积。选择The adding problem测试网络的长期依赖性。  

The adding problem：输入一个深度为2，长度为n的序列，其中一维是取值在[0,1]内的实数，另一维除了两个元素为1，其余元素均为0。将这两维数字做内积，最终得到的结果是一个实数，取值范围是[0,2]。使用TCN网络预测结果，与真实值比较，误差使用MSE衡量。

- 参考论文：

    http://export.arxiv.org/pdf/1803.01271

- 参考实现：

    https://github.com/philipperemy/keras-tcn 

- 适配昇腾 AI 处理器的实现：
    
    https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/Keras-TCN_ID2665_for_TensorFlow
        


- 通过Git获取对应commit\_id的代码方法如下：
    
    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    ```

## 默认配置<a name="section91661242121611"></a>

- Input shape

  - 3D tensor with shape `(batch_size, timesteps, input_dim)`.

- Output shape

  - 2D tensor with shape `(batch_size, nb_filters)`.

- 训练超参

  - Batch size： 512
  - Train epoch: 50
  - Train step: 391


## 支持特性<a name="section1899153513554"></a>

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 否    |
| 混合精度  | 否    |
| 并行数据  | 否    |


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
  
由`utils.py`生成  
Length of the adding problem data = 600  
\# of data in the train set = 200000  
\# of data in the validation set = 40000

## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。

- 启动训练之前，首先要配置程序运行相关环境变量。

  环境变量配置信息参见：

     [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

- 单卡训练 

  1. 配置训练参数。

     首先在脚本test/train_full_1p.sh中，配置batch_size、learning_rate、dropout_rate、epochs等参数，请用户根据实际路径配置路径，或者在启动训练的命令行中以参数形式下发。


  2. 启动训练。

     启动单卡训练 （脚本为Keras-TCN_ID2665_for_TensorFlow/test/train_full_1p.sh） 

     ```
     bash train_full_1p.sh 
     ```

<h2 id="训练结果.md">训练结果</h2>

- 精度结果比对

|精度指标项|论文发布|GPU实测|NPU实测|
|---------|--------|------|-------|
|loss|6.9630e-04|2.2935e-04|4.7023e-04|
|val_loss|3.7180e-04|2.1177e-04|5.4636e-04|

GPU loss：   
<div align=center>
<img src=https://s3.bmp.ovh/imgs/2021/09/92fc0134a77c3760.png />
</div>
GPU val_loss:
<div align=center>
<img src=https://s3.bmp.ovh/imgs/2021/09/3367b1149e398fe8.png />
</div> 
NPU loss：
<div align=center>
<img src=https://s3.bmp.ovh/imgs/2021/09/3b969c09def2440a.png />
</div>
NPU val_loss:
<div align=center>
<img src=https://s3.bmp.ovh/imgs/2021/09/79d457fbd8e533c6.png />
</div>          

- 性能结果比对  

|性能指标项|GPU实测|NPU实测|
|---------|-------|-------|
|1 step|347ms|143ms|


<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码<a name="section08421615141513"></a>

```
├── train.py                              //网络训练与测试代码
├── README.md                             //代码说明文档
├── freeze_graph.py                       //训练模型固化为pb模型代码
├── utils.py　　　　　　　　　　　　　　　   //数据生成代码
├── tcn.py　　　　　　　　　　　            //模型代码
├── requirements.txt                      //训练python依赖列表
├── test
│    ├──train_performance_1p.sh           //单卡训练验证性能启动脚本
│    ├──train_full_1p.sh                  //单卡全量训练启动脚本

```

## 脚本参数<a name="section6669162441511"></a>

```
--batch_size             每个NPU的batch size，默认：512
--learing_rata           初始学习率，默认：0.005
--dropout_rate           随机失活的概率，默认：0.1
--epochs                 训练epcoh数量，默认：50
```

## 训练过程<a name="section1589455252218"></a>

1.  通过“模型训练”中的训练指令启动单卡卡训练。

2.  参考脚本的模型存储路径为./output/tcn.h5。

链接：   
[GPU训练精度性能完整数据及日志 提取码：1234](https://pan.baidu.com/s/1sHniPqIwLn7VC2lWdQK0wQ)   

[NPU训练精度性能完整数据及日志 提取码：1234](https://pan.baidu.com/s/18nDU6eFti0vAeTSapk_iVw)