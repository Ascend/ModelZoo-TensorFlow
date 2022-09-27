
# SESEMI_for_TensorFlow
## 目录


## 基本信息

发布者（Publisher）：Huawei

应用领域（Application Domain）： Image Classification

版本（Version）：1.2

修改时间（Modified） ：2021.11.18

大小（Size）：25.3MB

框架（Framework）：TensorFlow 1.15.0

模型格式（Model Format）：h5

精度（Precision）：Mixed

处理器（Processor）：昇腾910

应用级别（Categories）：Research

描述（Description）：基于TensorFlow框架的MASF图像分类网络训练代码


## 概述
SESEMI的工作属于半监督学习(SSL)的框架，在图像分类的背景下，它可以利用大量的未标记数据，在有限的标记数据中显著改进监督分类器的性能。具体来说，我们利用自监督损失项作为正则化(应用于标记数据)和SSL方法(应用于未标记数据)类似于一致性正则化。尽管基于一致性正则化的方法获得了最先进的SSL结果，但这些方法需要仔细调优许多超参数，在实践中通常不容易实现。为了追求简单和实用，我们的模型具有自监督正则化，不需要额外的超参数来调整最佳性能。


    参考实现：
    
    适配昇腾 AI 处理器的实现：
    
    https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/graph/SESEMI_ID1270_for_TensorFlow
    
    通过Git获取对应commit_id的代码方法如下：
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换


​    
### 默认配置<a name="section91661242121611"></a >

    -   训练数据集：
          -  数据集采用cifar-10数据集
    
    -   测试数据集：
          -  测试数据集与训练数据集相同,使用cifar-10数据集


### 支持特性<a name="section1899153513554"></a >
| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 否       |
| 混合精度   | 否       |
| 数据并行   | 否       |


## 训练环境准备
1.  硬件环境准备请参见各硬件产品文档"[驱动和固件安装升级指南]( https://support.huawei.com/enterprise/zh/category/ai-computing-platform-pid-1557196528909)"。需要在硬件设备上安装与CANN版本配套的固件与驱动。
2.  宿主机上需要安装Docker并登录[Ascend Hub中心](https://ascendhub.huawei.com/#/detail?name=ascend-tensorflow-arm)获取镜像。

### 数据集准备
1. 模型训练使用cifar-10数据集，请用户自行获取。


### 模型训练

- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练    
    - 1. 启动训练之前，首先要配置程序运行相关环境变量。
       环境变量配置信息参见：
          [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)

       
## 迁移学习指导

- 数据集准备。
    1.  获取数据。请参见“快速上手”中的数据集准备。
    2.  数据目录结构如下：    
        ```
        |--cifar-10
        │   |--airplane
        │   |--automobile 
        │   |--bird 
        │   |--cat 
        │   |--deer 
        │   |--dog 
        │   |--frog 
        │   |--horse 
        │   |--ship 
        │   |--truck 
        ```
-   模型训练。
    1. 配置训练参数 首先在脚本run_1p.sh中，配置训练数据集--data、选择网络--network、输入训练数据量--labels参数。

    2. 启动训练 运行boot_modelarts.py

-   模型评估。
    输出分类错误率

## 高级参考
- 脚本和示例代码
```
.
├── LICENSE
├── README
├── SESEMI_tf_zxy295689322
│   |--boot_modelarts.py
│   |--run_1p.sh
│   |--train_evaluate_asl2.py
│   |--utils.py
│   |--help_modelarts.py
│   |--tttttttttttt.py
│   |--yyyy.py
```




### 训练过程<a name="section1589455252218"></a >
1. 通过“模型训练”中的训练指令启动训练。 


### 推理/验证过程<a name="section1465595372416"></a >
1. 训练和测试结束后，模型会保存成.h5文件。下面我们给出复现精度

### 训练精度

以下是各精度对比数据。

| 样本数量 | 论文精度   | GPU精度 | NPU精度 |
| -------- | ---------- | ------- | ------- |
| 1000     | 29.44±0.24 | 0.2876  | 0.2983  |
| 2000     | 21.53±0.18 | 0.2186  | 0.2179  |

在训练数据量为1000和2000的条件下，我们复现的GPU和NPU精度都能达到论文指标

### 训练性能

| 样本数量 | GPU性能   | NPU性能   |
| -------- | --------- | --------- |
| 1000     | 75ms/step | 53ms/step |

备注：使用x86机器本地复现。

总结：GPU下训练性能为75ms/step。NPU下训练性能为53ms/step。总体来看，NPU的训练性能强于GPU。

#### 数据集说明：
我们提供了CIFAR-10数据集，其他数据集需要修改其中的相对路径才可跑通，我们提供的数据如下：
URL:
https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=sD3UuWOA7+bXMQtWCv8y4+U8h1F32v1qT6N1/xkznQhDes8t6dinTzBhKrnPqNYqVDVNfbANARmzk2zF006mmER/6izbNTJLeIz0UlG9b6b0cOmeYkm/ftRvmAlE802cyHOesS57GvfBqT4goeQM4+wgNg/9x0Ccfp309tEq4PmoqxTBbeoM29Ocd3o2pMTnv4oWRYR8KKb/7D0TVhbOzBtZshz89Fmajg0ehho7uwHBW0HYyhdmry2FhGR/JsuutA2yLQ68NeBcyXLYCf0ARJidSH3T5gOHO/470zpRasnsNXE9m2T4RrLpObDOu9v5dz9cxSA/GmBtVqe6C0issrP/cLSyCgrtfCPKQgPpJo39mkBSDtQsetgD45uUMPodLz9k2G+qFnlpoZrU5YhB/Q1hfCSqlZhp3n5Fbu8tyGx6AJWZXOuiZuEeZxhMZAybD9oJcRmcoSqtZ1+QYyvRQmGw1a3+O+Fidt7CpxdjOJ9YnxgDA4s3PedC8EkXBMUFJLGsUDSd65HIH/d1K7Zb1E7Ti6v8cM8CDAqmI97VP11LupaVMasyJ16+TKpgx07cSxioUwH5HP72NwG2rVjeCJPYMYEKZ1oeNPpSMpSXoxft/E8XMPYPrW2DOJU47rklpr9kzYL+PsASc2M3jAsJgg==

提取码:
123456

*有效期至: 2022/11/08 09:28:47 GMT+08:00
#### 训练日志
\训练日志
