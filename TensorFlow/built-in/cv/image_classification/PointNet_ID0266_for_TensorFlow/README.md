-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [快速上手](#快速上手.md)
-   [迁移学习指导](#迁移学习指导.md)
-   [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：huawei**

**应用领域（Application Domain）：CV**

**版本（Version）：1.1**

**修改时间（Modified） ：2021.06.24**

**大小（Size）**_**：【深加工】**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：【深加工】**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：【深加工】**

**描述（Description）：基于TensorFlow框架的PointNet网络训练代码**

<h2 id="概述.md">概述</h2>

-    PointNet 基于点云数据，对三位形态进行分类以及语义分割的CV类网络。

-    -   参考论文：

        https://arxiv.org/pdf/1706.02413.pdf

    -   参考实现：
    
        ```
        https://github.com/charlesq34/pointnet
        ```
    
-   适配昇腾 AI 处理器的实现：【深加工】
    
        ```
        https://gitee.com/ascend/modelzoo/tree/master/built-in/TensorFlow/Research/cv/image_classification/PointNet_ID0266_for_TensorFlow
        branch=master
        commit_id= 477b07a1e95a35885b3a9a569b1c8ccb9ad5d7af
        ```


    -   通过Git获取对应commit\_id的代码方法如下：
    
        ```
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
        ```

## 默认配置【深加工】<a name="section91661242121611"></a>
-   网络结构
    -   初始学习率为0.0015，使用Exponential learning rate 衰减
    -   优化器：Adam
    -   单卡batchsize：32
    -   8卡batchsize：32*8
    -   总Epoch数设置为250
    -   Momentum: 0.9
    
-   训练数据集预处理（当前代码以ModelNet40为例）：
    -   每个图形对应的点云数量为2048
    -   网络能够适应不同的输入排列
    
-   测试数据集预处理（当前代码以ModelNet40测试集为例）：
    -   每个图形对应的点云数量为2048
    -   网络能够适应不同的输入排列

-   训练超参（单卡）：
    -   Batch size: 32
    -   Momentum: 0.9
    -   LR scheduler: exponential
    -   Learning rate\(LR\): 0.0015
    -   Train epoch: 250


## 支持特性【深加工】<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 是       |
| 混合精度   | 是       |
| 数据并行   | 是       |


## 混合精度训练【深加工】<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用，规避了MatMulv2算子的下降。

## 开启混合精度【深加工】<a name="section20779114113713"></a>
相关代码示例。



```
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = False
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name =  "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True 
custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
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

## 数据集准备<a name="section361114841316"></a>

1. 模型训练使用modelnet40_ply_hdf4_2048数据集，数据集请用户自行获取。

2. 放入模型目录下，在训练脚本中指定数据集路径，可正常使用。


## 模型训练【深加工】<a name="section715881518135"></a>
- 下载训练脚本。
- 检查scripts/目录下是否有存在8卡IP的json配置文件“8p.json”。
  
```
 {
"server_count":"1",
"server_list":[{
    "device":[{"device_id":"0","device_ip":"192.168.100.101","rank_id":"0"},
              {"device_id":"1","device_ip":"192.168.101.101","rank_id":"1"},
              {"device_id":"2","device_ip":"192.168.102.101","rank_id":"2"},
              {"device_id":"3","device_ip":"192.168.103.101","rank_id":"3"},
              {"device_id":"4","device_ip":"192.168.100.100","rank_id":"4"},
              {"device_id":"5","device_ip":"192.168.101.100","rank_id":"5"},
              {"device_id":"6","device_ip":"192.168.102.100","rank_id":"6"},
              {"device_id":"7","device_ip":"192.168.103.100","rank_id":"7"}],
    "server_id":"127.0.0.2"}],
"status":"completed",
"version":"1.0"
}
```

- 开始训练。
  
    1. 启动训练之前，首先要配置程序运行相关环境变量。

       环境变量配置信息参见：

          [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)
    

    2. 单卡训练
       
       
        单卡训练指令（脚本位于PointNet_ID0266_for_TensorFlow/test/train_perf_1p.sh）

```
        `bash train_perf_1p.sh`
```

3. 8卡训练【深加工】

    设置8卡训练参数（PointNet_ID0266_for_TensorFlow/test/train_perf_8p.sh），示例如下。

            ` bash train_perf_8p.sh`

<h2 id="开始测试.md">开始测试【深加工】</h2>

 - 参数配置
    1. 单卡训练指令（脚本位于PointNet_ID0266_for_TensorFlow/test/train_full_1p.sh）

        用例会在执行训练后，自动执行测试脚本 evaluate.py

    2. 测试生成的精度结果，存放于：
       
        `--test/output/**${**ASCEND_DEVICE_ID**}**/eval_**${**ASCEND_DEVICE_ID**}**.log


## 脚本和示例代码<a name="section08421615141513"></a>

    ├── README.md                                //说明文档
    ├── requirements.txt						 //依赖
    ├── data
    ├── config
    │    ├──8p.json 
    ├── utils
    │    ├──tf_util.py              
    │    ├──pc_util.py              
    │    ├──plyfile.py      
    │    ├──eulerangles.py
    │    ├──data_prep_util.py
    ├── models
    │    ├──pointnet_cls.py              
    │    ├──pointnet_cls_basic.py              
    │    ├──transform_nets.py      
    ├── train.py
    ├── train_8p_rev.py
    ├── evaluate.py
    ├──	provide.py
    ├──	LICENSE
    ├──	Modelzoolevel.txt
    ├── ops
    │    ├──ops_info.json                                     

## 脚本参数<a name="section6669162441511"></a>


```
--max_epoch                最大的训练epoch数， 默认：250
--model_name               网络模型
--moving_average_decay     滑动平均的衰减系数， 默认：None
--batch_size               每个NPU的batch size， 默认：32
--learning_rate_decay_type 学习率衰减的策略， 默认：expontonal
--learning_rate            学习率， 默认：0.0015
--optimizer                优化器， 默认：adam
--momentum                 动量， 默认：0.9 
--num_points               点云数量，默认：2048
```


## 训练/评估结果<a name="section1589455252218"></a>


| Acc     | FPS       | Npu_nums | Epochs   |
| :------: | :------:  | :------: | :------: |
| 89.12   |    212     |   1      |   250    |
| 87.97   |   1466     |   8      |    41    |