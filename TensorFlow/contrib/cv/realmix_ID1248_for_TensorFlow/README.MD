<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：** CV 

**修改时间（Modified） ：2020.10.30**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Research**

**描述（Description）：    半监督学习模型用于cifar10数据集分类** 

<h2 id="概述.md">概述</h2>

Realmix是一个半监督学习的模型，以较少的标记数据达到非常高的准确率，而且当未标记和标记的数据分布不一致时不匹配时，能够超过基线性能。

- 参考论文
[Towards Realistic Semi-Supervised Deep Learning Algorithms](https://arxiv.org/abs/1912.08766)

- 参考实现
<https://github.com/uizard-technologies/realmix>
    

- 适配昇腾 AI 处理器的实现：
    
        
  https://gitee.com/zhuzhennan7/modelzoo/edit/master/contrib/TensorFlow/Research/cv/realmix
        



## 默认配置<a name="section91661242121611"></a>

- 训练数据集预处理（以CIFAR10训练集为例，仅作为用户参考示例）：

  - 图像的输入尺寸为32*32
  - 图像输入格式：TFRecord
 
- 测试数据集预处理（以ImageNet2012验证集为例，仅作为用户参考示例）

  - 图像的输入尺寸为32*32
  - 图像输入格式：TFRecord
  - 根据CIFAR10数据集通用的平均值和标准偏差对输入图像进行归一化

- 训练超参

  - Weight decay(wd): 0.02
  - ema: 0.999
  - beta: 0.75
  - scales: 0
  - filters: 32
  - repeat: 4
  - Learning rate(LR): 0.002
  - Optimizer: AdamOptimizer
  - batch: 64
  - epoch: 700


## 支持特性<a name="section1899153513554"></a>

| 特性列表  | 是否支持 |
|-------|------|
| 混合精度  | 是    |

## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。


<h2 id="训练环境准备.md">训练环境准备</h2>

1.  硬件环境准备请参见各硬件产品文档"[驱动和固件安装升级指南]( https://support.huawei.com/enterprise/zh/category/ai-computing-platform-pid-1557196528909)"。需要在硬件设备上安装与CANN版本配套的固件与驱动。
2.  宿主机上需要安装Docker并登录[Ascend Hub中心](https://ascendhub.huawei.com/#/detail?name=ascend-tensorflow-arm)获取镜像。
3.  模型需要的包在requirements.txt里面

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
1. 模型训练使用cifar10数据集，数据集请用户自行获取，原始数据集需要运行predata.py进行数据预处理。桶obs://realmixdata/dataset中已经有处理好的数据集，直接使用即可
   
## 模型训练以及评估精度<a name="section715881518135"></a>
如果下载数据集后更改了桶的位置，请修改read_fromobs.py里面的信息。或者直接运行run_sh.py，脚本会自动打屏精度等信息

## 代码文件夹结构<a name="section08421615141513"></a>

```
│  realmix.py          模型代码
│  requirements.txt            训练依赖说明
│  run_sh.py    模型运行代码
│  read_fromobs.py   将数据从obs读到内存
│  initialization.sh  模型初始化脚本
│ result.txt   评估执行结果
├─libml   模型训练的相关代码
├─scripts  模型的数据处理部分
├─scripts  模型的数据处理部分
```

## 脚本参数<a name="section6669162441511"></a>

```
-- Weight decay(wd): 0.02
-- ema: 0.999
-- beta: 0.75
-- scales: 0
-- filters: 32
-- repeat: 4
-- Learning rate(LR): 0.002
-- Optimizer: AdamOptimizer
-- batch: 64
-- epoch: 700
```


## 训练过程<a name="section1589455252218"></a>

GPU参考精度，在750个epoch后ema所对应测试集精度达到论文最低要求。
```

                  labeled    valid     test
             raw   100.00    87.00    86.05
             ema   100.00    88.20    89.59
       tuned_raw   100.00    87.00    86.52
       tuned_ema   100.00    88.40    89.50
```
NPU训练精度，在700个epoch后达到论文要求，ckpt文件储存在obs://realmixdata/ckpt/中，日志文件储存在obs://realmixdata/dataset/log/MA-new-realmix-master_npu_for_TensorFlow-10-28-17-12/中
```
                    labeled    valid     test
             raw   100.00    81.20    80.35
             ema   100.00    88.40    89.87
       tuned_raw   100.00    84.40    83.41
       tuned_ema   100.00    88.40    89.60
```
性能,设置日志级别```ASCEND_GLOBAL_LOG_LEVEL=4 export ASCEND_GLOBAL_EVENT_ENABLE=0```后，GPU和NPU的训练速度对比如下

GPU
```
Epoch 7/1024:  15%|█▌        | 9856/65536 [00:13<01:20, 687.43img/s]
Epoch 7/1024:  15%|█▌        | 9984/65536 [00:13<01:17, 713.83img/s]
Epoch 7/1024:  15%|█▌        | 10112/65536 [00:13<01:14, 739.51img/s]
Epoch 7/1024:  16%|█▌        | 10240/65536 [00:13<01:19, 692.11img/s]
Epoch 7/1024:  16%|█▌        | 10368/65536 [00:13<01:17, 716.42img/s]
Epoch 7/1024:  16%|█▌        | 10496/65536 [00:14<01:14, 739.26img/s]
Epoch 7/1024:  16%|█▌        | 10624/65536 [00:14<01:14, 739.75img/s]
Epoch 7/1024:  16%|█▋        | 10752/65536 [00:14<01:12, 751.46img/s]
Epoch 7/1024:  17%|█▋        | 10880/65536 [00:14<01:11, 763.25img/s]
Epoch 7/1024:  17%|█▋        | 11008/65536 [00:14<01:10, 772.94img/s]
Epoch 7/1024:  17%|█▋        | 11136/65536 [00:14<01:10, 770.57img/s]
Epoch 7/1024:  17%|█▋        | 11264/65536 [00:15<01:10, 766.49img/s]
Epoch 7/1024:  17%|█▋        | 11392/65536 [00:15<01:09, 778.50img/s]
Epoch 7/1024:  18%|█▊        | 11520/65536 [00:15<01:16, 704.58img/s]
Epoch 7/1024:  18%|█▊        | 11648/65536 [00:15<01:15, 716.97img/s]
Epoch 7/1024:  18%|█▊        | 11776/65536 [00:15<01:13, 731.68img/s]
Epoch 7/1024:  18%|█▊        | 11904/65536 [00:15<01:12, 736.00img/s]
Epoch 7/1024:  18%|█▊        | 12032/65536 [00:16<01:11, 746.00img/s]
Epoch 7/1024:  19%|█▊        | 12160/65536 [00:16<01:11, 746.47img/s]
Epoch 7/1024:  19%|█▉        | 12288/65536 [00:16<01:12, 731.21img/s]
```
NPU
```
Epoch 7/1024:  15%|█▍        | 9536/65536 [00:28<00:37, 1484.07img/s]
Epoch 7/1024:  15%|█▍        | 9728/65536 [00:28<00:37, 1484.33img/s]
Epoch 7/1024:  15%|█▌        | 9920/65536 [00:28<00:37, 1484.25img/s]
Epoch 7/1024:  15%|█▌        | 10112/65536 [00:28<00:37, 1484.87img/s]
Epoch 7/1024:  16%|█▌        | 10304/65536 [00:29<00:37, 1486.61img/s]
Epoch 7/1024:  16%|█▌        | 10496/65536 [00:29<00:37, 1486.80img/s]
Epoch 7/1024:  16%|█▋        | 10688/65536 [00:29<00:36, 1487.55img/s]
Epoch 7/1024:  17%|█▋        | 10880/65536 [00:29<00:36, 1487.28img/s]
Epoch 7/1024:  17%|█▋        | 11072/65536 [00:29<00:36, 1486.61img/s]
Epoch 7/1024:  17%|█▋        | 11264/65536 [00:29<00:36, 1487.95img/s]
Epoch 7/1024:  17%|█▋        | 11456/65536 [00:29<00:36, 1487.04img/s]
Epoch 7/1024:  18%|█▊        | 11648/65536 [00:29<00:36, 1487.04img/s]
Epoch 7/1024:  18%|█▊        | 11840/65536 [00:30<00:36, 1488.57img/s]
Epoch 7/1024:  18%|█▊        | 12032/65536 [00:30<00:35, 1489.90img/s]
Epoch 7/1024:  19%|█▊        | 12224/65536 [00:30<00:35, 1490.96img/s]
Epoch 7/1024:  19%|█▉        | 12416/65536 [00:30<00:35, 1491.38img/s]
Epoch 7/1024:  19%|█▉        | 12608/65536 [00:30<00:35, 1486.73img/s]
Epoch 7/1024:  20%|█▉        | 12800/65536 [00:30<00:35, 1486.70img/s]
Epoch 7/1024:  20%|█▉        | 12992/65536 [00:30<00:35, 1487.11img/s]
Epoch 7/1024:  20%|██        | 13184/65536 [00:31<00:35, 1489.90img/s]

```
可以看到NPU的速度接近GPU的两倍