-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [快速上手](#快速上手.md)
-   [迁移学习指导](#迁移学习指导.md)
-   [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：** Image Classification 

**版本（Version）：1.1**

**修改时间（Modified） ：2021.11.27**

**大小（Size）：74M**

**框架（Framework）：TensorFlow 2.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Benchmark**

**描述（Description）：基于TensorFlow框架的BYOL自监督学习训练代码** 

<h2 id="概述.md">概述</h2>

在BYOL之前，多数自我监督学习都可分为对比学习或生成学习，其中，生成学习一般GAN建模完整的数据分布，计算成本较高，相比之下，对比学习方法就很少面临这样的问题。BYOL的目标与对比学习相似，但一个很大的区别是，BYOL不关心不同样本是否具有不同的表征（即对比学习中的对比部分），仅仅使相似的样品表征类似。看上去似乎无关紧要，但这样的设定会显著改善模型训练效率和泛化能力：

1. 由于不需要负采样，BLOY有更高的训练效率。在训练中，每次遍历只需对每个样本采样一次，而无需关注负样本。
2. BLOY模型对训练数据的系统偏差不敏感，这意味着模型可以对未见样本也有较好的适用性。

BYOL最小化样本表征和该样本变换之后的表征间的距离。其中，不同变换类型包括0：平移、旋转、模糊、颜色反转、颜色抖动、高斯噪声等（在此以图像操作来举例说明，但BYOL也可以处理其他数据类型）。

- 参考论文：

    [Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning"](https://arxiv.org/pdf/2006.07733.pdf) 

- 参考实现：

  [Byol](https://github.com/garder14/byol-tensorflow2) 

- 适配昇腾 AI 处理器的实现：
  
  
  todo：添加该仓库的网页链接
        


- 通过Git获取对应commit\_id的代码方法如下：
  
    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    ```

## 默认配置<a name="section91661242121611"></a>

- 训练超参

  - Batch size: 512
  - Momentum: 0.9
  - LR scheduler: cosine
  - Learning rate(LR): 0.001
  - Optimizer: AdamOptimizer
  - Weight decay: 0.0001
  - Label smoothing: 0.1
  - Train epoch: 200


## 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 否       |
| 混合精度   | 是       |
| 并行数据   | 否       |

## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

脚本已默认开启混合精度，设置precision_mode参数的脚本参考如下。

  ```
  run_config = NPURunConfig(        
  		model_dir=flags_obj.model_dir,        
  		session_config=session_config,        
  		keep_checkpoint_max=5,        
  		save_checkpoints_steps=5000,        
  		enable_data_pre_proc=True,        
  		iterations_per_loop=iterations_per_loop,        			
  		log_step_count_steps=iterations_per_loop,        
  		precision_mode='allow_mix_precision',        
  		hcom_parallel=True      
        )
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

- 数据集准备
1. 模型训练使用Cifar10数据集，数据集请用户自行获取。


## 模型训练<a name="section715881518135"></a>

- 启动训练之前，首先要配置程序运行相关环境变量。

  环境变量配置信息参见：

     [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)

- 快速demo

  你可以通过训练1个epoch来验证代码的正确性。训练可以执行命令：

  ```
  python pretraining.py --encoder resnet18 --num_epochs 1 --batch_size 512
  ```

  要预训练 ResNet-18作为基础编码器 200 次（每 100 次保存权重），可以执行命令： 

  ```
  python pretraining.py --encoder resnet18 --num_epochs 200 --batch_size 512
  ```


<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码<a name="section08421615141513"></a>

```
├── README.md                                 //代码说明文档
├──models.py                                  //网络构建
├──datasets.py                                //数据加载
├──augmentation.py                            //数据增强
├──losses.py                                  //设置损失函数
├──pretrainning.py                            //网络训练
```

## 训练过程<a name="section1589455252218"></a>

1.  通过“模型训练”中的训练指令启动训练。

2.  我们在NVIDIA V100芯片上进行训练的部分日志如下：

```
[Epoch 98/100 Batch 80/97] Loss=0.34343.
[Epoch 98/100 Batch 90/97] Loss=0.37767.
[Epoch 99/100 Batch 10/97] Loss=0.33391.
[Epoch 99/100 Batch 20/97] Loss=0.33320.
[Epoch 99/100 Batch 30/97] Loss=0.35399.
[Epoch 99/100 Batch 40/971 Loss=0.35694.
[Epoch 99/100 Batch 50/97] Loss=0.36032.
[Epoch 99/100 Batch 60/97] Loss=0.35578
[Epoch 99/100 Batch 70/97] Loss=0.35547.
[Epoch 99/100 Batch 80/97] Loss=0.34882.
[Epoch 99/100 Batch 90/97] Loss=0.33286.
[Epoch 100/100 Batch 10/97] Loss=0.34306.
[Epoch 100/100 Batch 20/97] Loss=0.34042.
[Epoch 100/100 Batch 30/971 Loss=0.36179
[Epoch 100/100 Batch 40/97] Loss=0.35892.
[Epoch 100/100 Batch 50/97] Loss=0.34619.
[Epoch 100/100 Batch 60/97] Loss=0.33050.
[Epoch 100/100 Batch 70/971 Loss=0.32579.
[Epoch 100/100 Batch 80/97] Loss=0.34343.
[Epoch 100/100 Batch 90/97] Loss=0.37970. 
Weights of f saved.
```

​    3.我们在Ascend 910芯片上进行训练的部分日志如下： 

```
2021-11-27 11:29:30.466690: I core/npu_wrapper.cpp:154] Create device instance /job:localhost/replica:0/task:0/device:NPU:0 with extra options:
2021-11-27 11:29:30.469407: I core/npu_device_register.cpp:86] Npu device instance /job:localhost/replica:0/task:0/device:NPU:0 created
2021-11-27 11:29:30.476568: I tensorflow/compiler/jit/xla_cpu_device.cc:41] Not creating XLA devices, tf_xla_enable_xla_devices not set
2021-11-27 11:29:30.484576: I core/npu_wrapper.cpp:154] Create device instance /job:localhost/replica:0/task:0/device:NPU:0 with extra options:
2021-11-27 11:29:30.486762: I core/npu_device_register.cpp:86] Npu device instance /job:localhost/replica:0/task:0/device:NPU:0 created
Initializing online networks...
Shape of h: (256, 512)
Shape of z: (256, 128)
Shape of p: (256, 128)
Initializing target networks...
Shape of h: (256, 512)
Shape of z: (256, 128)
The encoders have 11173632 trainable parameters each.
Using Adam optimizer with learning rate 0.001.
2021-11-27 11:30:53.305650: I core/npu_device.cpp:1154] Graph __inference_train_step_pretraining_113222 can loop: false
[Epoch 1/200 Batch 10/97] Loss=1.00723.
[Epoch 1/200 Batch 20/97] Loss=0.89159.
[Epoch 1/200 Batch 30/97] Loss=0.82090.
```

