-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [快速上手](#快速上手.md)
-   [迁移学习指导](#迁移学习指导.md)
-   [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）： Image Registration**

**修改时间（Modified） ：2022.05.13**

**大小（Size）：8.71M**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的voxelmorph图像配准网络训练代码** 

<h2 id="概述.md">概述</h2>

voxelmorph是一种基于快速学习的可变形、成对的三维医学图像配准算法。该方法将配准定义为一个参数函数，并在给定一组感兴趣的图像的情况下优化其参数。给定一对新的图像对（待配准图像，参考图像），voxelmorph可以通过使用学习的参数直接计算函数来快速计算配准场，使用CNN对该配准函数进行建模，并使用空间变换层将待配准图像配准到参考图像，同时对配准场施加平滑度约束。该方法不需要有监督的信息，如地面真实度配准场或解剖地标。

- 参考论文：

    [VoxelMorph: A Learning Framework for Deformable Medical Image Registration](http://arxiv.org/abs/1809.05231)

- 参考实现：

    [voxelmorph/voxelmorph at legacy (github.com)](https://github.com/voxelmorph/voxelmorph/tree/legacy)

- 适配昇腾 AI 处理器的实现：
  
  [TensorFlow/contrib/cv/voxelmorph_ID2120_for_TensorFlow · Ascend/ModelZoo-TensorFlow - 码云 - 开源中国 (gitee.com)](https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/voxelmorph_ID2120_for_TensorFlow) 


- 通过Git获取对应commit\_id的代码方法如下：
  
    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    ```

## 默认配置<a name="section91661242121611"></a>

- 训练数据集预处理（以ABIDE freesurfer pipeline数据集为例，仅作为用户参考示例）：

  - 选取一个参考图像
  - 使用freesurfer将其它图像线性配准到参考图像，并进行提取配准场
  - 使用freesurfer应用配准场到其它图像的分割图像
- 测试数据集预处理（以ABIDE freesurfer pipeline数据集为例，仅作为用户参考示例）

  - 选取一个参考图像
  - 使用freesurfer将其它图像线性配准到参考图像，并进行提取配准场
  - 使用freesurfer应用配准场到其它图像的分割图像
- 训练超参

  - `--lr`：1e-4
  - `--epochs`：50
  - `--lambda`：0.01
  - `--batch_size`：1


## 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 否       |
| 混合精度   | 是       |
| 并行数据   | 是       |

## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

脚本默认不开启混合精度。


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
1. 模型训练使用ABIDE freesurfer pipeline数据集，数据集请用户自行获取。

2. 数据集训练前需要做预处理操作，请用户参考上文默认配置。

3. 数据集处理后，放入模型目录下，在训练脚本中指定数据集路径，可正常使用。
   

## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。

- 启动训练之前，首先要配置程序运行相关环境变量。

  环境变量配置信息参见：

     [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

- 单卡训练 

  1. 配置训练参数。

     首先在脚本src/train_1p.sh中，配置训练数据集路径，请用户根据实际路径配置，数据集参数如下所示：

     ```
     ${训练数据集地址} --atlas_file ${参考（配准模板）数据地址}
     ```

  2. 启动训练。

     启动单卡训练 （脚本为voxelmorph_ID2120_for_TensorFlow/src/run_1p_all.sh） 

     ```
     bash run_1p_all.sh
     ```


- 1. 测试的时候，需要修改脚本中的参数（脚本位于voxelmorph_ID2120_for_TensorFlow/src/test.sh），参数如下所示。

          ${测试数据集地址} --model_path ${ckpt权重地址}
      
    2. 测试指令（脚本位于voxelmorph_ID2120_for_TensorFlow/src/test.sh）
  
      bash test.sh

<h2 id="迁移学习指导.md">迁移学习指导</h2>

- 数据集准备。

  数据集要求如下：

  1. 获取数据。
2. 数据集可以放在其它目录，则修改对应的脚本入参data_dir即可。


- 加载预训练模型。 
    
    修改文件train_all.py，修改以下参数。
    
    
    ```
    --load_model_file：${要加载的ckpt权重地址}
    ```

-   模型训练。

    参考“模型训练”中训练步骤。

-   模型评估。
    
    参考“模型训练”中验证步骤。

<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码<a name="section08421615141513"></a>

```
├── ext                                  	//项目需要的外部库
│    ├── medipy-lib
│    ├── neuron
│    ├── pynd-lib
│    ├── pytools-lib
├── models                                 	//预训练权重
├── precision_tool2							//华为官方的精度工具，修改了fusion_switch.cfg，关闭了UB融合
├── src										//项目文件
│    ├── datagenerators.py            		//数据
│    ├── losses.py                       	//定义loss
│    ├── networks.py                   		//定义网络
│    ├── test_zyh.py                    	//测试代码
│    ├── train_all.py                		//训练代码
│    ├── run_1p_all.sh  					//训练入口
│    ├── test.sh  							//测试入口
│    ├── loss+perf_npu_all.txt				//打印日志
```

## 训练过程<a name="section1589455252218"></a>

1.  通过“模型训练”中的训练指令启动单卡训练。

2.  参考脚本的模型存储路径为`../models/`，训练脚本log中包括如下信息。

```
INFO:tensorflow:**********
2022-03-28 11:23:39.720381: W tensorflow/core/platform/profile_utils/cpu_utils.cc:98] Failed to find bogomips in /proc/cpuinfo; cannot determine CPU frequency
2022-03-28 11:23:39.730495: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x390ef190 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2022-03-28 11:23:39.730614: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2022-03-28 11:23:40.072392: W /home/jenkins/agent/workspace/Compile_GraphEngine_Centos_ARM/tensorflow/tf_adapter/util/ge_plugin.cc:124] [GePlugin] can not find Environment variable : JOB_ID
[PrecisionTool] Set Tensorflow random seed to 0 success.
[PrecisionTool] Set numpy random seed to 0 success.
[PrecisionTool] Set fusion switch file:  ../precision_tool2/fusion_switch.cfg
2022-03-28 11:23:49.988205: I /home/jenkins/agent/workspace/Compile_GraphEngine_Centos_ARM/tensorflow/tf_adapter/kernels/geop_npu.cc:739] The model has been compiled on the Ascend AI processor, current graph id is:1
INFO:tensorflow:*********　　epoch 0　　　***********
2022-03-28 11:24:14.073463: I /home/jenkins/agent/workspace/Compile_GraphEngine_Centos_ARM/tensorflow/tf_adapter/kernels/geop_npu.cc:739] The model has been compiled on the Ascend AI processor, current graph id is:11
INFO:tensorflow:step {0} --->  loss: {0.00995}, loss_mse: {0.00995}, loss_flow: {1.098e-10}
...
...
...

```

## 训练精度

|                                          | NPU          | GPU          | 原论文       |
| ---------------------------------------- | ------------ | ------------ | ------------ |
| DICE系数（[0, 1], 1 最优）/ 均值(标准差) | 0.703(0.134) | 0.708(0.133) | 0.752(0.140) |

NPU能够达到GPU训练精度

## 训练性能

|                    | NPU   | GPU   |
| ------------------ | ----- | ----- |
| 时间（1个step）/ s | 1.811 | 0.830 |

```
NPU

INFO:tensorflow:step {6} --->  loss: {0.00579}, loss_mse: {0.00542}, loss_flow: {3.773e-02}, time: {1.811}

GPU

INFO:tensorflow:step {7} --->  loss: {0.00996}, loss_mse: {0.00996}, loss_flow: {9.714e-05}, time: {0.830}

```

NPU与GPU性能相差大概2倍