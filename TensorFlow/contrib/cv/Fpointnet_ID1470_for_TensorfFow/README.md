-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [快速上手](#快速上手.md)
-   [迁移学习指导](#迁移学习指导.md)
-   [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：** 3D Object Detection 

**版本（Version）：1.2**

**修改时间（Modified） ：2021.11.19**

**大小（Size）：124M**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Research**

**描述（Description）：基于TensorFlow框架的F-PointNet3D目标检测网络训练代码** 

<h2 id="概述.md">概述</h2>

F-PointNet是一种研究了基于RGB-D数据的三维目标检测网络。提出了一种新的检测管道，结合成熟的二维物体探测器和最先进的3D深度学习技术。首先使用运行在RGB图像上的2D检测器构建对象，其中每个2D边界框定义一个3D截锥区域，然后，基于这些截锥区域的三维点云，使用PointNet/PointNet++网络实现三维实例分割和模态三维边界盒估计。利用二维目标检测器，大大减少了三维目标定位的搜索空间，图像的高分辨率和丰富的纹理信息也使得像行人或自行车这样的小物体的高召回率，这些小物体很难仅靠点云来定位。通过采用PointNet架构，能够直接在3D点云上工作，而不需要将它们体素化到网格或将它们投影到图像平面。因为直接在点云上工作，能够充分尊重和利用三维几何，例如我们应用的一系列坐标归一化，这有助于定位学习问题。在KITTI和SUNRGBD的基准上进行评估，该网络系统显著优于先前的技术状态，在当前的KITTI排行榜上仍然处于领先地位。

- 参考论文：

    [ Charles R. Qi, Wei Liu, Chenxia Wu, Hao Su and Leonidas J. Guibas from Stanford University and Nuro Inc. “Frustum PointNets for 3D Object Detection from RGB-D Data” arXiv:1711.08488](https://arxiv.org/pdf/1711.08488.pdf) 

- 参考实现：

    

- 适配昇腾 AI 处理器的实现：
    
  https://gitee.com/ascend/modelzoo/tree/master/contrib/TensorFlow/Research/cv/Fpointnet_ID1263_for_TensorfFow
        


- 通过Git获取对应commit\_id的代码方法如下：
    
    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    ```

## 默认配置<a name="section91661242121611"></a>

- 训练数据集预处理（以KITTI数据为例，仅作为用户参考示例）：
我们将原始KITTI数据转换为有组织的格式，用于训练我们的Frustum PointNets。
首先，您需要下载KITTI 3D目标检测数据集，包括左彩色图像、Velodyne点云、相机校准矩阵和训练标签。确保KITTI数据按照dataset/README.md中的要求组织。您可以运行kitti/kitti_object.py来查看数据是否下载和存储正确。如果一切正常，您应该可以看到数据的图像和3D点云可视化。然后准备数据，只需运行:(警告:这一步将产生约4.7 gb数据作为pickle文件)sh scripts/command_prep_data.sh
在这个过程中,我们从原始KITTI标签数据中提取视锥点云数据与标签数据。我们将使用2D框图标签 (kitti/ rgb_detection_val.txt)提取训练集(kitti/image_sets/train.txt)和验证集(kitti/image_sets/val.txt)，也从验证集中提取数据与预测的2D 框图 (kitti/ rgb_detection_val.txt)。你可以查看kitti/prepare_data.py了解更多细节，并运行kitti/prepare_data.py——demo来可视化数据准备中的步骤。命令执行后，您应该在kitti文件夹下看到三个新生成的数据文件。你可以运行train/provider.py来可视化训练数据。


- 训练超参

  - Batch size: 32
  - Momentum: 0.9
  - Learning rate(lr): 0.001
  - Optimizer: Adam
  - num_point:1024
  - decay_rate: 0.5
  - decay_step:800000
  - batch_size: 201


## 支持特性<a name="section1899153513554"></a>

| 特性列表  | 是否支持 |
|-------|------|
| 混合精度  | 是    |

## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

脚本已默认开启混合精度，设置参数参考如下。

  ```
    elif chip == 'npu':
        custom_op.parameter_map["use_off_line"].b = True
        custom_op.parameter_map["dynamic_input"].b = True
        custom_op.parameter_map["mix_compile_mode"].b = True
        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
        custom_op.parameter_map["dynamic_graph_execute_mode"].s = tf.compat.as_bytes("lazy_recompile")
        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF

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
    <td class="cellrowborder" valign="top" width="25.52%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001074498056_p1450714271532"><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><em id="i72359495125"><a name="i72359495125"></a><a name="i72359495125"></a>21.0.2</em></p>
    </td>
    <td class="cellrowborder" valign="top" width="27.16%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001074498056_p18244640152312"><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><em id="i162363492129"><a name="i162363492129"></a><a name="i162363492129"></a><a href="https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software" target="_blank" rel="noopener noreferrer">21.0.2</a></em></p>
    </td>
    </tr>
    </tbody>
    </table>


<h2 id="快速上手.md">快速上手</h2>

- 数据集准备

1. 模型训练使用[KITTI数据集](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)，数据集请用户自行获取。

2. 数据集处理后，放入模型目录下，在训练脚本中指定数据集路径，可正常使用。
   

## 模型训练<a name="section715881518135"></a>

- 单击[“立即下载”](https://github.com/charlesq34/frustum-pointnets)，并选择合适的下载方式下载源码包。

- 启动训练之前，首先要配置程序运行相关环境变量。

  环境变量配置信息参见：

     [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)

- 单卡训练 

  1. 配置训练参数。

     训练数据集放在obs桶中，配置训练数据集路径，请用户根据实际路径配置，数据集参数如下所示：

     ```
      --data_dir=fpoint/KITTI/object1
     ```

  2. 启动训练。

     启动单卡训练 （脚本为 Fpointnet_ID1263_for_TensorfFow / scripts / command_train_v1.sh） 

     ```
     command_train_v1.sh
     ```

<h2 id="迁移学习指导.md">迁移学习指导</h2>

- 数据集准备。

  数据集要求如下：

  1. 获取数据。

     如果要使用自己的数据集，需要将数据集放到自己的obs桶中。参考代码中的数据集存放路径如下：

     - 训练集： fpoint/KITTI/object1
     - 测试集： fpoint/KITTI/object1

     训练数据集和测试数据集以文件名中的train和test加以区分。

     数据集也可以放在其它目录，则修改对应的脚本入参data_dir即可。

  2. 准确标注类别标签的数据集。

  3. 数据集每个类别所占比例大致相同。


<h2 id="高级参考.md">高级参考</h2>


## 训练过程<a name="section1589455252218"></a>

1.  通过“模型训练”中的训练指令启动单卡训练。
```
## NPU/GPU 网络训练精度box estimation accuracy (IoU=0.7) 
| NPU  | GPU |
|-------|------|
| 0.59| 0.61 |
```
```
## NPU/GPU 网络训练性能 
| NPU  | GPU |
|-------|------|
| 0.088s/step| 0.082s/step|
```
其中GPU为v100
```