-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [快速上手](#快速上手.md)
-   [迁移学习指导](#迁移学习指导.md)
-   [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）： 其他（3D Classification and Segmentation）**

**版本（Version）：1.1**

**修改时间（Modified） ：2021.10.09**

**大小（Size）：0M**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的PointNet点云网络训练代码** 

<h2 id="概述.md">概述</h2>

PointNet是一种针对点云提出的深度网络架构。这种网络无需将点云转化为3D体素网格或图像集合，而是直接将点云数据作为输入数据。PointNet为从物体分类、零件分割到场景语义分析的应用程序提供了统一的架构。

- 参考论文：

    [Charles R. Qi, Hao Su, Kaichun Mo, Leonidas J. Guibas. “PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation.” arXiv:1608.06993](https://https://arxiv.org/pdf/1612.00593.pdf) 

- 参考实现（官方开源代码）：
  
  https://github.com/charlesq34/pointnet

    

- 适配昇腾 AI 处理器的实现：
  
  
  https://gitee.com/ascend/modelzoo/tree/master/contrib/TensorFlow/Research/cv/pointnet/PointNet_ID0266_for_TensorFlow
        


- 通过Git获取对应commit\_id的代码方法如下：
  
    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    ```

## 默认配置<a name="section91661242121611"></a>

- 训练数据集预处理（以ShapeNet数据集为例，仅作为用户参考示例）：

  - 点云数为2048
  - 图像输入格式：HDF5

- 测试数据集预处理（以ShapeNet数据集为例，仅作为用户参考示例）：

  - 点云数为3000
  - 

- 训练超参

  - Batch size: 32
  - Momentum: 0.9
  - point_num: 2048
  - BASE_LEARNING_RATE: 0.001
  - DECAY_STEP: 16881 * 20
  - DECAY_RATE: 0.5
  - LEARNING_RATE_CLIP = 1e-5
  - BN_INIT_DECAY: 0.5
  - BN_DECAY_DECAY_RATE: 0.5
  - BN_DECAY_DECAY_STEP: float(DECAY_STEP * 2)
  - BN_DECAY_CLIP = 0.99
  - Weight decay: 0.0
  - Train epoch: 200
  - Optimizer: AdamOptimizer


## 支持特性<a name="section1899153513554"></a>

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 否    |
| 混合精度  | 是    |
| 并行数据  | 否    |

## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

相关代码示例

  ```
  config = tf.ConfigProto()
  custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
  custom_op.name = "NpuOptimizer"
  custom_op.parameter_map["use_off_line"].b = True
  # mix precision
  if FLAGS.mix_precision == 'on':
      custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
  # mix precision end
  config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
  config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
  config.allow_soft_placement = True
  sess = tf.Session(config=config)
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

1. 零件分割任务中，模型训练使用ShapeNetPart数据集，ShapeNetPart数据集包含16个点云模型，文件夹train_test_split中为训练集，测试集和验证集划分文件。

2. 数据集训练的格式为HDF5，请用户在官网自行获取。（如果需要使用用户自己的数据集，可以使用utils/data_prep_util.py中的辅助函数来保存和加载HDF5文件）

3. 数据集下载后，放入相应目录下，在训练脚本中指定数据集路径，可正常使用。
   

## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。

- 启动训练之前，首先要配置程序运行相关环境变量。

  环境变量配置信息参见：

     [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)

- 单卡训练 

  1. 配置训练参数

     首先在脚本scripts/train_1p.sh中，配置训练数据集路，请用户根据实际路径配置，数据集参数如下所示：

     ```
      NA
     ```

  2. 启动训练

     启动单卡训练 （脚本为PointNet_ID0266_for_TensorFlow/scripts/run_1p.sh） 

     ```
     bash run_1p.sh
     ```

- 验证

    1. 测试的时候，需要修改脚本启动参数（脚本位于PointNet_ID0266_for_TensorFlow/scripts/test.sh）。

          ```
          NA
          ```

  2. 测试指令（脚本位于PointNet_ID0266_for_TensorFlow/scripts/test.sh）

      ```
      bash test.sh
      ```

- NPU与GPU训练精度对比

  - 在ShapeNet数据集上，计算mIoU作为零件分割任务的评价指标评测模型精度。
  - 选择第190个epoch训练模型进行测试
      ```
      NPU上的精度(IoU)结果: 83.66
      NPU上开启混合精度后的精度(IoU)结果: 83.81
      GPU上的精度(IoU)结果: 83.74
      ```
      [精度数据(提取码：7777)](https://pan.baidu.com/s/1aOcJkkVzLcobvvVLko_D-Q)

## 模型推理<a name="section715881518135"></a>
- 模型固化
    ```
    python3 frozen_graph.py --phase frozen_graph
    ```
- 在线推理
    ```
    python3 frozen_graph.py --phase test_pb
    ```

<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码<a name="section08421615141513"></a>

```
├── evaluate.py                               //物体分类网络测试代码
├── LICENSE                                   //license文件
├── provider.py                               //数据处理相关代码
├── README.md                                 //代码说明文档
├── train.py                                  //物体分类网络训练代码
├── models
│    ├── pointnet_cls_basic.py               //物体分类网络构建相关代码
│    ├── pointnet_cls.py                     //物体分类网络模型
│    ├── pointnet_seg.py                     //零件分割网络模型
│    ├── transform_nets.py                   //网络构建相关代码
├── part_seg
│    ├── frozen_graph.py                     //模型固化以及在线推理代码
│    ├── pointnet_part_seg.py                //零件分割网络模型
│    ├── test.py                             //零件分割网络测试代码
│    ├── testing_ply_file_list.txt           //测试辅助代码
│    ├── train.py                            //零件分割网络训练代码
├── scripts
│    ├── download_data.sh                    //零件分割任务数据集下载脚本
│    ├── test.sh                             //零件分割任务测试脚本
│    ├── train_1p.sh                         //单卡训练脚本
├── sem_seg
│    ├── meta
│    │    ├── all_data_label.txt            //数据集相关代码
│    │    ├── anno_paths.txt                //数据集相关代码
│    │    ├── area6_data_label.txt          //数据集相关代码
│    │    ├── class_names.txt               //数据集相关代码
│    ├── batch_inference.py                  //室内场景语义分析测试代码
│    ├── collect_indoor3d_data.py            //室内场景语义分析数据预处理相关代码
│    ├── eval_iou_accuracy.py                //室内场景语义分析网络精度评估
│    ├── gen_indoor3d_h5.py                  //HDF5数据文件生成代码
│    ├── indoor3d_util.py                    //数据处理辅助函数代码
│    ├── model.py                            //室内场景语义分析网络模型
│    ├── train.py                            //室内场景语义分析训练代码
├── utils
│    ├── data_prep_util.py                   //数据集相关辅助函数
│    ├── eulerangles.py                      //数据坐标处理相关函数
│    ├── pc_util.py                          //点云数据处理相关函数
│    ├── plyfile.py                          //数据文件处理相关函数
│    ├── tf_util.py                          //模型构建辅助函数
```

## 脚本参数<a name="section6669162441511"></a>

```
NA
```

## 训练过程<a name="section1589455252218"></a>

NA

## 推理/验证过程<a name="section1465595372416"></a>

NA
