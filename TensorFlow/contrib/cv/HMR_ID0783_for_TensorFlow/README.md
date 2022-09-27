-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [快速上手](#快速上手.md)
-   [模型性能](#模型性能.md)
-   [高级参考](#高级参考.md)

<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：** Human body 3D reconstruction 

**版本（Version）：1.2**

**修改时间（Modified） ：2021.11.17**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt, pb, om**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910A, 昇腾310**

**应用级别（Categories）：Demo**

**描述（Description）：基于TensorFlow框架的HMR人体三维重建网络训练、推理代码** 

<h2 id="概述.md">概述</h2>

- HMR是一种人体三维重建算法，通过采用2D和3D形式的监督，并引入生成对抗网络修正输出结果的分布来实现从单张图像恢复人体的三维模型。

- 模型架构

  ![overview](overview.png)

- 参考论文：

    [Kanazawa A, Black M J, Jacobs D W, et al. End-to-end recovery of human shape and pose[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2018: 7122-7131.](https://openaccess.thecvf.com/content_cvpr_2018/papers/Kanazawa_End-to-End_Recovery_of_CVPR_2018_paper.pdf)

- [参考实现](https://github.com/akanazawa/hmr)

## 默认配置<a name="section91661242121611"></a>

- 训练超参
  - d_lr: 1e-4
  - e_lr: 1e-5
  - e_loss_weight: 60.
  - e_3d_weight: 60.
  - batch_size: 64

## 支持特性<a name="section1899153513554"></a>

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 否    |
| 混合精度  | 是    |
| 并行数据  | 否    |

## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

脚本已默认开启混合精度。

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

1. 数据集[下载](https://disk.pku.edu.cn:443/link/4213EF310253E1C75B96CEB3B6C89135)

2. 预训练模型[下载](https://disk.pku.edu.cn:443/link/7B61F0BED0FF405DA1A4E31856EE09A5)

3. 数据集和预训练模型下载后，在训练/测试脚本中指定相应路径，可正常使用。
   

## 模型训练<a name="section715881518135"></a>

- 选择合适的下载方式下载源码包。

- 启动训练之前，首先要配置程序运行相关环境变量。

  环境变量配置信息参见：

     [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)

- 根据requirements.txt配置所需依赖。

- 单卡训练 

  1. 在脚本`scripts/train_1p_ci.sh`中，配置所需路径和训练超参。

  2. 启动训练。 

     ```
     bash scripts/train_1p_ci.sh
     ```

- 测试

  1. 在脚本`scripts/test.sh`中，配置所需路径。

  2. 启动测试。

     ```
     bash scripts/test.sh
     ```

- Demo

  1. 在脚本`scripts/demo.sh`中，配置所需路径。

  2. 运行Demo。

     ```
     bash scripts/demo.sh
     ```
  
  3. 样例：
  
     输入图片：

     ![input](demo/example/input.jpg)

     输出图片：

     ![3D_Mesh_overlay](demo/example/3D_Mesh_overlay.jpg) ![3D_mesh](demo/example/3D_mesh.jpg) ![diff_vp1](demo/example/diff_vp1.jpg) ![diff_vp2](demo/example/diff_vp2.jpg) ![joint_projection](demo/example/joint_projection.jpg)

- Checkpoint

     训练好的Checkpoint文件可在[此处](https://disk.pku.edu.cn:443/link/215748269F58022E672DCAFD661DD251)下载。

- GPU版本

     GPU版本的代码和训练好的checkpoint可在[网盘](https://disk.pku.edu.cn:443/link/EE3D34ADCDCF8B479AE56216DF62F276)或[Gitee](https://gitee.com/huayuz/hmr)获取。

<h2 id="模型性能.md">模型性能</h2>

- 训练

| Parameters           | NPU                       | GPU                  |
| -------------------- | ------------------------- | -------------------- |
| Resource             | Ascend 910A               | Tesla V100-PCIE-16GB |
| Speed                | about 1.9 steps/s         | about 1.2 steps/s    |
| Total time           | about 8 days              | about 8 days         |

- 测试

| Parameters           | NPU                       | GPU                  |
| -------------------- | ------------------------- | -------------------- |
| Resource             | Ascend 910A               | Tesla V100-PCIE-16GB |
| Batch size           | 1                         | 1                    |
| Speed                | about 0.041 s/batch       | about 0.024 s/batch  |
| Metrics              | MPJPE=150.5, PA_MPJPE=101.4 | MPJPE=146.9, PA_MPJPE=90.6 |
| Total time           | about 144 seconds         | about 93 seconds     |

<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码<a name="section08421615141513"></a>

```
|-- LICENSE
|-- README.md
|-- demo
|   |-- coco1.png
|   |-- coco2.png
|   |-- coco3.png
|   |-- coco4.png
|   |-- coco5.png
|   |-- coco6.png
|   |-- example
|   |   |-- 3D_Mesh_overlay.jpg
|   |   |-- 3D_mesh.jpg
|   |   |-- diff_vp1.jpg
|   |   |-- diff_vp2.jpg
|   |   |-- input.jpg
|   |   `-- joint_projection.jpg
|   |-- im1954.jpg
|   |-- im1963.jpg
|   |-- random.jpg
|   `-- random_keypoints.json
|-- modelzoo_level.txt
|-- overview.png
|-- requirements.txt
|-- scripts
|   |-- demo.sh
|   |-- freeze_graph.sh
|   |-- inference_from_pb.sh
|   |-- test.sh
|   |-- train_1p_ci.sh
|   `-- train_testcase.sh
`-- src
    |-- RunModel.py
    |-- __init__.py
    |-- benchmark
    |   |-- __init__.py
    |   `-- eval_util.py
    |-- config.py
    |-- data_loader.py
    |-- datasets
    |   |-- __init__.py
    |   `-- common.py
    |-- demo.py
    |-- eval.py
    |-- freeze_graph.py
    |-- inference_from_pb.py
    |-- main.py
    |-- models.py
    |-- ops.py
    |-- ops_info.json
    |-- tf_smpl
    |   |-- LICENSE
    |   |-- __init__.py
    |   |-- batch_lbs.py
    |   |-- batch_smpl.py
    |   `-- projection.py
    |-- trainer.py
    `-- util
        |-- __init__.py
        |-- data_utils.py
        |-- image.py
        |-- openpose.py
        `-- renderer.py
```

## 训练过程<a name="section1589455252218"></a>

以下是训练过程中的部分输出日志(省略了警告等信息)：

```
Using 3D labels!!
making /home/data/zhanghy/logs/HMR_3DSUP_coco-lsp-lsp_ext-mpi_inf_3dhp-mpii_CMU-jointLim_resnet_fc3_dropout_Elr1e-05_kp-weight60_Dlr1e-04_3dsup-weight60_Nov17_1032
Using translation jitter: 20
Using translation jitter: 20
Iteration 0
Iteration 1
Reuse is on!
Iteration 2
Reuse is on!
collecting batch norm moving means!!
Setting up optimizer..
Done initializing trainer!
Fine-tuning from /home/data/zhanghy/hmr_models/resnet_v2_50.ckpt
[*] MODEL dir: /home/data/zhanghy/logs/HMR_3DSUP_coco-lsp-lsp_ext-mpi_inf_3dhp-mpii_CMU-jointLim_resnet_fc3_dropout_Elr1e-05_kp-weight60_Dlr1e-04_3dsup-weight60_Nov17_1032
[*] PARAM path: /home/data/zhanghy/logs/HMR_3DSUP_coco-lsp-lsp_ext-mpi_inf_3dhp-mpii_CMU-jointLim_resnet_fc3_dropout_Elr1e-05_kp-weight60_Dlr1e-04_3dsup-weight60_Nov17_1032/params.json
itr 1/(epoch 0.0): time 176.181, Enc_loss: 50.4540, Disc_loss: 29.1965, MPJPE: 443.7, PA_MPJPE: 223.8
itr 2/(epoch 0.0): time 37.8204, Enc_loss: 48.7965, Disc_loss: 29.1529, MPJPE: 416.6, PA_MPJPE: 217.3
itr 3/(epoch 0.0): time 0.540956, Enc_loss: 46.9744, Disc_loss: 29.0016, MPJPE: 432.4, PA_MPJPE: 213.5
itr 4/(epoch 0.0): time 0.514123, Enc_loss: 45.3052, Disc_loss: 28.9504, MPJPE: 405.8, PA_MPJPE: 200.2
itr 5/(epoch 0.0): time 0.509263, Enc_loss: 44.0956, Disc_loss: 28.7756, MPJPE: 405.2, PA_MPJPE: 197.6
itr 6/(epoch 0.0): time 0.525802, Enc_loss: 45.8788, Disc_loss: 28.7634, MPJPE: 449.1, PA_MPJPE: 231.7
itr 7/(epoch 0.0): time 0.527166, Enc_loss: 44.3681, Disc_loss: 28.5843, MPJPE: 422.5, PA_MPJPE: 204.1
itr 8/(epoch 0.0): time 0.525894, Enc_loss: 41.2173, Disc_loss: 28.4972, MPJPE: 380.6, PA_MPJPE: 197.5
itr 9/(epoch 0.0): time 0.519032, Enc_loss: 44.4687, Disc_loss: 28.4676, MPJPE: 408.9, PA_MPJPE: 201.3
itr 10/(epoch 0.0): time 0.521638, Enc_loss: 41.1050, Disc_loss: 28.5805, MPJPE: 412.5, PA_MPJPE: 208.5
itr 11/(epoch 0.0): time 0.505246, Enc_loss: 42.8206, Disc_loss: 28.4334, MPJPE: 413.6, PA_MPJPE: 196.8
itr 12/(epoch 0.0): time 0.500703, Enc_loss: 42.9351, Disc_loss: 28.2978, MPJPE: 429.4, PA_MPJPE: 216.5
itr 13/(epoch 0.0): time 0.501726, Enc_loss: 40.1954, Disc_loss: 28.0917, MPJPE: 411.2, PA_MPJPE: 216.4
itr 14/(epoch 0.0): time 0.500709, Enc_loss: 40.8933, Disc_loss: 28.0480, MPJPE: 394.0, PA_MPJPE: 205.4
itr 15/(epoch 0.0): time 0.509687, Enc_loss: 41.7622, Disc_loss: 27.9708, MPJPE: 422.2, PA_MPJPE: 208.8
itr 16/(epoch 0.0): time 0.504324, Enc_loss: 41.1454, Disc_loss: 28.0635, MPJPE: 450.0, PA_MPJPE: 229.1
itr 17/(epoch 0.0): time 0.506356, Enc_loss: 39.5239, Disc_loss: 27.9105, MPJPE: 400.3, PA_MPJPE: 225.0
itr 18/(epoch 0.0): time 0.509052, Enc_loss: 39.4046, Disc_loss: 27.8875, MPJPE: 444.2, PA_MPJPE: 217.0
itr 19/(epoch 0.0): time 0.518193, Enc_loss: 39.7470, Disc_loss: 27.8729, MPJPE: 412.5, PA_MPJPE: 206.0
itr 20/(epoch 0.0): time 0.507331, Enc_loss: 38.5256, Disc_loss: 27.7573, MPJPE: 365.3, PA_MPJPE: 201.1
itr 21/(epoch 0.0): time 0.50522, Enc_loss: 39.3089, Disc_loss: 27.7024, MPJPE: 403.2, PA_MPJPE: 217.8
itr 22/(epoch 0.0): time 0.503716, Enc_loss: 38.7906, Disc_loss: 27.6706, MPJPE: 391.5, PA_MPJPE: 245.5
itr 23/(epoch 0.0): time 0.512286, Enc_loss: 38.2671, Disc_loss: 27.6135, MPJPE: 394.9, PA_MPJPE: 197.9
itr 24/(epoch 0.0): time 0.510647, Enc_loss: 38.9132, Disc_loss: 27.4497, MPJPE: 407.7, PA_MPJPE: 193.8
itr 25/(epoch 0.0): time 0.497592, Enc_loss: 39.3508, Disc_loss: 27.6080, MPJPE: 419.8, PA_MPJPE: 232.0
itr 26/(epoch 0.0): time 0.502402, Enc_loss: 38.2512, Disc_loss: 27.4439, MPJPE: 395.4, PA_MPJPE: 213.8
itr 27/(epoch 0.0): time 0.509332, Enc_loss: 38.0693, Disc_loss: 27.2890, MPJPE: 385.8, PA_MPJPE: 202.3
itr 28/(epoch 0.0): time 0.509825, Enc_loss: 37.8147, Disc_loss: 27.3451, MPJPE: 411.4, PA_MPJPE: 212.1
itr 29/(epoch 0.0): time 0.509754, Enc_loss: 38.3129, Disc_loss: 27.3144, MPJPE: 388.7, PA_MPJPE: 217.2
itr 30/(epoch 0.0): time 0.505074, Enc_loss: 37.7354, Disc_loss: 27.3034, MPJPE: 400.5, PA_MPJPE: 218.8
itr 31/(epoch 0.0): time 0.497437, Enc_loss: 36.9592, Disc_loss: 27.2251, MPJPE: 374.6, PA_MPJPE: 214.4
itr 32/(epoch 0.0): time 0.500021, Enc_loss: 37.7306, Disc_loss: 27.0888, MPJPE: 413.2, PA_MPJPE: 212.0
itr 33/(epoch 0.0): time 0.506365, Enc_loss: 37.9198, Disc_loss: 27.2632, MPJPE: 404.5, PA_MPJPE: 215.1
itr 34/(epoch 0.0): time 0.506459, Enc_loss: 37.9004, Disc_loss: 27.1561, MPJPE: 421.2, PA_MPJPE: 195.2
itr 35/(epoch 0.0): time 0.49973, Enc_loss: 36.6326, Disc_loss: 26.9937, MPJPE: 389.5, PA_MPJPE: 206.8
itr 36/(epoch 0.0): time 0.503747, Enc_loss: 37.2520, Disc_loss: 26.9077, MPJPE: 414.2, PA_MPJPE: 232.8
itr 37/(epoch 0.0): time 0.506596, Enc_loss: 36.5826, Disc_loss: 26.8975, MPJPE: 386.9, PA_MPJPE: 221.8
itr 38/(epoch 0.0): time 0.526306, Enc_loss: 37.0464, Disc_loss: 26.7545, MPJPE: 405.5, PA_MPJPE: 225.6
itr 39/(epoch 0.0): time 0.515768, Enc_loss: 36.3895, Disc_loss: 26.8762, MPJPE: 408.7, PA_MPJPE: 221.9
itr 40/(epoch 0.0): time 0.498872, Enc_loss: 36.9410, Disc_loss: 26.7707, MPJPE: 378.2, PA_MPJPE: 219.9
itr 41/(epoch 0.0): time 0.505314, Enc_loss: 35.6102, Disc_loss: 26.7671, MPJPE: 404.5, PA_MPJPE: 212.5
itr 42/(epoch 0.0): time 0.519993, Enc_loss: 34.7663, Disc_loss: 26.7192, MPJPE: 369.1, PA_MPJPE: 209.5
itr 43/(epoch 0.0): time 0.504079, Enc_loss: 35.6807, Disc_loss: 26.6076, MPJPE: 393.1, PA_MPJPE: 202.6
itr 44/(epoch 0.0): time 0.504334, Enc_loss: 37.3809, Disc_loss: 26.5914, MPJPE: 421.5, PA_MPJPE: 228.3
itr 45/(epoch 0.0): time 0.559708, Enc_loss: 35.4037, Disc_loss: 26.6130, MPJPE: 366.0, PA_MPJPE: 239.2
itr 46/(epoch 0.0): time 0.545779, Enc_loss: 34.8465, Disc_loss: 26.4795, MPJPE: 371.6, PA_MPJPE: 217.0
itr 47/(epoch 0.0): time 0.518779, Enc_loss: 37.1610, Disc_loss: 26.4679, MPJPE: 403.5, PA_MPJPE: 232.8
itr 48/(epoch 0.0): time 0.508157, Enc_loss: 36.1911, Disc_loss: 26.3141, MPJPE: 377.7, PA_MPJPE: 228.4
itr 49/(epoch 0.0): time 0.525561, Enc_loss: 35.7774, Disc_loss: 26.4405, MPJPE: 377.0, PA_MPJPE: 215.4
itr 50/(epoch 0.0): time 0.512972, Enc_loss: 35.0450, Disc_loss: 26.3283, MPJPE: 384.8, PA_MPJPE: 220.5
itr 51/(epoch 0.0): time 0.517002, Enc_loss: 35.8949, Disc_loss: 26.2280, MPJPE: 393.1, PA_MPJPE: 214.1
itr 52/(epoch 0.0): time 0.514142, Enc_loss: 34.8833, Disc_loss: 26.2201, MPJPE: 386.5, PA_MPJPE: 214.9
itr 53/(epoch 0.0): time 0.538229, Enc_loss: 34.7545, Disc_loss: 26.2244, MPJPE: 380.3, PA_MPJPE: 213.8
itr 54/(epoch 0.0): time 0.515343, Enc_loss: 35.8696, Disc_loss: 26.1917, MPJPE: 393.1, PA_MPJPE: 219.6
itr 55/(epoch 0.0): time 0.51406, Enc_loss: 35.2001, Disc_loss: 26.1996, MPJPE: 384.9, PA_MPJPE: 212.0
itr 56/(epoch 0.0): time 0.515298, Enc_loss: 34.7053, Disc_loss: 26.1178, MPJPE: 352.0, PA_MPJPE: 227.9
itr 57/(epoch 0.0): time 0.532556, Enc_loss: 34.4111, Disc_loss: 25.9166, MPJPE: 379.6, PA_MPJPE: 197.1
itr 58/(epoch 0.0): time 0.520014, Enc_loss: 34.3310, Disc_loss: 26.1053, MPJPE: 355.8, PA_MPJPE: 216.2
itr 59/(epoch 0.0): time 0.514048, Enc_loss: 34.9275, Disc_loss: 25.9895, MPJPE: 413.1, PA_MPJPE: 217.8
itr 60/(epoch 0.0): time 0.527809, Enc_loss: 33.7980, Disc_loss: 25.9843, MPJPE: 378.1, PA_MPJPE: 215.8
itr 61/(epoch 0.0): time 0.548115, Enc_loss: 34.4206, Disc_loss: 25.9207, MPJPE: 386.7, PA_MPJPE: 220.4
itr 62/(epoch 0.0): time 0.510225, Enc_loss: 34.7810, Disc_loss: 25.8139, MPJPE: 388.5, PA_MPJPE: 215.5
itr 63/(epoch 0.0): time 0.515452, Enc_loss: 33.2548, Disc_loss: 25.7526, MPJPE: 328.8, PA_MPJPE: 203.9
itr 64/(epoch 0.0): time 0.505665, Enc_loss: 33.1721, Disc_loss: 25.7438, MPJPE: 390.4, PA_MPJPE: 198.1
itr 65/(epoch 0.0): time 0.506251, Enc_loss: 33.4285, Disc_loss: 25.6607, MPJPE: 382.6, PA_MPJPE: 213.7
itr 66/(epoch 0.0): time 0.513007, Enc_loss: 33.4508, Disc_loss: 25.6198, MPJPE: 389.8, PA_MPJPE: 230.5
itr 67/(epoch 0.0): time 0.514643, Enc_loss: 32.7640, Disc_loss: 25.6297, MPJPE: 365.4, PA_MPJPE: 224.7
itr 68/(epoch 0.0): time 0.52928, Enc_loss: 34.2783, Disc_loss: 25.3520, MPJPE: 379.2, PA_MPJPE: 222.7
itr 69/(epoch 0.0): time 0.519741, Enc_loss: 32.9280, Disc_loss: 25.5007, MPJPE: 353.8, PA_MPJPE: 218.7
itr 70/(epoch 0.0): time 0.498592, Enc_loss: 33.3684, Disc_loss: 25.3150, MPJPE: 385.2, PA_MPJPE: 216.6
itr 71/(epoch 0.0): time 0.499223, Enc_loss: 32.7738, Disc_loss: 25.4097, MPJPE: 357.9, PA_MPJPE: 206.1
itr 72/(epoch 0.0): time 0.509215, Enc_loss: 33.3006, Disc_loss: 25.2183, MPJPE: 360.3, PA_MPJPE: 206.7
itr 73/(epoch 0.0): time 0.55148, Enc_loss: 33.1100, Disc_loss: 25.2352, MPJPE: 403.5, PA_MPJPE: 231.1
itr 74/(epoch 0.0): time 0.509599, Enc_loss: 33.5956, Disc_loss: 25.1852, MPJPE: 401.2, PA_MPJPE: 201.6
itr 75/(epoch 0.0): time 0.504649, Enc_loss: 32.5600, Disc_loss: 25.2372, MPJPE: 347.9, PA_MPJPE: 209.7
itr 76/(epoch 0.0): time 0.510423, Enc_loss: 32.4261, Disc_loss: 25.0331, MPJPE: 345.2, PA_MPJPE: 222.5
itr 77/(epoch 0.0): time 0.510194, Enc_loss: 33.1493, Disc_loss: 24.8876, MPJPE: 369.0, PA_MPJPE: 209.7
itr 78/(epoch 0.0): time 0.511836, Enc_loss: 32.3667, Disc_loss: 24.8472, MPJPE: 352.2, PA_MPJPE: 215.7
itr 79/(epoch 0.0): time 0.498796, Enc_loss: 32.1514, Disc_loss: 25.0173, MPJPE: 367.9, PA_MPJPE: 216.2
itr 80/(epoch 0.0): time 0.506917, Enc_loss: 33.2448, Disc_loss: 24.9488, MPJPE: 381.4, PA_MPJPE: 230.3
itr 81/(epoch 0.0): time 0.508122, Enc_loss: 32.6500, Disc_loss: 24.7671, MPJPE: 380.5, PA_MPJPE: 218.7
itr 82/(epoch 0.0): time 0.500829, Enc_loss: 33.3059, Disc_loss: 24.9127, MPJPE: 369.1, PA_MPJPE: 219.6
itr 83/(epoch 0.0): time 0.506948, Enc_loss: 32.1658, Disc_loss: 24.7611, MPJPE: 364.5, PA_MPJPE: 199.2
itr 84/(epoch 0.0): time 0.508403, Enc_loss: 32.4626, Disc_loss: 24.6321, MPJPE: 370.4, PA_MPJPE: 220.9
itr 85/(epoch 0.0): time 0.506809, Enc_loss: 31.7236, Disc_loss: 24.5858, MPJPE: 353.7, PA_MPJPE: 213.8
itr 86/(epoch 0.0): time 0.507526, Enc_loss: 31.8495, Disc_loss: 24.6716, MPJPE: 382.1, PA_MPJPE: 211.0
itr 87/(epoch 0.0): time 0.514525, Enc_loss: 32.6836, Disc_loss: 24.5465, MPJPE: 395.5, PA_MPJPE: 204.9
itr 88/(epoch 0.0): time 0.51765, Enc_loss: 32.1957, Disc_loss: 24.6057, MPJPE: 356.7, PA_MPJPE: 211.9
itr 89/(epoch 0.0): time 0.502519, Enc_loss: 31.6698, Disc_loss: 24.5197, MPJPE: 361.0, PA_MPJPE: 221.4
itr 90/(epoch 0.0): time 0.506953, Enc_loss: 31.4918, Disc_loss: 24.5070, MPJPE: 395.9, PA_MPJPE: 226.2
itr 91/(epoch 0.0): time 0.499261, Enc_loss: 31.5426, Disc_loss: 24.3930, MPJPE: 353.4, PA_MPJPE: 212.0
itr 92/(epoch 0.0): time 0.510226, Enc_loss: 32.2600, Disc_loss: 24.4342, MPJPE: 408.0, PA_MPJPE: 231.9
itr 93/(epoch 0.0): time 0.524061, Enc_loss: 31.2443, Disc_loss: 24.3663, MPJPE: 353.9, PA_MPJPE: 213.4
itr 94/(epoch 0.0): time 0.519159, Enc_loss: 31.7764, Disc_loss: 24.3052, MPJPE: 358.0, PA_MPJPE: 203.6
itr 95/(epoch 0.0): time 0.511798, Enc_loss: 31.7357, Disc_loss: 24.1852, MPJPE: 337.6, PA_MPJPE: 210.6
itr 96/(epoch 0.0): time 0.507727, Enc_loss: 31.5453, Disc_loss: 24.2592, MPJPE: 373.0, PA_MPJPE: 222.9
itr 97/(epoch 0.0): time 0.514523, Enc_loss: 30.8900, Disc_loss: 24.2750, MPJPE: 354.1, PA_MPJPE: 221.0
itr 98/(epoch 0.0): time 0.512634, Enc_loss: 32.3578, Disc_loss: 24.0895, MPJPE: 372.3, PA_MPJPE: 219.7
itr 99/(epoch 0.0): time 0.505851, Enc_loss: 31.8015, Disc_loss: 24.0320, MPJPE: 355.8, PA_MPJPE: 228.3
itr 100/(epoch 0.0): time 0.512768, Enc_loss: 32.0025, Disc_loss: 24.2050, MPJPE: 407.4, PA_MPJPE: 226.6
```