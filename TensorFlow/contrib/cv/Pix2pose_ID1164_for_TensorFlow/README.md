## Pix2pose

### 基本信息

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：**6D Posture Estimation

**版本（Version）：1.2**

**修改时间（Modified） ：2021.12.27**

**大小（Size）：74M**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Benchmark**

**描述（Description）：基于TensorFlow框架的Pix2pose的6D姿态估计网络训练代码**

### 概述

Pix2Pose是一种经典的6D姿势估计方法。该模型可以在没有纹理的3D模型的情况下预测每个目标像素的三维坐标，解决了遮挡、对称和无纹理等问题，仅使用RGB图像来估计物体的6D姿势，并且能够构建具有精确纹理的三维模型。Pix2Pose设计了一个自动编码器结构来估计三维坐标和每个像素的预期误差，然后在多个阶段使用这些像素的预测来形成二维到三维的对应关系，用RANSAC迭代的PnP算法直接计算姿态，并利用生成式对抗训练来精确地覆盖被遮挡的部分，对遮挡的情况具有鲁棒性，Pix2Pose还提出了一个新的损失函数，即变换器损失函数，用于将预测的姿态引导到最近的对称姿态来处理对称目标。

- 参考论文：

  [Park K ,  Patten T et al.  "Pix2Pose: Pixel-Wise Coordinate Regression of Objects for 6D Pose Estimation."  *2019 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 2020.]
(https://arxiv.org/pdf/1908.07433.pdf)

- 参考实现：

  [Pix2Pose](https://github.com/kirumang/Pix2Pose)

- 适配昇腾 AI 处理器的实现：

  todo：添加该仓库的网页链接

- 通过Git获取对应commit_id的代码方法如下：

  ```
  git clone {repository_url}		# 克隆仓库的代码
  cd {repository_name}    		# 切换到模型的代码仓目录
  git checkout {branch}			# 切换到对应分支
  git reset --hard {commit_id}	# 代码设置到对应的commit_id
  cd {code_path}					# 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```

### 默认配置

- 训练超参

  - Batch size: 50
  - LR scheduler: linear
  - Learning rate(LR):  0.0001
  - Optimizer: AdamOptimizer
  - Train epoch: 11

### 支持特性

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 否       |
| 混合精度   | 否       |
| 并行数据   | 否       |

###  混合精度训练

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

### 开启混合精度

脚本已默认开启混合精度，设置precision_mode参数的脚本参考如下。

```python
global_config = tf.ConfigProto()
custom_op = global_config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True
custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
global_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
```

### 训练环境准备

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
### 快速上手

- 源码准备

  单击“立即下载”，并选择合适的下载方式下载源码包，并解压到合适的工作路径。

- 数据集准备

  1. 模型训练使用国际BOP挑战赛中的T-LESS数据集以及coco2017，数据集请用户自行申请。
  2. 成功注册并获得下载权限后，登陆并下载`T-LESS`数据集中的`[Base archive, Object models, Real training images of isolated objects, All test images]`以及`coco`数据集中的`[train2017]`，并将它们放到`dataset/tless`文件夹中，此时你的文件结构应该如下图所示。

     ```bash
     src/
     README.md
     LICENCE
     ...
     dataset/
       └── tless/
         ├── tless_base.zip
         ├── tless_models.zip
         ├── tless_train_primesense.zip
         ├── tless_test_primesense_all
         └── train2017.zip
     ```

  3. 移动到这个文件目录，解压数据集。

     ```bash
     cd dataset/tless/
     for file in *.zip; do unzip -xvzf $file; done
     ```
    解压数据集后，你的文件结构应该如下图所示。
     ```bash
     dataset/
       └── tless/
         ├── tless_base
         ├── tless_models
         ├── test_primesense
         ├── train_primesense
         └── train2017
     ```

  4. 将‘tless_base’文件夹中的文件全部复制到`dataset/tless/`文件夹下，将‘tless_models’文件夹中的三个文件夹全部复制到`dataset/tless/`文件夹下。
     现在，你的文件结构应该如下图所示。

     ```bash
     dataset/
       └── tless/
         ├── tless_cad
         ├── tless_eval
         ├── tless_reconst
         ├── camera_primesense.json
         ├── dataset_info.md
         ├── test_targets_bop18.json
         ├── test_targets_bop19.json
         ├── test_primesense
         ├── train_primesense
         └── train2017
     ```
  5. 下载github源代码公布的参数后，将`weight_detection`移至该文件夹下，你的文件结构应该如下图所示。
     ```bash
     dataset/
       └── tless/
         ├── tless_cad
         ├── tless_eval
         ├── tless_reconst
         ├── camera_primesense.json
         ├── dataset_info.md
         ├── test_targets_bop18.json
         ├── test_targets_bop19.json
         ├── test_primesense
         ├── train_primesense
         ├── weight_detection
         └── train2017
     ```
  6. 你需要对文件名做一些简单的修改，以保证文件名的一致。

     ```bash
     mv dataset/tless/camera_primesense.json \
        dataset/tless/camera.json
     ```
  7. 至此，所有准备工作已经完成。


  8. 执行以下命令用于获得预处理数据，由于代码的执行需要使用OpenGL库，因此下列两行命令建议在GPU环境下运行，运行成功后分别生成model_xyz和train_xyz两个数据集：

     1) 执行`2_1_ply_file_to_3d_coord_model.py`，执行该命令时，需要将`bop_io.py`中的`get_dataset`函数的两个参数`train`和`eval`都置为`False`，执行其余步骤均执行代码默认设置即`train`为`True`，`eval`为`False`。输入：`model_reconst`、`train_primesense`和`test_primesense`。输出：`model_xyz`。



     2) 执行2_2_render_pix2pose_training.py，输入：`model_cad`、`train_primesense`和`test_primesense`。输出：train_xyz。

 ```
 python3.7 ${code_dir}/2_1_ply_file_to_3d_coord_model.py --data_path=${dataset_path} --output_path=${output_path} cfg/cfg_tless_paper.json tless
 python3.7 ${code_dir}/2_2_render_pix2pose_training.py --data_path=${dataset_path} --output_path=${output_path} cfg/cfg_tless_paper.json tless
 ``` 




  1. 将预训练所获得数据集移入`tless`文件夹下
 ```bash
     dataset/
       └── tless/
         ├── tless_cad
         ├── tless_eval
         ├── tless_reconst
         ├── camera_primesense.json
         ├── dataset_info.md
         ├── test_targets_bop18.json
         ├── test_targets_bop19.json
         ├── test_primesense
         ├── train_primesense
         ├── weight_detection
         ├── train_xyz
         ├── models_xyz
         └── train2017
 ```

  7. 至此，所有准备工作已经完成。



### 模型训练

- 启动训练之前，首先要配置程序运行相关环境变量。

  环境变量配置信息参见：

  [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend 910训练平台环境变量设置?sort_id=3148819)

- 快速demo

  模型测试所需数据集：`models_cad`、`model_eval`、`models_reconst`、`models_xyz`、`train_xyz`、`train_primesense`、`test_primesense`、`tarin2017`（coco数

据集）和`weight_detection`。

  以及参数文件：`camera.json`、`cfg_tless_paper.json`和`test_target_bop19.json`。

  你可以通过训练1个类来验证代码的正确性。obj_id表示类的序号，训练可以执行命令：

  ```
  python3.7 ${code_dir}/3_train_pix2pose.py --data_path=${dataset_path} --output_path=${output_path}  --obj_id='01' 
  ```
  输出：`pix2pose_weights`包含每个类的训练结果，存储于inference.hdf5文件，
  
  如果要完整地训练Pix2pose的模型需要训练30 个类，将obj_id修改为"01"、"02" ... "30".

### 模型测试
  模型测试所需数据集：`models_cad`、`model_eval`、`models_reconst`、`models_xyz`、`pix2pose_weights`、`test_primesense`、`train_xyz`和`weight_detection`

  以及参数文件：`camera.json`、`cfg_tless_paper.json`、`test_target_bop19.json`

 ```
 python3.7 ${code_dir}/5_evaluation_bop_basic.py --data_path=${dataset_path} --output_path=${output_path}
 ```
  输出：pix2pose-iccv19_tless-test-primesense.csv
### 高级参考

#### 脚本和示例代码

```bash
├── README.md                                 //代码说明文档
├── LICENCE                                   //许可证
├── src
│    ├──tool
│        ├──2_1_ply_file_to_3d_coord_model.py             //数据处理，将点云数据转化为3D模型，需要使用OpenGL库，仅能在GPU上实现
│        ├──2_2_render_pix2pose_training.py                //数据处理，将点云数据转化为np数据，需要使用OpenGL库，仅能在GPU上实现
│        ├──3_train_pix2pose.py                                   //数据训练代码
│        ├──4_convert_weights_inference .py               //数据处理，将训练权重转化为inference文件，非必需
│        ├──5_evaluation_bop_basic.py                        //网络训练和测试代码，代码将测试结果存于CSV文件中
│        ├──bop_io.py                                                 //bop挑战赛的io接口，用于读取json里面的参数
│        ├──cfg_tless_paper.json                                 //训练和测试参数
│        ├──modelarts_entry.py                                  //ModelArts启动代码
│        ├──npu_train.md                                          //网络训练和测试代码的执行命令
│        ├──render_training_img.py                           //渲染图像
│        ├──train_performance.sh                        //单卡运行启动脚本
```


#### 训练过程

1. 按照“模型训练”中的步骤可完成训练流程。

2. 我们在NVIDIA V100芯片上进行训练,训练类`15`的部分日志如下：
```
Epoch10-Iter2088/2096:Mean-[0.06593], Disc-[0.4911], Recon-[0.0622], Gen-[1.3465]],lr=0.000010
Epoch10-Iter2089/2096:Mean-[0.06593], Disc-[0.7870], Recon-[0.0704], Gen-[0.9368]],lr=0.000010
Epoch10-Iter2090/2096:Mean-[0.06593], Disc-[0.3239], Recon-[0.0699], Gen-[1.4635]],lr=0.000010
Epoch10-Iter2091/2096:Mean-[0.06593], Disc-[0.4481], Recon-[0.0660], Gen-[0.8735]],lr=0.000010
Epoch10-Iter2092/2096:Mean-[0.06594], Disc-[0.5016], Recon-[0.0876], Gen-[0.9202]],lr=0.000010
Epoch10-Iter2093/2096:Mean-[0.06595], Disc-[0.3563], Recon-[0.0768], Gen-[1.7627]],lr=0.000010
Epoch10-Iter2094/2096:Mean-[0.06595], Disc-[0.7449], Recon-[0.0698], Gen-[0.7307]],lr=0.000010
Epoch10-Iter2095/2096:Mean-[0.06595], Disc-[0.4868], Recon-[0.0605], Gen-[1.2816]],lr=0.000010
Epoch10-Iter2096/2096:Mean-[0.06595], Disc-[0.3766], Recon-[0.0679], Gen-[1.2838]],lr=0.000010
Epoch10-Iter2097/2096:Mean-[0.06595], Disc-[0.5384], Recon-[0.0601], Gen-[0.9551]],lr=0.000010
disc_loss: 0.5384470224380493
dcgan_loss: [6.9644904, 0.060094148, 0.9550757]
loss improved from 0.0676 to 0.0659 saved weights
/home/pix2pose/Pix2Pose-master/Pix2Pose-master/dataset/tless/pix2pose_weights/15/pix2pose.11-0.0659.hdf5
Train finished
```

3. 我们在Ascend 910芯片上进行训练,训练类`15`的部分日志如下：
```
   Epoch10-Iter2077/2096:Mean-[0.07750], Disc-[0.7981], Recon-[0.0720], Gen-[1.4102]],lr=0.000010
   Epoch10-Iter2078/2096:Mean-[0.07749], Disc-[0.9505], Recon-[0.0699], Gen-[1.3753]],lr=0.000010
   Epoch10-Iter2079/2096:Mean-[0.07749], Disc-[0.3463], Recon-[0.0755], Gen-[1.3235]],lr=0.000010
   Epoch10-Iter2080/2096:Mean-[0.07748], Disc-[0.3442], Recon-[0.0570], Gen-[0.8986]],lr=0.000010
   Epoch10-Iter2081/2096:Mean-[0.07748], Disc-[0.3914], Recon-[0.0678], Gen-[1.0524]],lr=0.000010
   Epoch10-Iter2082/2096:Mean-[0.07747], Disc-[0.7480], Recon-[0.0649], Gen-[1.1183]],lr=0.000010
   Epoch10-Iter2083/2096:Mean-[0.07746], Disc-[0.5832], Recon-[0.0624], Gen-[0.5142]],lr=0.000010
   Epoch10-Iter2084/2096:Mean-[0.07746], Disc-[0.4171], Recon-[0.0726], Gen-[1.0140]],lr=0.000010
   Epoch10-Iter2085/2096:Mean-[0.07746], Disc-[0.3772], Recon-[0.0763], Gen-[1.2385]],lr=0.000010
   Epoch10-Iter2086/2096:Mean-[0.07746], Disc-[0.6575], Recon-[0.0815], Gen-[0.8145]],lr=0.000010
   Epoch10-Iter2087/2096:Mean-[0.07746], Disc-[0.5178], Recon-[0.0794], Gen-[0.5486]],lr=0.000010
   Epoch10-Iter2088/2096:Mean-[0.07747], Disc-[0.9572], Recon-[0.0832], Gen-[1.2526]],lr=0.000010
   Epoch10-Iter2089/2096:Mean-[0.07746], Disc-[0.4130], Recon-[0.0693], Gen-[0.9494]],lr=0.000010
   Epoch10-Iter2090/2096:Mean-[0.07746], Disc-[0.3749], Recon-[0.0693], Gen-[1.2845]],lr=0.000010
   Epoch10-Iter2091/2096:Mean-[0.07746], Disc-[0.9384], Recon-[0.0740], Gen-[1.4234]],lr=0.000010
   Epoch10-Iter2092/2096:Mean-[0.07746], Disc-[0.5098], Recon-[0.0914], Gen-[1.5642]],lr=0.000010
   Epoch10-Iter2093/2096:Mean-[0.07746], Disc-[0.3655], Recon-[0.0752], Gen-[1.0943]],lr=0.000010
   Epoch10-Iter2094/2096:Mean-[0.07746], Disc-[0.5028], Recon-[0.0622], Gen-[1.2247]],lr=0.000010
   Epoch10-Iter2095/2096:Mean-[0.07745], Disc-[0.9028], Recon-[0.0650], Gen-[1.2567]],lr=0.000010
   Epoch10-Iter2096/2096:Mean-[0.07745], Disc-[0.4163], Recon-[0.0690], Gen-[1.0187]],lr=0.000010
   Epoch10-Iter2097/2096:Mean-[0.07745], Disc-[0.3689], Recon-[0.0840], Gen-[1.4939]],lr=0.000010
```
4. 默认训练11Epoch*2096iter次后，可在obs/训练作业/output/model文件夹下得到相应的网络模型，在训练日志中得到相应的指标参数

   | 迁移模型    | 训练次数  |  NPU精度  |  GPU精度  | 
   | ---------- | --------  | --------  |  -------- |
   |Pix2pose   | 11*2096   | 0.077      |0.06~0.08    |

