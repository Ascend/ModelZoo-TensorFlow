## 3d-pose-baseline

### 基本信息

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：**Human Pose Estimation

**版本（Version）：1.1**

**修改时间（Modified） ：2021.10.26**

**大小（Size）：74M**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Benchmark**

**描述（Description）：基于TensorFlow框架的3d-pose-baseline姿态估计网络训练代码**

### 概述

3d-pose-baseline是一个经典的2d-to-3d人体姿态预测网络，同时也是3d人体姿态预测的一个重要baseline。该模型的体系结构借鉴了许多这些年来深度神经网络优化方面的改进，包括但不限于(1)使用2d/3d点作为输入输出，而不是原始图像、2d概率分布作为输入，3d概率、3d动作信息、姿态系数作为输出，这能显著降低模型的收敛难度和训练时长；(2)根据模型的特点而采用已经被广泛使用的Leaky-Relu激活函数、残差连接和最大归一约束等模型参数或构造，以取得最优的模型性能。3d-pose-baseline证明了仅需要一个很简单的模型架构，就能从人体2d骨骼点中还原出其在3d空间中的骨骼点坐标。

- 参考论文：

  [Martinez, Julieta et al. “A Simple Yet Effective Baseline for 3d Human Pose Estimation.” *2017 IEEE International Conference on Computer Vision (ICCV)* (2017): 2659-2668.](https://arxiv.org/pdf/1705.03098.pdf)

- 参考实现：

  [3d-pose-baseline](https://github.com/una-dinosauria/3d-pose-baseline)

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
  - Linear size：1024
  - Batch size：64
  - Learning rate(LR)：0.001
  - Optimizer：Adam
  - Decay rate：0.96
  - Decay steps：100000
  - Train epoch：200
  - number of joints：17

### 支持特性

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 否       |
| 混合精度   | 是       |
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

1. 硬件环境准备请参见各硬件产品文档"[驱动和固件安装升级指南](https://support.huawei.com/enterprise/zh/category/ai-computing-platform-pid-1557196528909)"。需要在硬件设备上安装与CANN版本配套的固件与驱动。

2. 宿主机上需要安装Docker并登录[Ascend Hub中心](https://ascendhub.huawei.com/#/detail?name=ascend-tensorflow-arm)获取镜像。

   当前模型支持的镜像列表如[表1](https://gitee.com/ascend/modelzoo/blob/master/contrib/TensorFlow/Research/cv/3D-POSE-BASELINE_ID0795_for_TensorFlow/README.md#zh-cn_topic_0000001074498056_table1519011227314)所示。

   **表 1** 镜像列表

   | *镜像名称*                                                   | *镜像版本* | *配套CANN版本*                                               |
   | ------------------------------------------------------------ | ---------- | ------------------------------------------------------------ |
   | $\circ$ *ARM架构：[ascend-tensorflow-arm](https://ascend.huawei.com/ascendhub/#/detail?name=ascend-tensorflow-arm)* <br>$\circ$ *x86架构：[ascend-tensorflow-x86](https://ascend.huawei.com/ascendhub/#/detail?name=ascend-tensorflow-x86)* | *20.2.0*   | *[20.2](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software)* |

### 快速上手

- 源码准备

  单击“立即下载”，并选择合适的下载方式下载源码包，并解压到合适的工作路径。

- 数据集准备

  1. 模型训练使用Human3.6M数据集，数据集请用户自行前往[Human3.6M](http://vision.imar.ro/human3.6m/)申请。

  2. 成功注册并获得下载权限后，登陆并下载`D3 Positions`文件中的主题(subject)`[1, 5, 6, 7, 8, 9, 11]`，并将它们放到`data/h36m`文件夹中，此时你的文件结构应该如下图所示。

     ```bash
     src/
     README.md
     LICENCE
     ...
     data/
       └── h36m/
         ├── Poses_D3_Positions_S1.tgz
         ├── Poses_D3_Positions_S11.tgz
         ├── Poses_D3_Positions_S5.tgz
         ├── Poses_D3_Positions_S6.tgz
         ├── Poses_D3_Positions_S7.tgz
         ├── Poses_D3_Positions_S8.tgz
         └── Poses_D3_Positions_S9.tgz
     ```

  3. 移动到这个文件目录，解压数据集。

     ```bash
     cd data/h36m/
     for file in *.tgz; do tar -xvzf $file; done
     ```

  4. 下载`code-v1.2.zip` 文件，解压并复制 `metadata.xml` 文件到`data/h36m/`文件夹下。

     现在，你的文件结构应该如下图所示。

     ```bash
     data/
       └── h36m/
         ├── metadata.xml
         ├── S1/
         ├── S11/
         ├── S5/
         ├── S6/
         ├── S7/
         ├── S8/
         └── S9/
     ```

  5. 最后，你需要对文件名做一些简单的修改，以保证文件名的一致。

     ```bash
     mv h36m/S1/MyPoseFeatures/D3_Positions/TakingPhoto.cdf \
        h36m/S1/MyPoseFeatures/D3_Positions/Photo.cdf
     
     mv h36m/S1/MyPoseFeatures/D3_Positions/TakingPhoto\ 1.cdf \
        h36m/S1/MyPoseFeatures/D3_Positions/Photo\ 1.cdf
     
     mv h36m/S1/MyPoseFeatures/D3_Positions/WalkingDog.cdf \
        h36m/S1/MyPoseFeatures/D3_Positions/WalkDog.cdf
     
     mv h36m/S1/MyPoseFeatures/D3_Positions/WalkingDog\ 1.cdf \
        h36m/S1/MyPoseFeatures/D3_Positions/WalkDog\ 1.cdf
     ```

     至此，所有准备工作已经完成。

### 模型训练

- 启动训练之前，首先要配置程序运行相关环境变量。

  环境变量配置信息参见：

  [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend 910训练平台环境变量设置?sort_id=3148819)

- 快速Demo

  你可以通过训练1个epoch并可视化结果来验证代码的正确性。训练可以执行命令：

  ```bash
  python src/predict_3dpose.py --camera_frame --residual --batch_norm --dropout 0.5 --max_norm --evaluateActionWise --epochs 1
  ```

  然后，你可以通过一下命令可视化结果：

  ```bash
  python src/predict_3dpose.py --camera_frame --residual --batch_norm --dropout 0.5 --max_norm --evaluateActionWise --epochs 1 --sample --load 24371
  ```

  你应该会得到一个类似下图的结果：

  ![visualization](src/imgs/viz_example.png)

- 若你想要用Ground Truth的2d探测结果，进行正式训练，可以使用如下命令：

  ```bash
  python src/predict_3dpose.py --camera_frame --residual --batch_norm --dropout 0.5 --max_norm --evaluateActionWise
  ```

  该结果对应于对应于论文中Table 2的最后一行 `Ours (GT detections) (MA)`。

  该模型还提供了很多额外的可选训练参数，具体参数请参考`高级参考/脚本参数`。

### 高级参考

#### 脚本和示例代码

```bash
├── README.md                                 //代码说明文档
├── LICENCE                                   //许可证
├── src
│    ├──boot_modelarts.py                     //ModelArts启动代码，非必需
│    ├──camera.py                             //处理相机的基本操作
│    ├──data_utils.py                         //数据处理
│    ├──help_modelarts.py                     //ModelArts与OBS之间进行文件传输，非必需
│    ├──linear_model.py                       //网络构建
│    ├──predict_3dpose.py                     //网络训练和测试代码
│    ├──procrustes.py                         //Procrustes Analysis的numpy版实现
│    ├──viz.py                                //可视化结果
│    ├── scripts
│    │    ├──train_performance.sh             //单卡运行启动脚本
```

#### 脚本参数

```bash
--learning_rate         初始学习率，默认：0.001
--dropout               dropout层的保留权重概率，1代表不dropout，默认：1
--batch_size            训练时的batch大小，默认：64
--epochs                训练的epoch数，默认：200
--camera_frame          是否将3d位姿转换到相机坐标系，默认：False
--max_norm              是否使用Max-Norm约束，默认：False
--batch_norm            是否使用batchNorm，默认：False
# 数据加载
--predict_14            数据集是否是使用14个关节点，默认：False
--action                数据集使用哪些种类的动作，默认："All"
# 网络结构
--linear_size           模型每一层的大小，默认：1024
--num_layers            模型的层数，默认：2
--residual              是否使用残差结构，默认：False
# 验证
--procrustes            在测试时是否使用procrustes analysis，默认：False
--evaluateActionWise    使用h36m数据集(True)还是heva数据集(False)，默认：False
# 路径
--cameras_path          包含相机参数的h36m的元数据文件路径，默认："data/h36m/metadata.xml"
--date_dir              数据路径，默认："data/h36m/"
--train_dir             训练路径，所有日志和模型文件都会保存在此，默认："experiments"
# 训练权重还是加载权重
--sample                是否进行采样，默认：False
--use_cpu               是否使用cpu，默认：False，已弃用。
--load                  尝试加载现有权重文件，默认：0
# 精度
--use_fp16              使用fp16(True)还是fp32(False)，默认：False，已弃用。
# ModelArts
--result                模型在ModelArts上保存权重的路径，默认："result"，非必需。
--obs_dir               obs上的数据集路径，请根据实际情况修改，默认："obs://simple-pose-baseline-dataset"，非必需。
```

#### 训练过程

1. 按照“模型训练”中的步骤可完成训练流程。

2. GPU训练时的模型存储在`experiments`文件夹下，具体路径基于我们设计的模型结构和训练配置，例如：

   ```bash
   \experiments\All\dropout_0.5\epochs_200\lr_0.001\residual\depth_2\linear_size1024\batch_size_64\no_procrustes\maxnorm\batch_normalization\predict_17
   ```

   NPU训练时的模型存储路径基于用户设置的ModelArts路径和obs路径，需要用户根据自己的需求自行设置。

3. 我们在NVIDIA V100芯片上进行训练的部分日志如下：

   ```
   ......
   Working on epoch 200, batch 23500 / 24371, loss 0.031016893684864044 ... done in 4.92 ms
   Working on epoch 200, batch 23600 / 24371, loss 0.04409865662455559 ... done in 4.89 ms
   Working on epoch 200, batch 23700 / 24371, loss 0.037763651460409164 ... done in 4.90 ms
   Working on epoch 200, batch 23800 / 24371, loss 0.041793253272771835 ... done in 4.89 ms
   Working on epoch 200, batch 23900 / 24371, loss 0.03778040409088135 ... done in 4.88 ms
   Working on epoch 200, batch 24000 / 24371, loss 0.04783359915018082 ... done in 4.89 ms
   Working on epoch 200, batch 24100 / 24371, loss 0.034223660826683044 ... done in 4.87 ms
   Working on epoch 200, batch 24200 / 24371, loss 0.038663435727357864 ... done in 4.88 ms
   Working on epoch 200, batch 24300 / 24371, loss 0.04263804480433464 ... done in 4.86 ms
   =============================
   Global step:         4874200
   Learning rate:       1.37e-04
   Train loss avg:      0.0379
   =============================
   ===Action=== ==mm==
   Directions    36.97
   Discussion    43.78
   Eating        38.18
   Greeting      41.30
   Phoning       44.53
   Photo         52.10
   Posing        44.48
   Purchases     39.26
   Sitting       51.34
   SittingDown   53.76
   Smoking       42.60
   Waiting       42.89
   WalkDog       44.46
   Walking       33.06
   WalkTogether  36.17
   Average       42.99
   ===================
   Saving the model... done in 4110.07 ms
   ```

4. 我们在Ascend 910芯片上进行训练的部分日志如下：

   ```
   Working on epoch 200, batch 23500 / 24371, loss 0.031016893684864044 ... done in 4.76 ms
   Working on epoch 200, batch 23600 / 24371, loss 0.04409865662455559 ... done in 4.77 ms
   Working on epoch 200, batch 23700 / 24371, loss 0.037763651460409164 ... done in 4.80 ms
   Working on epoch 200, batch 23800 / 24371, loss 0.041793253272771835 ... done in 4.78 ms
   Working on epoch 200, batch 23900 / 24371, loss 0.03778040409088135 ... done in 4.84 ms
   Working on epoch 200, batch 24000 / 24371, loss 0.04783359915018082 ... done in 4.77 ms
   Working on epoch 200, batch 24100 / 24371, loss 0.034223660826683044 ... done in 4.77 ms
   Working on epoch 200, batch 24200 / 24371, loss 0.038663435727357864 ... done in 4.78 ms
   Working on epoch 200, batch 24300 / 24371, loss 0.04263804480433464 ... done in 4.79 ms
   =============================
   Global step:         4867528
   Learning rate:       1.37e-04
   Train loss avg:      0.0392
   =============================
   ===Action=== ==mm==
   Directions    36.91
   Discussion    43.14
   Eating        39.23
   Greeting      41.47
   Phoning       44.35
   Photo         51.72
   Posing        43.68
   Purchases     39.08
   Sitting       51.58
   SittingDown   53.06
   Smoking       43.06
   Waiting       42.84
   WalkDog       44.32
   Walking       33.28
   WalkTogether  36.87
   Average       42.97
   ===================
   Saving the model... done in 3046.34 ms
   ```

5. 综上，我们可以分别得到GPU和NPU环境下的模型精度和模型性能，我们经过多次实验求取平均值后的数据如下：

   |                    | 模型精度（MSE, mm, $\downarrow$） | 模型性能（time cost / batch, ms, $\downarrow$） |
   | :----------------: | :-------------------------------: | :---------------------------------------------: |
   | GPU（NVIDIA V100） |               43.21               |                      4.88                       |
   | NPU（Ascend 910）  |             **43.18**             |                    **4.83**                     |
   
   
