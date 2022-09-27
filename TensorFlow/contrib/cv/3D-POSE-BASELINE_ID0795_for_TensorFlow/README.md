- [基本信息](#基本信息.md)
- [概述](#概述.md)
- [训练环境准备](#训练环境准备.md)
- [快速上手](#快速上手.md)
- [迁移学习指导](#迁移学习指导.md)
- [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Pose Estimation**

**版本（Version）：1.1**

**修改时间（Modified） ：2022.8.26**

**大小（Size）：74MB**

**框架（Framework）：TensorFlow_1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的3d-pose-baseline姿态估计网络训练代码**

<h2 id="概述.md">概述</h2>

## 简述<a name="section194554031510"></a>

3d-pose-baseline是一个经典的2d-to-3d人体姿态预测网络，同时也是3d人体姿态预测的一个重要baseline。该模型的体系结构借鉴了许多这些年来深度神经网络优化方面的改进，包括但不限于(1)使用2d/3d点作为输入输出，而不是原始图像、2d概率分布作为输入，3d概率、3d动作信息、姿态系数作为输出，这能显著降低模型的收敛难度和训练时长；(2)根据模型的特点而采用已经被广泛使用的Leaky-Relu激活函数、残差连接和最大归一约束等模型参数或构造，以取得最优的模型性能。3d-pose-baseline证明了仅需要一个很简单的模型架构，就能从人体2d骨骼点中还原出其在3d空间中的骨骼点坐标
- 参考论文：

  [https://arxiv.org/pdf/1705.03098.pdf](Martinez, Julieta et al. “A Simple Yet Effective Baseline for 3d Human Pose Estimation.” *2017 IEEE International Conference on Computer Vision (ICCV)* (2017): 2659-2668.)

- 参考实现：

  https://github.com/una-dinosauria/3d-pose-baseline

- 适配昇腾 AI 处理器的实现：
  
  https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/3D-POSE-BASELINE_ID0795_for_TensorFlow

- 通过Git获取对应commit\_id的代码方法如下：
  
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    

## 默认配置<a name="section91661242121611"></a>

-   训练超参（单卡）：

      - learning_rate=1e-3
      - cameras_path
      - data_dir
      - epochs
      - dropout 0.5
    

## 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 否       |
| 混合精度   | 是       |
| 数据并行   | 否       |


## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

脚本已默认开启混合精度，设置precision_mode参数的脚本参考如下。

```python
global_config = tf.ConfigProto()
custom_op = global_config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True
custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
global_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
```

<h2 id="训练环境准备.md">训练环境准备</h2>

-  硬件环境和运行环境准备请参见《[CANN软件安装指南](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=installation-update)》
-  运行以下命令安装依赖。
```
pip3 install requirements.txt
```
说明：依赖配置文件requirements.txt文件位于模型的根目录

<h2 id="快速上手.md">快速上手</h2>

## 数据集准备<a name="section361114841316"></a>

1、模型训练使用Human3.6M数据集，数据集请用户自行前往[Human3.6M](http://vision.imar.ro/human3.6m/)申请

2、成功注册并获得下载权限后，登陆并下载`D3 Positions`文件中的主题(subject)`[1, 5, 6, 7, 8, 9, 11]`，并将它们放到`data/h36m`文件夹中，此时你的文件结构应该如下图所示。

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

3、移动到这个文件目录，解压数据集。

     ```bash
     cd data/h36m/
     for file in *.tgz; do tar -xvzf $file; done
     ```

4、下载`code-v1.2.zip` 文件，解压并复制 `metadata.xml` 文件到`data/h36m/`文件夹下。

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
5、最后，你需要对文件名做一些简单的修改，以保证文件名的一致。

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


## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练。

    - 启动训练之前，首先要配置程序运行相关环境变量。

      环境变量配置信息参见：

      [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)

    - 单卡训练

          1. 配置训练参数
        
             首先在脚本train_full_1p.sh中，配置训练数据集路径和checkpoint保存路径，请用户根据实际路径配置data_path，train_epochs，示例如下所示：
        
             ```
             # 路径参数初始化
                --data_dir ${data_path} 
                --cameras_path ${data_path}/metadata.xml 
                --camera_frame 
                --residual 
                --batch_norm 
                --dropout 0.5 
                --max_norm 
                --evaluateActionWise 
                --epochs $train_epochs 
             ```
        
          2. 启动训练（脚本为./test/train_full_1p.sh） 
        
             ```
                bash train_full_1p.sh --data_path
             ```

          3. 训练精度结果

               |                    | 模型精度（MSE, mm, $\downarrow$） | 模型性能（time cost / batch, ms, $\downarrow$）   |
               | :----------------: | :-------------------------------: | :---------------------------------------------: |
               | GPU（NVIDIA V100） |               43.21               |                      4.88                       |
               | NPU（Ascend 910）  |             **43.18**             |                    **4.83**                     |
           


<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码

```
├── LICENSE
├── README.md
├── modelzoo_level.txt
├── requirements.txt
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
├── test     
│    ├──train_performance_1p.sh                //训练性能入口
│    ├──train_full_1p.sh                       //训练精度入口，包含准确率评估
```

## 脚本参数<a name="section6669162441511"></a>

```
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

## 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡训练。模型存储路径为${cur_path}/output/$ASCEND_DEVICE_ID，包括训练的log以及checkpoints文件。以单卡训练为例，loss信息在文件${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。