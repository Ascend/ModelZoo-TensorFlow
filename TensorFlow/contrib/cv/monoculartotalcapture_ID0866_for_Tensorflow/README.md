- [基本信息](#基本信息.md)
- [概述](#概述.md)
- [训练环境准备](#训练环境准备.md)
- [快速上手](#快速上手.md)
- [迁移学习指导](#迁移学习指导.md)
- [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Object Detection**

**版本（Version）：1.1**

**修改时间（Modified） ：2022.8.29**

**大小（Size）：136.54MB**

**框架（Framework）：TensorFlow_1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的纹理跟踪方法获得时间相干运动捕获输出网络训练代码**

<h2 id="概述.md">概述</h2>

## 简述<a name="section194554031510"></a>

我们提出了第一种从单目视图输入中捕获目标人的 3D 总运动的方法。给定图像或单目视频，我们的方法从由 3D 可变形网格模型表示的身体、面部和手指重建运动。我们使用称为 3D Part Orientation Fields (POF) 的有效表示来编码公共 2D 图像空间中所有身体部位的 3D 方向。POF 由全卷积网络 (FCN) 以及联合置信图预测。为了训练我们的网络，我们收集了一个新的 3D 人体运动数据集，该数据集在多视图系统中捕获了 40 个受试者的不同全身运动。我们利用 3D 可变形人体模型通过利用模型中的姿势和形状先验从 CNN 输出重建全身姿势。我们还提出了一种基于纹理的跟踪方法来获得时间相干的运动捕捉输出。我们进行了彻底的定量评估，包括与现有的特定于身体和特定于手的方法进行比较，以及对相机视点和人体姿势变化的性能分析。最后，我们在各种具有挑战性的野外视频中展示了我们的全身动作捕捉结果

- 参考论文：

  [https://arxiv.org/abs/1812.01598](Monocular Total Capture: Posing Face, Body, and Hands in the Wild)

- 参考实现：

  https://github.com/CMU-Perceptual-Computing-Lab/MonocularTotalCapture

- 适配昇腾 AI 处理器的实现：
  
  https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/monoculartotalcapture_ID0866_for_Tensorflow

- 通过Git获取对应commit\_id的代码方法如下：
  
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    

## 默认配置<a name="section91661242121611"></a>

-   训练超参（单卡）：
   
      - lr: [1e-4, 1e-5]
      - r_iter: [100000]
      - max_iter: 400000
      - show_loss_freq: 1
      - snapshot_freq: 5000
      - loss_weight_PAF: 1.0
    

## 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 是       |
| 混合精度   | 是       |
| 数据并行   | 是       |


## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

模型默认开始混合精度：

```
    config_proto = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    custom_op = config_proto.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = 'NpuOptimizer'
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes('allow_mix_precision')
    config = npu_config_proto(config_proto=config_proto)

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

1、模型训练使用COCO2017数据集，数据集请用户自行获取

2、COCO2017数据集需要经过处理，处理方式参照github [Realtime Multi-Person Pose Estimation](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation)

3、数据集存储的路径可以在data/COCOReader.py中修改

4、下载预训练模型（文件夹名：Final_qual_domeCOCO_chest_noPAF2D）。网盘分享链接在download.md中。将下载的文件夹放入snapshots文件夹

5、monoculartotalcapture训练的模型及数据集可以参考"简述 -> 参考实现"


## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练。

    - 启动训练之前，首先要配置程序运行相关环境变量。

      环境变量配置信息参见：

      [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)

    - 单卡训练

          1. 配置训练参数
        
             首先在脚本train_full_1p.sh中，配置训练数据集路径和checkpoint保存路径，请用户根据实际路径配置data_path数据集路径，示例如下所示：
        
             ```
             # 路径参数已经代码写死可以用以下方法修改，对应的目录文件./data/COCOReader.py
             sed -i "s|/tmp/COCO_data/train2017/|$data_path/COCO_data/train2017/|g" data/COCOReader.py
             sed -i "s|/tmp/COCO_data/mask2017/|$data_path/COCO_data/mask2017/|g" data/COCOReader.py
             sed -i "s|/tmp/COCO_data/COCO.json|$data_path/COCO_data/COCO.json|g" data/COCOReader.py
             ```
        
          2. 启动训练（脚本为./test/train_full_1p.sh） 
        
             ```
             bash train_full_1p.sh --data_path
             ```

          3. 训练精度结果

            | 指标项     | GPU   | NPU   |
            |-----------|-------|-------|
            |   最后10个Loss的平均值  | 0.0038 | 0.0043 |
           


<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码

```
├── snapshots
├── training_e2e_PAF.py
├── train.sh
├── LICENSE
├── README.md
├── modelzoo_level.txt
├── requirements.txt
├──  data					        //数据集配置文件
    |--Base2DReader.py
    |--BaseReader.py
    |--COCOReader.py                                    //修改数据集路径
    |--DomeReader.py
    |--DomeReaderTempConst.py
    |--GAneratedReader.py 
    |--HumanReader.py
    |--MultiDataset.py
    |--OpenposeReader.py
    |--RHDReader.py
    |--STBReader.py
    |--TempConstReader.py
    |--TsimonDBReader.py 
    |--collect_stb.py
    |--process_MPII_mask.py			
├── test     
│    ├──train_performance_1p.sh                //训练性能入口
│    ├──train_full_1p.sh                       //训练精度入口，包含准确率评估
```

## 脚本参数<a name="section6669162441511"></a>

```
-- lr: [1e-4, 1e-5]
-- r_iter: [100000]
-- max_iter: 400000
-- show_loss_freq: 1
-- snapshot_freq: 5000
-- loss_weight_PAF: 1.0
--self.image_root = '/tmp/COCO_data/train2017/'           //数据集路径
--self.mask_root = '/tmp/COCO_data/mask2017/'             
--path_to_db = '/tmp/COCO_data/COCO.json'
```

## 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡训练。模型存储路径为${cur_path}/output/$ASCEND_DEVICE_ID，包括训练的log以及checkpoints文件。以单卡训练为例，loss信息在文件${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。