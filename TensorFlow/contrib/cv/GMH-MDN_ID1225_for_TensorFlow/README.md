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

**修改时间（Modified） ：2022.8.24**

**大小（Size）：249MB**

**框架（Framework）：TensorFlow_1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Benchmark**

**描述（Description）：基于TensorFlow框架的基于多峰混合密度网络生成多个可行的 3D 姿态假设的网络**

<h2 id="概述.md">概述</h2>

## 简述<a name="section194554031510"></a>

   GMH-MDN：一种基于多峰混合密度网络生成多个可行的 3D 姿态假设的网络

- 参考论文：

  [https://arxiv.org/pdf/1904.05547.pdf](Generating Multiple Hypotheses for 3D Human Pose Estimation with Mixture Density Network)

- 参考实现：

  https://github.com/chaneyddtt/Generating-Multiple-Hypotheses-for-3D-Human-Pose-Estimation-with-Mixture-Density-Network

- 适配昇腾 AI 处理器的实现：
  
  https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/GMH-MDN_ID1225_for_TensorFlow

- 通过Git获取对应commit\_id的代码方法如下：
  
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    

## 默认配置<a name="section91661242121611"></a>

-   训练超参（单卡）：
    -  Batch size: 64
    -  LR scheduler: exponential decay
    -  Learning rate\(LR\): 0.001
    -  Train epoch: 200
    -  dropout：0.5
    -  linear_size：1024 \#ps: size of each layer(每一层神经元的个数)


## 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 否       |
| 混合精度   | 是       |
| 数据并行   | 否       |


## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

拉起脚本中，传入--precision_mode='allow_mix_precision'

```
parameter explain:
    --precision_mode         precision mode(allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision)
    --over_dump                  if or not over detection, default is False
    --data_dump_flag         data dump flag, default is False
    --data_dump_step             data dump step, default is 10
    --profiling                  if or not profiling for performance debug, default is False
    --data_path                  source data of training
    -h/--help                    show help message
```

相关代码示例:

```
  config = tf.ConfigProto(allow_soft_placement=True)
  custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
  custom_op.name = "NpuOptimizer"
  custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
  config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
  config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
  with tf.Session(config=config) as sess:
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

1、模型预训练使用 [Human3.6M]数据集  ，需用户自行申请。因申请较慢，故可在[此处](https://github.com/MendyD/human36m) 下载

2、数据集下载后，放入模型目录下，在训练脚本中指定数据集路径，可正常使用

3、GMH-MDN训练的模型及数据集可以参考"简述 -> 参考实现"



## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练。

    - 启动训练之前，首先要配置程序运行相关环境变量。

      环境变量配置信息参见：

      [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

    - 单卡训练

         1. 设置单卡训练参数（脚本位于./GMH—MDN_ID1225_for_TensorFlow/test/train_full_1p.sh），示例如下。请确保下面例子中的“data_dir，batch_size，epochs”修改为用户数据集的路径。

              ```
                --cameras_path ${cameras_path} 
                --data_dir ${data_dir} 
                --train_dir ${train_dir} 
                --load_dir ${load_dir} 
                --test ${test} 
                --load ${load} 
                --batch_size ${batch_size} 
                --epochs ${epochs} 
                --learning_rate ${learning_rate} 
              ```


              

         2. 单卡训练指令（脚本位于./GMH—MDN_ID1225_for_TensorFlow/test/train_full_1p.sh）

             ```
               bash train_full_1p.sh --train_dir
             ```
         3. 精度训练结果
             ```
             |                |  GPU | NPU  | 
             |----------------|------|--------|
             | root - Average | 58.63 | 58.37 |
             ```

<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码

```
├── README.md                                //说明文档
├── requirements.txt
├── LICENSE
├── modelarts_entry_acc.py					 
├── modelarts_entry_perf.py
├── modelzoo_level.txt
├── src
│   ├── cameras.py
│   ├── data_utils.py
│   ├── logging.conf
│   ├── mix_den_model.py
│   ├── predict_3dpose_mdm.py
│   ├── procrustes.py
│   └── viz.py
├── test
│   ├── train_full_1p.sh
│   ├── train_performance_1p.sh
```

## 脚本参数<a name="section6669162441511"></a>

```
--learning_rate		Learning rate	default:0.001
--dropout		Dropout keep probability 1 means no dropout	default:0.5	
--batch_size		batch size to use during training	default:64	
--epochs		How many epochs we should train for	default:200	
--camera_frame		Convert 3d poses to camera coordinates	default:TRUE	
--max_norm		Apply maxnorm constraint to the weights	default:TRUE	
--batch_norm		Use batch_normalization	default:TRUE				
--predict_14		predict 14 joints	default:FALSE	
--use_sh		Use 2d pose predictions from StackedHourglass	default:TRUE	
--action		The action to train on 'All' means all the actions	default:All					
--linear_size		Size of each model layer	default:1024	
--num_layers		Number of layers in the model	default:2	
--residual		Whether to add a residual connection every 2 layers	default:TRUE					
--procrustes		Apply procrustes analysis at test time	default:FALSE	
--evaluateActionWise		The dataset to use either h36m or heva	default:TRUE					
--cameras_path		Directory to load camera parameters	default:/data/h36m/cameras.h5	
--data_dir		Data directory	default:   /data/h36m/	
--train_dir		Training directory	default:/experiments/test_git/	
--load_dir		Specify the directory to load trained model	default:/Models/mdm_5_prior/				
--sample		Set to True for sampling	default:FALSE	
--test		        Set to True for sampling	default:FALSE	
--use_cpu		Whether to use the CPU	default:FALSE	
--load		        Try to load a previous checkpoint	default:0	
--miss_num		Specify how many missing joints	default:1
```

## 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡，8卡网络训练。模型存储路径为${cur_path}/output/$ASCEND_DEVICE_ID，包括训练的log以及checkpoints文件。以8卡训练为例，loss信息在文件${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。