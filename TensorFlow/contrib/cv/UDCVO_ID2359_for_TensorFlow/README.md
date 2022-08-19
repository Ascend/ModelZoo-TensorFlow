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

**修改时间（Modified） ：2022.8.19**

**大小（Size）：204KB**

**框架（Framework）：TensorFlow_1.15**

**模型格式（Model Format）：ckpt**

**精度（Precision）：FP32**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架通过摄像头运动和视觉惯性里程计估计出的稀疏深度推断密集深度训练代码**

<h2 id="概述.md">概述</h2>

## 简述<a name="section194554031510"></a>

此模型通过给定场景的RGB图像来增密稀疏点云，继而恢复3D场景。首先构造场景的分段平面脚手架，然后使用它与图像以及稀疏点一起推断密集深度。使用类似于“自我监督”的预测性交叉模态准则，跨时间测量光度一致性，前后姿势一致性以及与稀疏点云的几何兼容性。使用网格三角剖分和线性插值作为预处理步骤，使网络的参数减少了 80%，同时性能优于其他方法。提出了第一个视觉惯性+深度数据集。

- 参考论文：
  
  [https://arxiv.org/pdf/1905.08616.pdf](Wong, Alex and Fei, Xiaohan and Tsuei, Stephanie and Soatto, Stefano, "Unsupervised Depth Completion from Visual Inertial Odometry", in ICRA, 2020.)

- 参考实现：

  https://github.com/alexklwong/unsupervised-depth-completion-visual-inertial-odometry#unsupervised-depth-completion-from-visual-inertial-odometry

- 适配昇腾 AI 处理器的实现：
  
  https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/UDCVO_ID2359_for_TensorFlow

- 通过Git获取对应commit\_id的代码方法如下：
  
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    

## 默认配置<a name="section91661242121611"></a>

-   训练超参（单卡）：
    - batch-size: 8
    - height: 480
    - width: 640
    - channel: 3
    - n_epoch: 20
    - learning-rate: $0.50\times10^{-4}$（前 12 epoch），$0.25\times10^{-4}$（4 epoch），$0.12\times10^{-4}$（后 4 epoch）

## 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 否       |
| 混合精度   | 否      |
| 数据并行   | 否       |


## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

拉起脚本中，

```
 ./train_full_1p.sh --help

parameter explain:
    --precision_mode         #precision mode(allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision)
    --data_path              # dataset of training
    --output_path            # output of training
    --train_steps            # max_step for training
    --train_epochs           # max_epoch for training
    --batch_size             # batch size
    -h/--help                show help message
```
float16 类型会导致训练错误，且精度不达标。**不支持混合精度** 。修改 `voiced_main.py` 中的相关代码为 `LossScale + force_fp32` 模式解决算子溢出以及精度问题。

开启 LossScale：

```python
optimizer = tf.train.AdamOptimizer(learning_rate)

# Add Loss Scale
loss_scale_opt = optimizer
loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=2 ** 32,
                                                       incr_every_n_steps=1000,
                                                       decr_every_n_nan_or_inf=2,
                                                       decr_ratio=0.5)
optimizer = NPULossScaleOptimizer(loss_scale_opt, loss_scale_manager)
```
force_fp32精度相关代码示例:

 ```
    # Add force_fp32
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True
    # Resolve accuracy issue
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("force_fp32")
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
    session = tf.Session(config=config)

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

1、使用密度为 0.5%（1500个稀疏点）的室内场景数据集，用户需自行准备VOID_1500数据集到训练环境

2、UDCVO的模型及数据集可以参考"简述 -> 参考实现"


## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练。

    - 启动训练之前，首先要配置程序运行相关环境变量。

      环境变量配置信息参见：

      [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

    - 单卡训练


        1.首先在脚本test/train_full_1p.sh中, 训练需要根据安装教程，配置输入与输出的路径。配置训练数据集路径，请用户根据实际路径配置，数据集参数如下所示：

             ```

            --train_image_path training/void_train_image_1500.txt 
            --train_interp_depth_path training/void_train_interp_depth_1500.txt 
            --train_validity_map_path training/void_train_validity_map_1500.txt 
            --train_intrinsics_path training/void_train_intrinsics_1500.txt 
            --n_batch 8 
            --n_height 480 
            --n_width 640 
            --n_channel 3 
            --n_epoch 20 
            --n_summary 1000 
            --n_checkpoint 5000 
            --checkpoint_path ${output_path}
            ```

        2.启动训练
        
             启动单卡训练  
        
             ```
             bash train_full_1p.sh
             ```
        2.精度训练结果
        
             ```
             |         |   MAE   | RMSE     | iMAE    | iRMSE   |
            | :-----: | :-----: | :-----: | :-----: | :-----: |
            | 原项目  |  82.27  | 141.99   | 49.23   | 99.67   |
            | GPU复现 | 96.9058 | 149.2545 | 54.2917 | 89.1794 |
            | NPU复现 | 91.9726 | 143.6669 | 52.1458 | 88.7566 |
             ```             
    

<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码

```
|--LICENSE
|--README.md                                                      #说明文档									
|--test			           	                          #训练脚本目录
|	|--data_utils.py
|	|--dataloader.py     
|	|--eval_utils.py
|	|--evaluate_model.py
|	|--train_voiced.py
|	|--voiced_main.py  
|	|--voiced_model.py
|	|--losses.py                                      
|--modelarts_entry_acc.py
|--modelarts_entry_perf.py
|--modelzoo_level.txt									
|--requirements.txt                                               #所需依赖                                                 
|--test			           	                          #训练脚本目录
|	|--train_full_1p.sh
|	|--train_performance_1p.sh
```

## 脚本参数<a name="section6669162441511"></a>

```
--train_image_path training/testcase_image_1500.txt 
--train_interp_depth_path training/testcase_interp_depth_1500.txt 
--train_validity_map_path training/testcase_validity_map_1500.txt 
--train_intrinsics_path training/testcase_intrinsics_1500.txt 
--n_batch 8 
--n_height 480 
--n_width 640 
--n_channel 3 
--n_epoch 20
--learning_rates 0.50e-4,0.25e-4,0.12e-4 
--learning_bounds 12,16 
--occ_threshold 1.5 
--occ_ksize 7 
--net_type vggnet11 
--im_filter_pct 0.75 
--sz_filter_pct 0.25 
--min_predict_z 0.1 
--max_predict_z 8.0 
--w_ph 1.00 
--w_co 0.20 
--w_st 0.80 
--w_sm 0.15 
--w_sz 1.00 
--w_pc 0.10 
--pose_norm frobenius 
--rot_param exponential 
--n_summary 10 
--n_checkpoint 10 
--checkpoint_path trained_models/vggnet11_void_model
```

## 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡。模型存储路径为${cur_path}/output/$ASCEND_DEVICE_ID，包括训练的log以及checkpoints文件。以1卡训练为例，loss信息在文件${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。