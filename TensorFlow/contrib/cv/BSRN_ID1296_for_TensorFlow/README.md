- [基本信息](#基本信息.md)
- [概述](#概述.md)
- [训练环境准备](#训练环境准备.md)
- [快速上手](#快速上手.md)
- [迁移学习指导](#迁移学习指导.md)
- [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Super Resolution**

**版本（Version）：1.1**

**修改时间（Modified） ：2022.8.05**

**大小（Size）：5841704KB**

**框架（Framework）：TensorFlow_1.15**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的块状态递归网络的轻量级高效图像超分辨率**

<h2 id="概述.md">概述</h2>

## 简述<a name="section194554031510"></a>

BSRN模型的全称是：Block State-based Recursive Network。BSRN模型基于TensorFlow框架的块状态递归网络的轻量级高效图像超分辨率。

- 参考论文：

  https://paperswithcode.com/paper/lightweight-and-efficient-image-super

- 参考实现：

  https://github.com/idearibosome/tf-bsrn-sr/

- 适配昇腾 AI 处理器的实现：
  
  https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/BSRN_ID1296_for_TensorFlow

- 通过Git获取对应commit\_id的代码方法如下：
  
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    

## 默认配置<a name="section91661242121611"></a>

-   网络结构
    Super-resolution of the training dataset image、Verifying the Super Resolution of Dataset Images
-   训练超参（单卡）：
    - data_input_path：DIV2K/DIV2K_train_LR_bicubic
    - data_truth_path：DIV2K/DIV2K_train_HR
    - train_path：./checkpoints 
    - chip:'npu' 
    - model:'bsrn' 
    - dataloader:'div2k_loader' 
    - batch_size:8 
    - max_steps:300000
    - save_freq:50000 
    - scales:'4' 


## 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 是       |
| 混合精度   | 是       |
| 数据并行   | 是       |


## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

拉起脚本中，test目录下面:

```
 ./train_full_1p.sh --help

parameter explain:
    -data_input_path
    -data_truth_path
    -train_path
    -chip='npu' 
    -model='bsrn' 
    -dataloader='div2k_loader' 
    -batch_size=8 
    -max_steps=300000
    -save_freq=50000 
    -scales='4'
    -h/--help                    show help message
```

相关代码示例:

```
  custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
  custom_op.name = "NpuOptimizer"
  custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
  sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
  sess_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
  custom_op.parameter_map["dynamic_input"].b = True
  custom_op.parameter_map["dynamic_graph_execute_mode"].s = tf.compat.as_bytes("lazy_recompile")
  self.tf_session = tf.compat.v1.Session(config=sess_config)

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

1、用户自行准备好数据集，本网络包括BSRN的训练数据集DIV2K和验证数据集BSD100任务

2、请参考(https://github.com/idearibosome/tf-bsrn-sr/）自行下载

3、BSRN训练的模型及数据集可以参考"简述 -> 参考实现"

    数据集组织
    ```
    ├── dataset									----数据集文件
        ├── BSD100								----验证数据集
        │   ├── LR
        │   │   ├── x2
        │   │   ├── x3
        │   │   └── x4
        │   └── SR
        └── DIV2K								----训练数据集
            ├── DIV2K_train_HR
            └── DIV2K_train_LR_bicubic
                ├── X2
                ├── X3
                └── X4
    ```

## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练。

    - 启动训练之前，首先要配置程序运行相关环境变量。

      环境变量配置信息参见：

      [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

    - 单卡训练

         网络共包含2个训练，其中train.py训练任务， 另一个是验证validate_bsrn.py数据集任务。

         **training训练DIV2K任务**:

            python3.7 ./train.py 
                --data_input_path=${data_path}${relative_path_LR}
                --data_truth_path=${data_path}${relative_path_HR} 
                --train_path=./checkpoints 
                --chip='npu' 
                --model='bsrn' 
                --dataloader='div2k_loader' 
                --batch_size=8 
                --max_steps=300000
                --save_freq=50000 
                --scales='4' 

         **validate_bsrn.py数据集BSD100任务**

            python3.7 ./validate_bsrn.py 
                --dataloader=basic_loader 
                --data_input_path=${data_path}${relative_path_LR} --data_truth_path=${data_path}${relative_path_HR} 
                --restore_path=./checkpoints/${relative_path_checkpoint}  
                --model=bsrn 
                --scales=4 
                --save_path=./result-pictures 
            


<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码

```
├── temp								----存放训练结果及数据集文件
│   ├── result								----存放训练结果（gpu训练自动生成）
│   │   ├── model.ckpt							----存放固化的模型pbtxt文件
│   │   ├── result-pictures						----存放验证数据（运行validate_gpu.sh）自动生成的超分辨率图片
│   │   │   ├── ensemble
│   │   │   │   ├── x2
│   │   │   │   ├── x3
│   │   │   │   └── x4
│   └── dataset								----数据集文件
│       ├── BSD100							----验证数据集
│       │   ├── LR
│       │   │   ├── x2
│       │   │   ├── x3
│       │   │   └── x4
│       │   └── SR
│       └── DIV2K							----训练数据集
│           ├── DIV2K_train_HR
│           └── DIV2K_train_LR_bicubic
│               ├── X2
│               ├── X3
│               └── X4
├── tf-bsrn-sr
│   ├── checkpoints							----原始代码提供的训练好的模型文件，用作精度和性能比较
│   ├── dataloaders							----数据预处理和加载脚本，可以得到batch-size大小的数据
│   ├── models								----模型网络定义，保存，恢复及优化相关脚本
│   ├── scripts								----存放模型训练和验证脚本
│    	├── run_gpu.sh							----使用gpu(v100)
│    	├── run_npu.sh							----使用npu(modelarts)
│    	├── run_npu_restore.sh					        ----从中断点恢复npu训练
│    	├── test.sh							----推理
│    	├── validate_gpu.sh						----gpu(v100)上验证模型精度
│    	└── validate_npu.sh						----npu(modelarts)上验证模型精度
│   ├── test									----新模板（训练入口）
│    	├── train_full_1p.sh					        ----gpu(v100)上验证模型精度
│    	└── train_performance_1p.sh				        ----npu(modelarts)上验证模型精度
│   ├── boot_modelarts.py（已过期，可用于旧版训练）
│   ├── help_modelarts.py（已过期，可用于旧版训练）
│   ├── modelarts_entry_acc.py					        ----训练启动文件
│   ├── modelarts_entry_perf.py					        ----性能测试启动文件
│   ├── test_bsrn.py							----测试模型
│   ├── train.py							----训练模型
│   ├── output.txt							----训练输出(gpu训练自动生成)
│   └── validate_bsrn.py						----验证模型
├── statics								----存放图片静态数据(用于md文件)
├── LICENSE
├── README.md
└── requirments.txt  							---- 依赖配置文件
```

## 脚本参数<a name="section6669162441511"></a>

```
  batch_size
  input_patch_size
  target_patch_size
  dataloader
  model
  scales
  cuda_device
  chip
  train_path
  max_steps
  log_freq
  summary_freq
  save_freq
  save_max_keep
  sleep_ratio
  restore_path
  restore_target
  global_step
  platform
```
## 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡，8卡网络训练。模型存储路径为${cur_path}/output/$ASCEND_DEVICE_ID，包括训练的log以及checkpoints文件。以8卡训练为例，loss信息在文件${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。


