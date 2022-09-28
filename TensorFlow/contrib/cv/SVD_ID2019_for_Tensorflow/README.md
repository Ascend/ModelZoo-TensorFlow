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

**大小（Size）：13.67MB**

**框架（Framework）：TensorFlow_1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的经网络中3D旋转的可行性网络训练代码**

<h2 id="概述.md">概述</h2>

## 简述<a name="section194554031510"></a>

通过 SVD 的对称正交化和密切相关,这些工具长期以来一直用于计算机视觉中的应用，例如通过正交 Procrustes、旋转平均或基本矩阵分解解决的最佳 3D 对齐问题。尽管 SVD 正交化在不同的设置中有用，但作为生成旋转矩阵的过程通常在深度学习模型中被忽略，其中偏好倾向于经典表示，如单位四元数、欧拉角和轴角，或最近引入的方法。尽管 3D 旋转在计算机视觉和机器人技术中很重要，但仍然缺少一个普遍有效的表示。这里，我们探讨了 SVD 正交化在神经网络中3D旋转的可行性。我们提出了一个理论分析，表明SVD是投影到旋转组的自然选择。我们广泛的定量分析表明，简单地用SVD正交化过程替换现有表示可以在许多涵盖监督和无监督训练的深度学习应用中获得最先进的性能

- 参考论文：

  [https://arxiv.org/abs/2006.14616](An Analysis of SVD for Deep Rotation Estimation)

- 参考实现：

  https://github.com/google-research/google-research/tree/master/special_orthogonalization

- 适配昇腾 AI 处理器的实现：
  
  https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/SVD_ID2019_for_Tensorflow

- 通过Git获取对应commit\_id的代码方法如下：
  
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    

## 默认配置<a name="section91661242121611"></a>

-   训练超参（单卡）：

    - log_step_count=200 
    - save_summaries_steps=25000 
    - train_steps=2600000 
    - save_checkpoints_steps=100000
    - eval_examples=39900
    

## 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 否       |
| 混合精度   | 是       |
| 数据并行   | 否       |


## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

模型默认开启混合精度：

```
  config = NPURunConfig(
      save_summary_steps=save_summary_steps,
      save_checkpoints_steps=save_checkpoints_steps,
      log_step_count_steps=log_step_count,
      keep_checkpoint_max=None,
      precision_mode="allow_mix_precision",
      customize_dtypes="./switch_config.txt")
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

1、模型训练使用数据集，数据集请用户自行获取（方法见https://github.com/google-research/google-research/tree/master/special_orthogonalization）

2、数据集训练前需要做预处理操作，请用户参考上文默认配置如下:

    注：生成的文件test_points_modified、points已包含在dataset文件夹中。
    ```bash
    # 将路径设置到训练点云图文件
    IN_FILES=/points_test/*.pts
    
    NEW_TEST_FILES_DIR=/test_points_modified
    
    AXANG_SAMPLING=True
    
    # 决定旋转轴角的分布
    AXANG_SAMPLING=True
    
    python -m special_orthogonalization.gen_pt_test_data_gpu --input_test_files=$IN_FILES --output_directory=$NEW_TEST_FILES_DIR --random_rotation_axang=$AXANG_SAMPLING
    ```

3、数据集处理后，放入模型目录下，在训练脚本中指定数据集路径，可正常使用

4、SVD训练的模型及数据集可以参考"简述 -> 参考实现"


## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练。

    - 启动训练之前，首先要配置程序运行相关环境变量。

      环境变量配置信息参见：

      [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)

    - 单卡训练

          1. 配置训练参数
        
             首先在脚本train_full_1p.sh中，配置训练数据集路径和checkpoint保存路径，请用户根据实际路径配置data_path,output_path，示例如下所示：
        
             ```
             # 路径参数初始化
               train_data="points/*.pts"
               test_data="test_points_modified/*.pts"

               trainData_Path="$data_path/$train_data"
               testData_Path="$data_path/$test_data"
               --method=svd \
               --checkpoint_dir=${output_path} 
               --log_step_count=200 
               --save_summaries_steps=25000 
               --pt_cloud_train_files=${trainData_Path}
               --pt_cloud_test_files=${testData_Path} 
               --train_steps=2600000 
               --save_checkpoints_steps=100000 
               --eval_examples=39900 
             ```
        
          2. 启动训练（脚本为./test/train_full_1p.sh） 
        
             ```
             bash train_full_1p.sh --data_path
             ```

          3. 训练精度结果

            |      测地线误差（°）        | 论文发布 | GPU(初始代码未改动版本) | GPU实测|NPU实测 |
            | ------------------------ | ------- | ----- | --------- |----------|
            |     平均值                |   1.63  |   2.58   |     3.98    |  2.92  |
            |     中值                 |   0.89  |    1.68   |     2.6    |  1.7  |
            |     标准差             |   6.70  |  6.93    |     9.36    |  8.45  |
           


<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码

```
├── LICENSE
├── README.md
├── modelzoo_level.txt
├── requirements.txt
├── calc_perf.py                                 	
├── gen_pt_test_data.py
├── main_point_cloud_boostPerf.py
├── modelarts_entry_Axang.py
├── modelarts_entry_acc_train.py                               	
├── modelarts_entry_perf.py  
├── modelarts_entry_stat.py                           
├── switch_config.txt				        
├── test     
│    ├──train_performance_1p.sh                //训练性能入口
│    ├──train_full_1p.sh                       //训练精度入口，包含准确率评估
```

## 脚本参数<a name="section6669162441511"></a>

```
--pt_cloud_test_files  测试数据集路径 
--pt_cloud_train_files 熟练数据集路径 
--method   指定用于预测旋转的方式。选项为"svd", "svd-inf", or "gs"。默认为“svd”
--checkpoint_dir   训练模型的存放位置
--train_steps   训练迭代的次数。默认为2600000
--save_checkpoints_steps   保存检查点的频率。默认为10000
--log_step_count   日志记录一次的步数。默认为200
--save_summaries_steps   保存一次summary的步数。默认为5000
--learning_rate   默认为1e-5
--lr_decay   如果为真，则衰减learning rate。默认为假
--lr_decay_steps   learning rate衰减步数。默认为35000
--lr_decay_rate   learning rate衰减速率。默认为0.95
--predict_all_test   如果为真，则在最新的检查点上运行eval作业，并打印每个输入的误差信息。默认为假
--eval_examples   测试样本的数量。默认为0
--print_variable_names   打印模型变量名。默认为假
--num_train_augmentations   增加每个输入点云的随机旋转数。默认为10
```

## 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡训练。模型存储路径为${cur_path}/output/$ASCEND_DEVICE_ID，包括训练的log以及checkpoints文件。以单卡训练为例，loss信息在文件${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。