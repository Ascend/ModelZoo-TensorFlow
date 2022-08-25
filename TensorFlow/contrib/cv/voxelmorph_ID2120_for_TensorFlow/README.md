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

**修改时间（Modified） ：2022.8.25**

**大小（Size）：8.71MB**

**框架（Framework）：TensorFlow_1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：FP32**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的voxelmorph图像配准网络训练代码**

<h2 id="概述.md">概述</h2>

## 简述<a name="section194554031510"></a>

 voxelmorph是一种基于快速学习的可变形、成对的三维医学图像配准算法。该方法将配准定义为一个参数函数，并在给定一组感兴趣的图像的情况下优化其参数。给定一对新的图像对（待配准图像，参考图像），voxelmorph可以通过使用学习的参数直接计算函数来快速计算配准场，使用CNN对该配准函数进行建模，并使用空间变换层将待配准图像配准到参考图像，同时对配准场施加平滑度约束。该方法不需要有监督的信息，如地面真实度配准场或解剖地标

- 参考论文：

  [http://arxiv.org/abs/1809.05231](VoxelMorph: A Learning Framework for Deformable Medical Image Registration)

- 参考实现：

  https://github.com/voxelmorph/voxelmorph/tree/legacy

- 适配昇腾 AI 处理器的实现：
  
  https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/voxelmorph_ID2120_for_TensorFlow

- 通过Git获取对应commit\_id的代码方法如下：
  
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    

## 默认配置<a name="section91661242121611"></a>

-   训练超参（单卡）：

      - lr：1e-4
      - epochs：50
      - lambda：0.01
      - batch_size：1
      - atlas_file
    

## 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 否       |
| 混合精度   | 否       |
| 数据并行   | 否       |


## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

拉起脚本中，传入--precision_mode='allow_mix_precision'

```
 ./train_full_1p.sh --help

parameter explain:
    --precision_mode         precision mode(allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision)
    --over_dump                  if or not over detection, default is False
    --data_dump_flag         data dump flag, default is False
    --data_dump_step             data dump step, default is 10
    --profiling                  if or not profiling for performance debug, default is False
    --data_path                  source data of training
    -h/--help                    show help message
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

1、模型训练使用ABIDE freesurfer pipeline数据集，数据集请用户自行获取

2、数据集训练前需要做预处理操作，请用户参考上文默认配置

3、数据集处理后，放入模型目录下，在训练脚本中指定数据集路径，可正常使用

4、voxelmorph训练的模型及数据集可以参考"简述 -> 参考实现"


## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练。

    - 启动训练之前，首先要配置程序运行相关环境变量。

      环境变量配置信息参见：

      [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

    - 单卡训练

          1. 配置训练参数
        
             首先在脚本train_full_1p.sh中，配置训练数据集路径和checkpoint保存路径，请用户根据实际路径配置，示例如下所示：
        
             ```
             # 路径参数初始化
             ${data_path}/Dataset-ABIDE/train/ 
             --atlas_file=${data_path}/Dataset-ABIDE/atlas_abide_brain_crop.nii.gz 
             --model_dir=${output_path} 
             --tensorboard_log_dir=${output_path} 
             --batch_size=${batch_size}
             ```
        
          2. 启动训练（脚本为./test/train_full_1p.sh） 
        
             ```
             bash train_full_1p.sh --data_path
             ```

          3. 训练精度结果

            |                                          | NPU          | GPU          | 原论文       |
            | ---------------------------------------- | ------------ | ------------ | ------------ |
            | DICE系数（[0, 1], 1 最优）/ 均值(标准差) | 0.703(0.134) | 0.708(0.133) | 0.752(0.140) |
           


<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码

```
├── LICENSE
├── README.md
├── modelzoo_level.txt
├── requirements.txt
├── ext                                  	//项目需要的外部库
│    ├── medipy-lib
│    ├── neuron
│    ├── pynd-lib
│    ├── pytools-lib
├── models                                 	//预训练权重
├── precision_too                               //华为官方的精度工具，修改了fusion_switch.cfg，关闭了UB融合
├── src					        //项目文件
│    ├── datagenerators.py            		//数据
│    ├── losses.py                       	//定义loss
│    ├── networks.py                   		//定义网络
│    ├── test_zyh.py                    	//测试代码
│    ├── train_all.py                		//训练代码
│    ├── run_1p_all.sh  					
│    ├── test.sh  							//测试入口
│    ├── loss+perf_npu_all.txt				//打印日志
│    ├── fusion_switch.json					//融合规则配置文件
│    ├── train_full_1p.sh					
│    ├── train_performance_1p.sh			//
├── test     
│    ├──train_performance_1p.sh                //训练性能入口
│    ├──train_full_1p.sh                       //训练精度入口，包含准确率评估
```

## 脚本参数<a name="section6669162441511"></a>

```
--train_data_dir
--atlas_file
--model_dir
--lr
--epochs
--lambda
--batch_size
--load_model_file
--tensorboard_log_dir
```

## 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡训练。模型存储路径为${cur_path}/output/$ASCEND_DEVICE_ID，包括训练的log以及checkpoints文件。以单卡训练为例，loss信息在文件${cur_path}/output/${ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。