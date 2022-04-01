-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [快速上手](#快速上手.md)

[comment]: <> (-   [迁移学习指导]&#40;#迁移学习指导.md&#41;)

[comment]: <> (-   [高级参考]&#40;#高级参考.md&#41;)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Attribute-Object Compositions**

**版本（Version）：1.1**

**修改时间（Modified） ：2022.03.28**

**大小（Size）：74M**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的Symnet网络训练代码** 

<h2 id="概述.md">概述</h2>

Symnet结合属性-对象转换的对称性原理和群论公理，由耦合网络和解耦网络两个模块组成，提出了基于Relative Moving Distance(RMD)的识别方法，利用属性的变化而非属性本身去分类属性。在Attribute-Object Composition零样本学习任务上取得了重大改进。

- 参考论文：

    [Li, Yong-Lu, et al. "Symmetry and group in attribute-object compositions." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.](https://arxiv.org/abs/2004.00587) 

- 参考实现：

    [SymNet](https://github.com/DirtyHarryLYL/SymNet)

- 适配昇腾 AI 处理器的实现：
  
  [https://gitee.com/ascend/modelzoo/tree/master/built-in/TensorFlow/Official/cv/image_classification/DenseNet121_ID0067_for_TensorFlow](https://gitee.com/ascend/modelzoo/tree/master/built-in/TensorFlow/Official/cv/image_classification/DenseNet121_ID0067_for_TensorFlow)      


## 默认配置<a name="section91661242121611"></a>

- 数据集预处理：
  - 在本地运行`download_data.sh`下载数据集并进行预处理
- 训练超参
  - Batch size: 256
  - obj_pred: UT_obj_lr1e-3_test_ep260.pkl
  - data: UT
  - batchnorm: True
  - wordvec: onehot
  - lr: 1e-4
  - lambda_cls_attr: 1
  - lambda_cls_obj: 0.5
  - lambda_trip: 0.5
  - lambda_sym: 0.01
  - lambda_axiom: 0.03
  - Train epoch: 700 
    
## 支持特性<a name="section1899153513554"></a>

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 是    |
| 混合精度  | 是    |
| 并行数据  | 否    |

<h2 id="训练环境准备.md">训练环境准备</h2>
1. NPU环境&硬件环境：

```
NPU: 1*Ascend 910   
CPU: 24*vCPUs 96GB 
Image: ascend-share/5.1.rc1.alpha003_tensorflow-ascend910-cp37-euleros2.8-aarch64-training:1.15.0-21.0.2_0317
```

2. 第三方依赖：

```
pip-requirements.txt
```
 

<h2 id="快速上手.md">快速上手</h2>

### 数据集准备

用户在本地运行`download_data.sh`下载数据集并进行预处理，预处理后的数据集已上传至obs中。obs路径：obs://cann-id1292-symnet/data/。


### 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。

- 启动训练之前，首先要配置程序运行相关环境变量。环境变量配置信息参见：[Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

- 单卡训练 

1. 配置训练参数。 
   
首先在脚本test/train_full_1p.sh中，配置训练数据集路径及参数，请用户根据实际路径配置，参数如下所示：
```
--data_dir=/opt/npu/slimImagenet
--name UT_best 
--data UT 
--epoch=${epochs} 
--obj_pred UT_obj_lr1e-3_test_ep260.pkl 
--batchnorm  
--wordvec onehot  
--lr 1e-4 
--bz=${batch_size} 
--lambda_cls_attr 1 
--lambda_cls_obj 0.5 
--lambda_trip 0.5 
--lambda_sym 0.01 
--lambda_axiom 0.03 
--data_url=${data_path} 
--train_url=${output_path}
```
2. 启动训练。
   
启动单卡训练 （脚本为SYMNET_ID1292_for_TensorFlow/test/train_full_1p.sh） 

     ```
     bash ./test/train_full_1p.sh --data_path=%s --output_path=%s
     ```
### 验证
在train时已经执行模型验证测试，如果需要单独测试指定模型（指定epoch），按如下步骤操作。

1. 测试的时候，需要修改脚本启动参数（SYMNET_ID1292_for_TensorFlow/test/train_full_1p.sh），修改train_full_1p.sh运行test_symnet.py即可测试指定模型，请用户根据实际路径进行修改。

```
python3.7 ./test_symnet.py --name UT_best --data UT --epoch=${epochs} --obj_pred UT_obj_lr1e-3_test_ep260.pkl --wordvec onehot --batchnorm --data_url=${data_path} --train_url=${output_path}
```
2. 测试指令（脚本位于SYMNET_ID1292_for_TensorFlow/test/train_full_1p.sh）
```
bash ./test/train_full_1p.sh --data_path=%s --output_path=%s
```

## 训练过程<a name="section1589455252218"></a>

1. 执行`modelarts_entry_acc.py`拉起训练。
2. 保存checkpoint文件。
3. GPU复现和NPU复现精度如下

|   | 数据集 | EPOCH| 精度 |
|-------|------|------|------|
| 原文 | UT | <700 | T1:52.1 &nbsp; T2:67.8 &nbsp; T3:76.0 |
| GPU  | UT | 574 | T1:0.5116 &nbsp; T2:0.6719 &nbsp; T3:0.7616 |
| NPU | UT | 636 | T1:0.5007 &nbsp; T2:0.6684 &nbsp; T3:0.7571 |

	
[comment]: <> (## 推理/验证过程<a name="section1465595372416"></a>)

[comment]: <> (```)

[comment]: <> (待补充)

[comment]: <> (```)