# 基本信息

**发布者（Publisher）：** **Huawei**

**应用领域（Application Domain）：** **image depth estimation**

**版本（Version）：1.0**

**修改时间（Modified） ：** **2021.11.17**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：** **Demo**

**描述（Description）：基于TensorFlow框架的单目深度估计网络训练代码**

# 概述

- 模型功能：单目深度估计

- 模型架构：

​		使用在ImageNet上训练过的DenseNet-169为编码器，自定义的上采样模块为解码器的U型网络结构。

![在这里插入图片描述](https://gitee.com/DatalyOne/picGo-image/raw/master/202109121119059.png)

- 模型来源：[Github](https://github.com/ialhashim/DenseDepth)

- 参考论文：


```
@article{Alhashim2018,
  author    = {Ibraheem Alhashim and Peter Wonka},
  title     = {High Quality Monocular Depth Estimation via Transfer Learning},
  journal   = {arXiv e-prints},
  volume    = {abs/1812.11941},
  year      = {2018},
  url       = {https://arxiv.org/abs/1812.11941},
  eid       = {arXiv:1812.11941},
  eprint    = {1812.11941}
}
```

- 适配昇腾 AI 处理器的实现：

​		https://gitee.com/ascend/modelzoo/tree/master/contrib/TensorFlow/Research/cv/DENSEDEPTH_ID0806_for_TensorFlow

- 通过Git获取对应commit_id的代码方法如下：

  ```
  git clone {repository_url}    # 克隆仓库的代码
  cd {repository_name}    # 切换到模型的代码仓目录
  git checkout  {branch}    # 切换到对应分支
  git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
  cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
  ```

# 默认配置

- 训练数据集预处理（以nyu_data.zip训练集为例，仅作为用户参考示例）：
  - 图像的输入尺寸为RGB图像480\*640，深度图像480\*640
  - 图像输入格式：TFRecord
  - 随机水平翻转图像
  - 随机改变RGB图像颜色通道
  - 图像归一化
- 测试数据集预处理（以nyu_test.zip测试集为例，仅作为用户参考示例）
  - 图像的输入尺寸为RGB图像480\*640，深度图480\*640
  - 图像输入格式：npy
  - 为保证本代码能正常读取测试图片，请将RGB图像保存在eigen_test_rgb.npy，深度图像保存在eigen_test_depth.npy，裁剪因子保存在eigen_test_crop.npy，三者压缩为一个文件nyu_test.zip；或者修改相应代码。
- 部分训练超参
  - Batch size: 4
  - Learning rate(LR): 0.0001
  - Train epoch: 20

# 支持特性

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 是       |
| 混合精度   | 是       |

# 混合精度训练

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

- 开启混合精度

```
--precision-mode		算子精度模式:allow_fp32_to_fp16/force_fp16/allow_mix_precision
```

#  训练环境准备

1. 硬件环境准备请参见各硬件产品文档"[驱动和固件安装升级指南](https://support.huawei.com/enterprise/zh/category/ai-computing-platform-pid-1557196528909)"。需要在硬件设备上安装与CANN版本配套的固件与驱动。

2. 宿主机上需要安装Docker并登录[Ascend Hub中心](https://ascendhub.huawei.com/#/detail?name=ascend-tensorflow-arm)获取镜像。

   当前模型支持的镜像列表如[表1](https://gitee.com/ascend/modelzoo/blob/master/contrib/TensorFlow/Research/cv/DENSEDEPTH_ID0806_for_TensorFlow/README.md#zh-cn_topic_0000001074498056_table1519011227314)所示。

   **表 1** 镜像列表

   

   | *镜像名称*                                                   | *镜像版本* | *配套CANN版本*                                               |
   | ------------------------------------------------------------ | ---------- | ------------------------------------------------------------ |
   | *ARM架构：[ascend-tensorflow-arm](https://ascend.huawei.com/ascendhub/#/detail?name=ascend-tensorflow-arm)**x86架构：[ascend-tensorflow-x86](https://ascend.huawei.com/ascendhub/#/detail?name=ascend-tensorflow-x86)* | *20.2.0*   | *[20.2](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software)* |

# 数据集准备

1、用户自行准备好数据集，包括训练数据集和验证数据集。 本代码默认使用的训练数据集为**nyu_data.zip**，验证数据集为**nyu_tesp.zip**。获取方法见 *模型来源*

2、数据集的处理可以参考 *模型来源*

3、将数据集放置在dataset文件夹中 

4、使用nyu_data.zip生成tfrecords文件

```
python3.7.5 dataset_tool_tf.py --input_dir ./dataset/nyu_data.zip --out ./dataset/nyu_data.tfrecords
```

5、**提前下载densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5（densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5）到dataset文件夹**

# 训练脚本

**必须将数据集文件放置在与test文件夹同级的dataset文件夹中**

- 启动训练之前，首先要配置程序运行相关环境变量。

  环境变量配置信息参见：

  [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend 910训练平台环境变量设置?sort_id=3148819)

- 单卡训练

​		启动训练 （脚本为test/train_full_1p.sh）

```
bash train_full_1p.sh
```

- 8卡训练

​		启动训练 （脚本为test/train_full_8p.sh）

```
bash train_full_8p.sh
```

部分参数设置请参考下文描述

# 训练

```
python3.7.5 train.py --op_select_implmode=high_precision --precision_mode=allow_mix_precision --is_distributed=False --is_loss_scale=True --hcom_parallel=False --test_data=./dataset/nyu_test.zip --train_tfrecords=./dataset/nyu_data.tfrecords --bs=4 --lr=0.0001 --epochs=20 --steps=12672 --full 
```

参数解释：

```
--lr 				学习率 默认：0.0001
--bs 				批次大小 默认：4
--steps				每轮训练的迭代次数
--epochs 			训练轮次 默认：20
--test_data 		评估数据集路径 默认：./dataset/nyu_test.zip
--train_tfrecords 	训练数据集tfrecords路径 默认：./dataset/nyu_data.tfrecords 若tfrecords存在则使用tfrecords进行训练
--result 			输出路径 默认：./result
--full 				是否在训练过程输出精度(仅测试需保存的ckpt模型)，精度测试结果保存在相应的ckpt目录下 evaluate.log
--op_select_implmode	算子高精度模式：high_precision 高性能模型：high_performance 
--precision_mode		算子精度模式:allow_fp32_to_fp16/force_fp16/allow_mix_precision
--hcom-parallel			是否启用Allreduce梯度更新和前后向并行执行 
--is-distributed		是否为分布式训练
--is-loss-scale			是否启用LossScale
----------------------------------------------------------
--name
--mindepth
--maxdepth
```

# 测试

```
python3.7.5 test.py --ckptdir ./ckpt_npu --input './examples/*.png' --output ./result/test/test.png
```

生成图片进行展示

参数解释：

```
--ckptdir		ckpt模型位置 默认：./ckpt_npu
--input			测试图片位置： 默认：./examples/*.png
--output		输出图片位置及名称 默认：./result/test/test.png
```

# 评估

```
python3.7.5 evaluate.py --ckptdir ./ckpt_npu --test_data ./dataset/nyu_test.zip	--logdir ./ --bs 2
```

输出精度

参数解释：

```
--ckptdir 		模型路径 默认：./ckpt_npu （改为训练后生成的checkpoit文件位置）
--test_data		估数据集路径 默认：./dataset/nyu_test.zip
--logdir  		日志保存路径 evaluate.log
--bs 			评估批次 默认：2
```

# 性能

| Tesla V100-SXM2 | Ascend 910    | Batch Size | 备注 |
| ------------- | ------------- | ------------- | ------------- |
| 235 ms/step | 185 ms/step | 4 | 使用tfrecords，不开启--full |

# 精度

|                | $\delta~1$$\uparrow$ | $\delta~2$$\uparrow$ | $\delta~3$$\uparrow$ | rel$\downarrow$ | rms$\downarrow$ | $log10$$\downarrow$ |
| -------------- | :------------------: | -------------------- | -------------------- | --------------- | --------------- | --------------------- |
| GPU复现精度    |        0.8480        | 0.9721               | 0.9937               | 0.1265          | 0.5445          | 0.0536                |
| 论文精度       |        0.8460        | 0.9740               | 0.9940               | 0.1230          | 0.465           | 0.0530                |
| Ascend训练精度 |        0.8465        | 0.9699 | 0.9926 | 0.1273 | 0.5595 | 0.0539 |

注：GPU 与 Ascend 精度最大相差约2%。Ascend精度在Ascend 910上进行测试。

# 模型固化

```
python3.7.5 ckpt2pb.py --ckptdir ./ckpt_npu --pbdir ./result/pb --output_node_names conv3/BiasAdd --pb_name test
```

参数解释：

```
--ckptdir 			模型路径 默认：./ckpt_npu （改为训练后生成的checkpoit文件位置）
--pbdir     		转换后的pb模型的保存路径 默认：./result/pb
--output_node_names	输出节点 默认：conv3/BiasAdd
--pb_name			pb模型名字 默认：test
```

# pb模型推理

```
python3.7.5 infer_from_pb.py --bs 1 --model_path ./result/pb/test.pb --image_path ./dataset/nyu_test.zip 
```

参数解释：

```
--bs   				 推理批次 默认：1
--model_path		 pb模型位置 默认：./result/pb/test.pb	
--image_path		 推理数据集位置 默认：./dataset/nyu_test.zip 
---------------------------
--input_tensor_name  输入节点名称 默认：input_1:0
--output_tensor_name 输出节点名称 默认：conv3/BiasAdd:0
--minDepth			 最小深度 默认：10.0
--maxDepth			 最大深度 默认：1000.0
```

# om离线推理

详情请看**DENSEDEPTH_ID0806_for_ACL**  README.md



