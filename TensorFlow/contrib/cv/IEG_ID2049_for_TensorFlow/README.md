<h2 id="概述.md">概述</h2>

本文提出了一个高效率的噪声标签训练方法。


- 参考论文：
[Distilling Effective Supervision from Severe Label Noise](https://arxiv.org/pdf/1910.00701.pdf),
    CVPR2020
- 参考实现：
https://github.com/google-research/google-research/tree/master/ieg

- 适配昇腾 AI 处理器的实现：
https://gitee.com/lwrstudy/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/IEG_ID2049_for_TensorFlow
        


## 默认配置<a name="section91661242121611"></a>

- 训练超参

  - robe_dataset_hold_ratio:0.002
  - max_epoch:200
  - network_name:"wrn28-10"
  - dataset:"cifar10_uniform_0.2"


## 支持特性<a name="section1899153513554"></a>

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 是    |
| 混合精度  | 是    |
| 并行数据  | 是    |

## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

脚本已默认开启混合精度，设置precision_mode参数的脚本参考如下。

  ```
  custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
  custom_op.name = 'NpuOptimizer'
  custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes(str(args.precision_mode))
  ```

<h2 id="训练环境准备.md">训练环境准备</h2>

Linux version 4.18.0-193.el8.aarch64 (mockbuild@aarch64-01.mbox.centos.org) (gcc version 8.3.1 20191121 (Red Hat 8.3.1-5) (GCC))

<h2 id="快速上手.md">快速上手</h2>

- 数据集准备
1. 模型训练使用Cifar10数据集，数据集请用户自行获取。

## 模型训练<a name="section715881518135"></a>
  1.在裸机上训练，训练命令如下：
     ```
	 python3 main.py
     ```
2.在裸机上面验证，验证命令如下：
	```
	 python3 main.py --mode=evaluation
     ```
     
## 训练过程<a name="section1589455252218"></a>

1.  通过“模型训练”中的训练指令启动单卡卡训练。

2.  模型的存储路径可以由参数FLAGS.checkpoint_path设置,在main.py中39行，graph.pbtxt的存储路径在models文件夹中的basemodel.py中的第669行修改

3.  数据集路径设置在dataset_utils中的datasets.py中的350行设置，cifar10_load_data("/home/TestUser06/IEG_YYW/data/cifar-10-python/cifar-10-batches-py")，修改括号中的路径即可。
4.  四个日志路径修改，在models文件夹中的model.py中第470和471行修改的trainlog和infolog日志路径；在models文件夹中的basemodel.py中的第466行修改evallog日志路径，672行修改graphlog日志路径

## 训练精度对比
指标是acc
| gpu  | npu |
|-------|------|
|  0.962 |  0.946  |
