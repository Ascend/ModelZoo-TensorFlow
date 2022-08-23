<h2 id="概述.md">概述</h2>

提出了Slot Attention 模块，建立了感知表征 (perceptual representations, 如CNN 输出) 与 slots 之间的桥梁 (Feature map/Grid → Set of slots)
	

- 参考论文：

    @article{locatello2020object,
    title={Object-Centric Learning with Slot Attention},
    author={Locatello, Francesco and Weissenborn, Dirk and Unterthiner, Thomas and Mahendran, Aravindh and Heigold, Georg and Uszkoreit, Jakob and Dosovitskiy, Alexey and Kipf, Thomas},
    journal={arXiv preprint arXiv:2006.15055},
    year={2020}
}

- 参考实现：

   https://github.com/google-research/google-research/tree/master/slot_attention

- 适配昇腾 AI 处理器的实现：
    
   https://gitee.com/lwrstudy/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/Slot-Attention_ID2028_for_TensorFlow
        


## 默认配置<a name="section91661242121611"></a>

- 训练超参

  - Batch size： 64
  - Train step: 500000
  - num_slots:7
  - learning_rate:0.0004


## 支持特性<a name="section1899153513554"></a>

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 否    |
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

Linux version 4.15.0-29-generic (buildd@lgw01-amd64-057) (gcc version 7.3.0 (Ubuntu 7.3.0-16ubuntu3)) #31-Ubuntu SMP Tue Jul 17 15:39:52 UTC 2018

<h2 id="快速上手.md">快速上手</h2>

- 数据集准备
1. 模型训练使用CLEVR数据集，数据集请用户自行获取。

## 模型训练<a name="section715881518135"></a>
  1.在裸机上训练，训练命令如下：
     ```
	 python3 train.py
     ```

## 训练过程<a name="section1589455252218"></a>

1.  通过“模型训练”中的训练指令启动单卡卡训练。

2.  模型的存储路径可以由参数model_dir设置，graph.pbtxt的存储路径可以由train.py中165行的graphpath设置

3.  数据集路径设置在data.py中的83行，ds = tfds.load("clevr:3.1.0", split="train", shuffle_files=shuffle,data_dir="/home/test_user01/slot-attention/data")，修改data_dir即可。

## 训练精度对比
以下指标是loss,是训练的第319300个step的结果
| gpu  | npu |
|-------|------|
|  0.000867 |  0.000793  |
