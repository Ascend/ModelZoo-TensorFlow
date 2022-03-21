-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [快速上手](#快速上手.md)
-   [迁移学习指导](#迁移学习指导.md)
-   [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）： Image Synthesis**

**版本（Version）：1.0**

**修改时间（Modified） ：2022.1.13**

**大小（Size）：58M**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Research**

**描述（Description）：基于TensorFlow框架的深度多模态子空间聚类网络训练代码** 

<h2 id="概述.md">概述</h2>

深度多模态子空间聚类网络”（DMSC）研究了用于多模态子空间聚类任务的各种融合方法，并提出了一种新的融合技术，称为“亲和融合”，将来自两种模态的互补信息整合到数据点之间的相似性上 跨越不同的模式。

- 参考论文：

    [Deep Multimodal Subspace Clustering Networks | Papers With Code](https://paperswithcode.com/paper/deep-multimodal-subspace-clustering-networks)

- 参考实现：

  [GitHub - mahdiabavisani/Deep-multimodal-subspace-clustering-networks: Tensorflow implementation of "Deep Multimodal Subspace Clustering Networks"](https://github.com/mahdiabavisani/Deep-multimodal-subspace-clustering-networks)

- 适配昇腾 AI 处理器的实现：
  
  [https://gitee.com/ascend/modelzoo/tree/master/built-in/TensorFlow/Research/cv/image_synthesis/Deep-multimodal-subspace-clustering-networks_ID1259_for_TensorFlow](https://gitee.com/abe_explore/modelzoo/tree/master/built-in/TensorFlow/Research/cv/image_synthesis/Deep-multimodal-subspace-clustering-networks_ID1259_for_TensorFlow)


- 通过Git获取对应commit\_id的代码方法如下：
  
    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    ```

## 默认配置<a name="section91661242121611"></a>

- 训练数据集预处理：

  - 使用代码自带数据集即可

- 训练超参

  - mat: YaleB
  - epoch: 100000
  - model: mymodel


## 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 否       |
| 混合精度   | 是       |
| 并行数据   | 否       |

## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

脚本已默认开启混合精度，设置precision_mode参数的脚本参考如下。

  ```
  ...
  session_config = tf.ConfigProto()
  custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
  custom_op.name = "NpuOptimizer"
  custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
  ...
  ```


<h2 id="训练环境准备.md">训练环境准备</h2>

1.  硬件环境准备请参见各硬件产品文档"[驱动和固件安装升级指南]( https://support.huawei.com/enterprise/zh/category/ai-computing-platform-pid-1557196528909)"。需要在硬件设备上安装与CANN版本配套的固件与驱动。


<h2 id="快速上手.md">快速上手</h2>

- 数据集准备
1. 模型训练使用Data/EYB_fc.mat，代码内自带，无需处理，也可准备其他mat格式数据集


## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。

- 启动训练之前，首先要配置程序运行相关环境变量。

  环境变量配置信息参见：

     [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

- 单卡训练 

  1. 配置训练参数。

     首先在脚本scripts/train_1p.sh中，配置训练数据集路径，请用户根据实际路径配置，数据集参数如下所示：

     ```
      --data_dir=/opt/npu/data
     ```

  2. 启动训练。

     启动单卡训练 （脚本为Deep-multimodal-subspace-clustering-networks_ID1259_for_TensorFlow/test/train_full_1p.sh） 

     ```
     bash train_full_1p.sh
     ```

<h2 id="迁移学习指导.md">迁移学习指导</h2>

- 数据集准备。

  数据集要求如下：

  1. 获取数据。

     如果要使用自己的数据集，需要将数据集放到Data目录下。参考代码中的数据集存放路径举例如下：

     - 训练集： Data/EYB_fc.mat
  
     数据集也可以放在其它目录，则修改对应的脚本入参data_dir即可。
  
-   模型训练。

    参考“模型训练”中训练步骤。

-   模型评估。
    
    参考“模型训练”中验证步骤。

<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码<a name="section08421615141513"></a>

```
├── affinity_fusion.py                        //测试代码
├── pretrain_affinity_fusion.py               //网络代码
├── metrics.py                                //测量代码
├── README.md                                 //代码说明文档
├── Data
│    ├──EYB_fc.mat                            //数据集，matlab格式
├── models                                    //模型ckpt
│    ├──EYBfc_af.ckpt.meta                    
│    ├──EYBfc_af.ckpt.index
│    ├──EYBfc_af.ckpt.data-00000-of-00001
├── test                                      // 启动脚本
│    ├──train_full_1p.sh
│    ├──train_performance_1p.sh
```

## 脚本参数<a name="section6669162441511"></a>

```
--rank_size              使用NPU卡数量，默认：1
--mode                   运行模式，默认train_and_evaluate；可选：train，evaluate，参见本小结说明
--max_train_steps        训练迭代次数，默认：150
```

## 训练过程<a name="section1589455252218"></a>

1.  通过“模型训练”中的训练指令（pretrain）启动训练。

2.  参考脚本的模型存储路径为models/，训练脚本log中包括如下信息。

```
WARNING:tensorflow:From /usr/local/Ascend/fwkplugin/python/site-packages/npu_bridge/estimator/npu/npu_optimizer.py:269: The name tf.train.Optimizer is deprecated. Please use tf.compat.v1.train.Optimizer instead.

WARNING:tensorflow:From pretrain_affinity_fusion.py:321: The name tf.reset_default_graph is deprecated. Please use tf.compat.v1.reset_default_graph instead.

WARNING:tensorflow:From pretrain_affinity_fusion.py:80: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.

WARNING:tensorflow:From pretrain_affinity_fusion.py:154: The name tf.variable_scope is deprecated. Please use tf.compat.v1.variable_scope instead.

WARNING:tensorflow:From pretrain_affinity_fusion.py:155: The name tf.get_variable is deprecated. Please use tf.compat.v1.get_variable instead.

WARNING:tensorflow:From pretrain_affinity_fusion.py:110: The name tf.train.Saver is deprecated. Please use tf.compat.v1.train.Saver instead.

WARNING:tensorflow:From pretrain_affinity_fusion.py:112: The name tf.losses.mean_squared_error is deprecated. Please use tf.compat.v1.losses.mean_squared_error instead.

WARNING:tensorflow:From /usr/local/python3.7/lib/python3.7/site-packages/tensorflow_core/python/ops/losses/losses_impl.py:121: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
WARNING:tensorflow:From pretrain_affinity_fusion.py:119: The name tf.summary.scalar is deprecated. Please use tf.compat.v1.summary.scalar instead.

WARNING:tensorflow:From pretrain_affinity_fusion.py:121: The name tf.summary.merge_all is deprecated. Please use tf.compat.v1.summary.merge_all instead.

WARNING:tensorflow:From pretrain_affinity_fusion.py:127: The name tf.train.AdamOptimizer is deprecated. Please use tf.compat.v1.train.AdamOptimizer instead.

WARNING:tensorflow:From /usr/local/Ascend/fwkplugin/python/site-packages/npu_bridge/estimator/npu/npu_loss_scale_optimizer.py:58: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead.

WARNING:tensorflow:From pretrain_affinity_fusion.py:133: The name tf.set_random_seed is deprecated. Please use tf.compat.v1.set_random_seed instead.

WARNING:tensorflow:From pretrain_affinity_fusion.py:135: The name tf.global_variables_initializer is deprecated. Please use tf.compat.v1.global_variables_initializer instead.

WARNING:tensorflow:From pretrain_affinity_fusion.py:137: The name tf.ConfigProto is deprecated. Please use tf.compat.v1.ConfigProto instead.

WARNING:tensorflow:From pretrain_affinity_fusion.py:145: The name tf.InteractiveSession is deprecated. Please use tf.compat.v1.InteractiveSession instead.

2022-01-13 14:07:49.594928: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2022-01-13 14:07:49.614469: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2700000000 Hz
2022-01-13 14:07:49.624868: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x8f56710 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2022-01-13 14:07:49.624905: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2022-01-13 14:07:49.627081: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /usr/local/Ascend/compiler/lib64:/usr/local/Ascend/compiler/lib64/plugin/opskernel:/usr/local/Ascend/compiler/lib64/plugin/nnengine:/usr/local/Ascend/runtime/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64::/usr/local/Ascend/compiler/lib64/stub
2022-01-13 14:07:49.627103: E tensorflow/stream_executor/cuda/cuda_driver.cc:318] failed call to cuInit: UNKNOWN ERROR (303)
2022-01-13 14:07:49.627123: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (bms-aiserver-118): /proc/driver/nvidia/version does not exist
2022-01-13 14:07:49.731326: W tf_adapter/util/ge_plugin.cc:124] [GePlugin] can not find Environment variable : JOB_ID
2022-01-13 14:07:53.828533: I tf_adapter/kernels/geop_npu.cc:746] The model has been compiled on the Ascend AI processor, current graph id is:1
WARNING:tensorflow:From pretrain_affinity_fusion.py:147: The name tf.summary.FileWriter is deprecated. Please use tf.compat.v1.summary.FileWriter instead.

2022-01-13 14:07:57.583340: I tf_adapter/kernels/geop_npu.cc:746] The model has been compiled on the Ascend AI processor, current graph id is:11
2022-01-13 14:07:58.411720: I tf_adapter/kernels/geop_npu.cc:746] The model has been compiled on the Ascend AI processor, current graph id is:21
2022-01-13 14:10:16.297619: I tf_adapter/kernels/geop_npu.cc:746] The model has been compiled on the Ascend AI processor, current graph id is:31
model restored
epoch: 49
cost: 0.01496168
perf: 0.1573
epoch: 99
cost: 0.00685987
perf: 0.1490
epoch: 149
cost: 0.00518302
perf: 0.1492
epoch: 199
cost: 0.00442632
perf: 0.1471
epoch: 249
cost: 0.00384156
perf: 0.1468
epoch: 299
cost: 0.00319912
perf: 0.1477
epoch: 349
cost: 0.00285121
perf: 0.1487
epoch: 399
cost: 0.00231214
perf: 0.1483
epoch: 449
cost: 0.00225497
perf: 0.1482
epoch: 499
cost: 0.00218622
perf: 0.1474
epoch: 549
...
```
