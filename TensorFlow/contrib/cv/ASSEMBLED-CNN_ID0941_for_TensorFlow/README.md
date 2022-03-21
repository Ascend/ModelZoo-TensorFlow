-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [快速上手](#快速上手.md)
-   [迁移学习指导](#迁移学习指导.md)
-   [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：** Image Classification 

**版本（Version）：1.1**

**修改时间（Modified） ：2021.8.27**

**大小（Size）：285M**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：fp32**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的图像分类网络训练代码** 

<h2 id="概述.md">概述</h2>

Assembled-CNN是一个经典的图像分类网络，主要特点是通过改进并集成不同CNN中诸如Selective Kernel, Anti Alias, Big Little Networks, Residual Blocks等方案，提升模型效率。该方法取得2019年iFood图像细粒度分类大赛第一名。 

- 参考论文：

    [Lee, Jungkyu, et al. "Compounding the performance improvements of assembled techniques in a convolutional neural network." arXiv preprint arXiv:2001.06268 (2020).](https://arxiv.org/pdf/2001.06268v2.pdf) 

- [参考实现](https://github.com/clovaai/assembled-cnn)



## 默认配置<a name="section91661242121611"></a>

- 训练数据集预处理（以Food-101训练集为例）：

  - 图像的输入尺寸为224*224
  - 图像输入格式：TFRecord
  - 自动搜索数据增强方式

- 测试数据集预处理（以ImageNet2012验证集为例，仅作为用户参考示例）

  - 图像的输入尺寸为256*256（将图像最小边缩放到256，同时保持宽高比，然后在中心裁剪图像）
  - 图像输入格式：TFRecord

- 训练超参
  - Batch size: 64
  - Momentum: 0.9
  - LR scheduler: cosine
  - Learning rate(LR): 0.004
  - Optimizer: MomentumOptimizer
  - Weight decay: 0.0001
  - Label smoothing: 0.1
  - Train epoch: 200


## 支持特性<a name="section1899153513554"></a>

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 否    |
| 混合精度  | 是    |
| 并行数据  | 否    |

## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

脚本默认关闭混合精度，开启需设置precision_mode参数的脚本参考如下。

  ```
  run_config = NPURunConfig(
    profiling_config=profiling_config,
    train_distribute=distribution_strategy, session_config=session_config,
    keep_checkpoint_max=flags_obj.keep_checkpoint_max,
    save_checkpoints_steps=int(steps_per_epoch),
    enable_data_pre_proc=True,
    precision_mode='allow_mix_precision',        
    hcom_parallel=True, 
    save_checkpoints_secs=None,
  )
  ```


<h2 id="训练环境准备.md">训练环境准备</h2>

1.  硬件环境准备请参见各硬件产品文档"[驱动和固件安装升级指南]( https://support.huawei.com/enterprise/zh/category/ai-computing-platform-pid-1557196528909)"。需要在硬件设备上安装与CANN版本配套的固件与驱动。
2.  宿主机上需要安装Docker并登录[Ascend Hub中心](https://ascendhub.huawei.com/#/detail?name=ascend-tensorflow-arm)获取镜像。

    当前模型支持的镜像列表如[表1](#zh-cn_topic_0000001074498056_table1519011227314)所示。

    **表 1** 镜像列表

    <a name="zh-cn_topic_0000001074498056_table1519011227314"></a>
    <table><thead align="left"><tr id="zh-cn_topic_0000001074498056_row0190152218319"><th class="cellrowborder" valign="top" width="47.32%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001074498056_p1419132211315"><a name="zh-cn_topic_0000001074498056_p1419132211315"></a><a name="zh-cn_topic_0000001074498056_p1419132211315"></a><em id="i1522884921219"><a name="i1522884921219"></a><a name="i1522884921219"></a>镜像名称</em></p>
    </th>
    <th class="cellrowborder" valign="top" width="25.52%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001074498056_p75071327115313"><a name="zh-cn_topic_0000001074498056_p75071327115313"></a><a name="zh-cn_topic_0000001074498056_p75071327115313"></a><em id="i1522994919122"><a name="i1522994919122"></a><a name="i1522994919122"></a>镜像版本</em></p>
    </th>
    <th class="cellrowborder" valign="top" width="27.16%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001074498056_p1024411406234"><a name="zh-cn_topic_0000001074498056_p1024411406234"></a><a name="zh-cn_topic_0000001074498056_p1024411406234"></a><em id="i723012493123"><a name="i723012493123"></a><a name="i723012493123"></a>配套CANN版本</em></p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="zh-cn_topic_0000001074498056_row71915221134"><td class="cellrowborder" valign="top" width="47.32%" headers="mcps1.2.4.1.1 "><a name="zh-cn_topic_0000001074498056_ul81691515131910"></a><a name="zh-cn_topic_0000001074498056_ul81691515131910"></a><ul id="zh-cn_topic_0000001074498056_ul81691515131910"><li><em id="i82326495129"><a name="i82326495129"></a><a name="i82326495129"></a>ARM架构：<a href="https://ascend.huawei.com/ascendhub/#/detail?name=ascend-tensorflow-arm" target="_blank" rel="noopener noreferrer">ascend-tensorflow-arm</a></em></li><li><em id="i18233184918125"><a name="i18233184918125"></a><a name="i18233184918125"></a>x86架构：<a href="https://ascend.huawei.com/ascendhub/#/detail?name=ascend-tensorflow-x86" target="_blank" rel="noopener noreferrer">ascend-tensorflow-x86</a></em></li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="25.52%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001074498056_p1450714271532"><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><em id="i72359495125"><a name="i72359495125"></a><a name="i72359495125"></a>20.2.0</em></p>
    </td>
    <td class="cellrowborder" valign="top" width="27.16%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001074498056_p18244640152312"><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><em id="i162363492129"><a name="i162363492129"></a><a name="i162363492129"></a><a href="https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software" target="_blank" rel="noopener noreferrer">20.2</a></em></p>
    </td>
    </tr>
    </tbody>
    </table>


<h2 id="快速上手.md">快速上手</h2>

- 数据集准备

  1、用户自行准备好数据集，包括训练数据集和验证数据集。使用的数据集是Food-101

  2、下载预训练模型，下载路径参考"简述->开源代码"

  2、数据集的处理可以参考"简述->开源代码"处理

- 2. 下载后无需解压，其中包含训练集和验证集，已被转化为TFRecord格式
  3. 数据集处理后，放入模型目录下，在训练脚本中指定数据集路径，可正常使用

  

- 依赖安装
  ```pip install -r requirements.txt```


## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。

- 启动训练之前，首先要配置程序运行相关环境变量。

  环境变量配置信息参见：

     [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

- 单卡训练 

  1. 配置训练参数。
      训练脚本中需要更改:
      ```DATA_DIR```为数据集存储路径，如```/ssd1/TFRecord_food101```
      ```MODEL_DIR```为自定义的训练输出路径
      ```PRETRAINED_PATH```为ImageNet预训练模型存储路径

  2. 启动训练。
     ```
     cd code
     sh scripts/train_full1p.sh
     ```


- 验证。
  1. 配置验证参数。
      验证脚本中需要更改:
      ```DATA_DIR```为数据集存储路径，如```/ssd1/TFRecord_food101```
      ```MODEL_DIR```为待验证模型路径

  2. 启动验证。
      ```
      cd code
      sh scripts/eval_assemble.sh
      ```

- 训练结果
  - 结果展示
    
    |          | Top1 Accuracy |     Speed      |
    | -------- | :-----------: | :------------: |
    | 原始论文 |    92.47%     |       -        |
    | GPU      |    92.24%     |  0.58 s/step   |
| NPU      |    92.22%     | 1.53 s/step \* |
    
    \*由于训练时[Mirorpad算子只有AICPU实现](https://gitee.com/ascend/modelzoo/issues/I416PK#note_6311574)，因此在当时的Ascend平台上速度表现较慢。




<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码<a name="section08421615141513"></a>

```
│  .gitignore
│  freeze_graph.py                                // 用于固化pb模型
│  main_classification.py                         // 训练/在线推理程序入口
│  offline_infer.py                               // 用于离线推理结果比较 & 生成所需bin文件
|  requirements.txt                               // 依赖文件
│  
├─functions                                       // 模型定义与输入
│      data_config.py                             // 数据集相关配置
│      input_fns.py                               // 根据参数确定模型输入
│      model_fns.py                               // 根据参数确定模型结构
│      __init__.py
│      
├─losses                                          // 训练损失函数 
│      cls_losses.py
│      __init__.py
│      
├─metric                                          // 定义评价指标
│      ece_metric.py
│      recall_metric.py
│      __init__.py
│      
├─nets                                            // 具体网络结构定义与训练循环
│      blocks.py                                  // 网络block设置
│      hparams_config.py                          // 超参设置
│      model_helper.py                            // 部分网络组件
│      optimizer_setting.py                       // 优化器设置
│      resnet_model.py                            // 具体网络结构
│      run_loop_classification.py                 // 定义网络训练与验证过程
│      __init__.py
│      
├─official                                        // 为方便程序实现额外改编的库
│  │  __init__.py
│  │  
│  └─utils
│      │  __init__.py
│      │  
│      ├─accelerator
│      │      tpu.py
│      │      tpu_test.py
│      │      __init__.py
│      │      
│      ├─data
│      │      file_io.py
│      │      file_io_test.py
│      │      __init__.py
│      │      
│      ├─export
│      │      export.py
│      │      export_test.py
│      │      __init__.py
│      │      
│      ├─flags
│      │      core.py
│      │      flags_test.py
│      │      guidelines.md
│      │      README.md
│      │      _base.py
│      │      _benchmark.py
│      │      _conventions.py
│      │      _device.py
│      │      _misc.py
│      │      _performance.py
│      │      __init__.py
│      │      
│      ├─logs
│      │      cloud_lib.py
│      │      cloud_lib_test.py
│      │      guidelines.md
│      │      hooks.py
│      │      hooks_helper.py
│      │      hooks_helper_test.py
│      │      hooks_test.py
│      │      logger.py
│      │      logger_test.py
│      │      metric_hook.py
│      │      metric_hook_test.py
│      │      __init__.py
│      │      
│      ├─misc
│      │      distribution_utils.py
│      │      distribution_utils_test.py
│      │      model_helpers.py
│      │      model_helpers_test.py
│      │      __init__.py
│      │      
│      └─testing
│          │  integration.py
│          │  mock_lib.py
│          │  pylint.rcfile
│          │  reference_data.py
│          │  reference_data_test.py
│          │  __init__.py
│          │  
│          └─scripts
│                  presubmit.sh
│                  
├─preprocessing                                 // 数据预处理、数据增强相关代码
│      autoaugment.py
│      imagenet_preprocessing.py
│      inception_preprocessing.py
│      reid_preprocessing.py
│      __init__.py
│      
├─scripts                                      // 训练、推理启动脚本
│      eval_assemble.sh                        // 在线验证
│      train_full1p.sh                         // Food-101上训练
│      train_assemble_from_scratch.sh          // ImageNet上，从头开始训练
│      
└─utils                                       // 为方便实现而整合的组件
        checkpoint_utils.py                   // checkpoint相关组件
        config_utils.py                       // 构建Estimator时的相关组件
        data_util.py                          // 数据读取相关组件
        hook_utils.py                         // 训练过程需要的hook
        log_utils.py                          // log相关组件
        __init__.py
```

## 脚本参数<a name="section6669162441511"></a>

```
--dataset_name                            数据集名称
--data_dir                                数据集所在路径
--model_dir                               训练输出路径
--pretrained_model_checkpoint_path        预训练模型路径
--resnet_version                          网络类型
--resnet_size                             网络深度
--anti_alias_filter_size                  抗锯齿滤波器大小
--anti_alias_type                         抗锯齿类型
--mixup_type                              采取mixup的方式
--autoaugment_type                        采取自动数据增强的方式
--label_smoothing                         标签平滑的比例
--use_sk_block                            开启Selective Kernel
--use_dropblock                           开启DropBlock
--dropblock_kp                            DropBlock初始与结束所对应的drop rate
--preprocessing_type                      预处理类型
--base_learning_rate                      初始学习率
--batch_size                              所输入每批样本的数目
--learning_rate_decay_type=cosine         学习率变化策略
--lr_warmup_epochs                        warm up训练次数
--train_epochs                            总共训练次数
--bn_momentum                             批归一化动量值
--weight_decay                            权重衰减值
--keep_checkpoint_max                     保存检查点的最大数目
--ratio_fine_eval                         训练次数与验证次数的比值
--epochs_between_evals                    训练过程中验证的间隔
--clean                                   训练前清理输出文件夹
```


## 训练过程<a name="section1589455252218"></a>

```
2021-08-25 12:56:50.888 I: global_step...220032
2021-08-25 12:56:52.418 I: global_step...220033
2021-08-25 12:56:53.947 I: global_step...220034
2021-08-25 12:56:55.485 I: global_step...220035
2021-08-25 12:56:57.021 I: global_step...220036
2021-08-25 12:56:58.557 I: global_step...220037
2021-08-25 12:57:00.098 I: global_step...220038
2021-08-25 12:57:00.099 I: Saving checkpoints for 220038 into /home/sunshk/assemble-multi/2_output_256a_b64_lr4e-3_wd1e-4_200epochs/model.ckpt.
2021-08-25 12:57:00.215020: W tf_adapter/util/infershape_util.cc:313] The InferenceContext of node _SOURCE is null.
2021-08-25 12:57:00.215140: W tf_adapter/util/infershape_util.cc:313] The InferenceContext of node _SINK is null.
2021-08-25 12:57:59.763804: W tf_adapter/kernels/geop_npu.cc:758] [GEOP] Out of range: End of sequence.
2021-08-25 12:57:59.764 I: NPUCheckpointSaverHook end...
2021-08-25 12:58:01.769 I: Stop log output thread controller
2021-08-25 12:58:01.769 I: Shutting down LogOutfeedController thread.
2021-08-25 12:58:01.773 I: Log outfeed thread finished
2021-08-25 12:58:11.792 I: Loss for final step: 2.331186.
2021-08-25 12:58:11.795 I: Starting to evaluate.
2021-08-25 12:58:11.839 I: The # of Supervised tfrecords: 16
2021-08-25 12:58:11.995 I: dataset = dataset.map(flatten_input)
2021-08-25 12:58:11.995 I: <DatasetV1Adapter shapes: ({image: (64, 256, 256, 3)}, (64,)), types: ({image: tf.float32}, tf.int32)>
2021-08-25 12:58:12.010 I: Calling model_fn.
2021-08-25 12:58:12.023 I: [loss type] softmax
2021-08-25 12:58:12.090 I: blModule0 On
2021-08-25 12:58:12.348 I: blModule1 On
2021-08-25 12:58:12.619 I: Anti-Alias stridedConv2 On
2021-08-25 12:58:13.635 I: Anti-Alias stridedConv2 On
2021-08-25 12:58:13.701 I: blModule2 On
2021-08-25 12:58:13.981 I: Anti-Alias stridedConv2 On
2021-08-25 12:58:15.263 I: Anti-Alias stridedConv2 On
2021-08-25 12:58:15.329 I: blModule3 On
2021-08-25 12:58:15.615 I: Anti-Alias stridedConv2 On
2021-08-25 12:58:17.926 I: Anti-Alias stridedConv2 On
2021-08-25 12:58:20.779 I: [Added final dense layer]
2021-08-25 12:58:20.813 I: [Classfication loss type; softmax]
2021-08-25 12:58:21.162 I: Done calling model_fn.
2021-08-25 12:58:21.195 I: Starting evaluation at 2021-08-25T12:58:21Z
2021-08-25 12:58:23.153 I: Graph was finalized.
2021-08-25 12:58:23.160 I: Restoring parameters from /home/sunshk/assemble-multi/2_output_256a_b64_lr4e-3_wd1e-4_200epochs/model.ckpt-220038
2021-08-25 12:58:26.919133: W tf_adapter/util/infershape_util.cc:313] The InferenceContext of node _SOURCE is null.
2021-08-25 12:58:26.919219: W tf_adapter/util/infershape_util.cc:313] The InferenceContext of node _SINK is null.
2021-08-25 12:58:26.927587: W tf_adapter/util/infershape_util.cc:313] The InferenceContext of node save/restore_all is null.
2021-08-25 12:58:36.541796: W tf_adapter/util/infershape_util.cc:313] The InferenceContext of node _SOURCE is null.
2021-08-25 12:58:36.541894: W tf_adapter/util/infershape_util.cc:313] The InferenceContext of node _SINK is null.
2021-08-25 12:58:36.547233: W tf_adapter/util/infershape_util.cc:337] The shape of node report_uninitialized_variables_1/boolean_mask/Where output 0 is [?,1], unknown shape.
2021-08-25 12:58:36.547281: W tf_adapter/util/infershape_util.cc:337] The shape of node report_uninitialized_variables_1/boolean_mask/Squeeze output 0 is [?], unknown shape.
2021-08-25 12:58:36.577 I: Running local_init_op.
2021-08-25 12:58:36.757188: W tf_adapter/util/infershape_util.cc:313] The InferenceContext of node _SOURCE is null.
2021-08-25 12:58:36.757274: W tf_adapter/util/infershape_util.cc:313] The InferenceContext of node _SINK is null.
2021-08-25 12:58:36.757502: W tf_adapter/util/infershape_util.cc:313] The InferenceContext of node group_deps_2 is null.
2021-08-25 12:58:37.006 I: Done running local_init_op.
2021-08-25 12:58:37.636465: W tf_adapter/util/infershape_util.cc:313] The InferenceContext of node _SOURCE is null.
2021-08-25 12:58:37.636545: W tf_adapter/util/infershape_util.cc:313] The InferenceContext of node _SINK is null.
2021-08-25 12:58:37.641941: W tf_adapter/util/infershape_util.cc:337] The shape of node report_uninitialized_variables/boolean_mask/Where output 0 is [?,1], unknown shape.
2021-08-25 12:58:37.641988: W tf_adapter/util/infershape_util.cc:337] The shape of node report_uninitialized_variables/boolean_mask/Squeeze output 0 is [?], unknown shape.
2021-08-25 12:58:37.952189: W tf_adapter/util/infershape_util.cc:313] The InferenceContext of node _SOURCE is null.
2021-08-25 12:58:37.952267: W tf_adapter/util/infershape_util.cc:313] The InferenceContext of node _SINK is null.
2021-08-25 12:58:38.010 I: Starting log outfeed thread controller.
2021-08-25 12:58:38.011 I: Starting log outfeed thread coordinate.
2021-08-25 12:58:38.012 I: Add log output coordinate thread to coord
2021-08-25 12:58:40.267291: W tf_adapter/util/infershape_util.cc:313] The InferenceContext of node _SOURCE is null.
2021-08-25 12:58:40.267381: W tf_adapter/util/infershape_util.cc:313] The InferenceContext of node _SINK is null.
2021-08-25 12:58:40.274197: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage0_1/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.274322: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage0/pool/batch_normalization_1/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.274359: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage0/pool/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.274422: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage0/pool/batch_normalization_2/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.274465: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage0/pool/batch_normalization_3/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.274513: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage0/pool/batch_normalization_4/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.274588: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage1/big1/batch_normalization_1/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.274615: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage1/little1/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.274641: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage1/little1/batch_normalization_1/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.274688: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage1/big1/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.274734: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage1/big1/batch_normalization_2/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.274761: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage1/little1/batch_normalization_2/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.274885: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage1/big1/sk_block/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.274913: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage1/little1/sk_block/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.275103: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage1/little1/batch_normalization_3/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.275147: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage1/big1/batch_normalization_3/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.275203: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage1/little1/batch_normalization_4/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.275241: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage1/big1/batch_normalization_4/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.275283: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage1/big1/batch_normalization_5/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.275355: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage1/big1/sk_block_1/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.275460: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage1/big1/batch_normalization_6/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.275537: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage1/merge1/batch_normalization_1/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.275581: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage1/merge1/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.275621: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage1/merge1/batch_normalization_2/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.275695: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage1/merge1/sk_block/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.275819: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage1/merge1/batch_normalization_3/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.275904: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage2/big2/batch_normalization_1/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.275933: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage2/little2/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.275969: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage2/little2/batch_normalization_1/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.276018: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage2/big2/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.276065: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage2/big2/batch_normalization_2/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.276092: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage2/little2/batch_normalization_2/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.276212: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage2/big2/sk_block/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.276241: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage2/little2/sk_block/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.276424: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage2/little2/batch_normalization_3/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.276469: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage2/big2/batch_normalization_3/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.276523: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage2/little2/batch_normalization_4/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.276561: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage2/big2/batch_normalization_4/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.276604: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage2/big2/batch_normalization_5/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.276677: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage2/big2/sk_block_1/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.276778: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage2/big2/batch_normalization_6/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.276827: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage2/big2/batch_normalization_7/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.276870: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage2/big2/batch_normalization_8/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.276943: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage2/big2/sk_block_2/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.277043: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage2/big2/batch_normalization_9/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.277119: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage2/merge2/batch_normalization_1/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.277163: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage2/merge2/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.277205: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage2/merge2/batch_normalization_2/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.277278: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage2/merge2/sk_block/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.277394: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage2/merge2/batch_normalization_3/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.277466: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/little3/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.277494: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/little3/batch_normalization_1/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.277528: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/big3/batch_normalization_1/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.277584: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/big3/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.277623: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/little3/batch_normalization_2/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.277648: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/big3/batch_normalization_2/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.277763: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/little3/sk_block/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.277790: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/big3/sk_block/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.277962: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/little3/batch_normalization_3/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.278022: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/big3/batch_normalization_3/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.278067: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/little3/batch_normalization_4/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.278109: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/big3/batch_normalization_4/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.278152: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/big3/batch_normalization_5/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.278223: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/big3/sk_block_1/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.278326: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/big3/batch_normalization_6/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.278373: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/big3/batch_normalization_7/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.278414: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/big3/batch_normalization_8/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.278486: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/big3/sk_block_2/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.278585: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/big3/batch_normalization_9/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.278638: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/big3/batch_normalization_10/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.278681: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/big3/batch_normalization_11/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.278753: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/big3/sk_block_3/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.278851: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/big3/batch_normalization_12/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.278900: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/big3/batch_normalization_13/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.278940: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/big3/batch_normalization_14/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.279015: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/big3/sk_block_4/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.279116: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/big3/batch_normalization_15/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.279185: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/merge3/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.279214: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/merge3/batch_normalization_1/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.279254: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/merge3/batch_normalization_2/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.279327: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/merge3/sk_block/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.279427: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/merge3/batch_normalization_3/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.279494: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage4/batch_normalization_1/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.279537: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage4/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.279572: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage4/batch_normalization_2/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.279646: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage4/sk_block/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.279763: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage4/batch_normalization_3/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.279811: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage4/batch_normalization_4/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.279853: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage4/batch_normalization_5/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.279924: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage4/sk_block_1/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.280036: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage4/batch_normalization_6/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.280086: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage4/batch_normalization_7/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.280135: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage4/batch_normalization_8/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.280205: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage4/sk_block_2/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.280303: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage4/batch_normalization_9/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-25 12:58:40.280573: W tf_adapter/util/infershape_util.cc:313] The InferenceContext of node group_deps is null.
2021-08-25 13:07:34.108715: W tf_adapter/kernels/geop_npu.cc:758] [GEOP] Out of range: End of sequence.
2021-08-25 13:07:34.316 I: Stop log output thread controller
2021-08-25 13:07:34.317 I: Shutting down LogOutfeedController thread.
2021-08-25 13:07:34.321 I: Log outfeed thread finished
2021-08-25 13:07:34.502574: W tf_adapter/util/infershape_util.cc:313] The InferenceContext of node _SOURCE is null.
2021-08-25 13:07:34.502676: W tf_adapter/util/infershape_util.cc:313] The InferenceContext of node _SINK is null.
2021-08-25 13:07:41.186 I: Finished evaluation at 2021-08-25-13:07:41
2021-08-25 13:07:41.187 I: Saving dict for global step 220038: accuracy = 0.9221526, accuracy_top_5 = 0.9860803, ece = 0.047184695, global_step = 220038, loss = 1.380606
2021-08-25 13:07:41.192 I: Saving 'checkpoint_path' summary for global step 220038: /home/sunshk/assemble-multi/2_output_256a_b64_lr4e-3_wd1e-4_200epochs/model.ckpt-220038
2021-08-25 13:07:49.145 I: Benchmark metric: {'name': 'accuracy', 'value': 0.922152578830719, 'unit': None, 'global_step': 220038, 'timestamp': '2021-08-25T17:07:49.145381Z', 'extras': []}
2021-08-25 13:07:49.145 I: Benchmark metric: {'name': 'accuracy_top_5', 'value': 0.9860802888870239, 'unit': None, 'global_step': 220038, 'timestamp': '2021-08-25T17:07:49.145852Z', 'extras': []}
2021-08-25 13:07:49.145 I: Benchmark metric: {'name': 'ece', 'value': 0.047184694558382034, 'unit': None, 'global_step': 220038, 'timestamp': '2021-08-25T17:07:49.145961Z', 'extras': []}
2021-08-25 13:07:49.146 I: Benchmark metric: {'name': 'loss', 'value': 1.3806060552597046, 'unit': None, 'global_step': 220038, 'timestamp': '2021-08-25T17:07:49.146054Z', 'extras': []}
```

## 推理/验证过程<a name="section1465595372416"></a>

```
2021-08-26 02:42:38.481343: W tf_adapter/util/infershape_util.cc:313] The InferenceContext of node _SINK is null.
2021-08-26 02:42:38.489807: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage0_1/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.489934: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage0/pool/batch_normalization_1/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.489984: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage0/pool/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.490043: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage0/pool/batch_normalization_2/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.490099: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage0/pool/batch_normalization_3/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.490164: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage0/pool/batch_normalization_4/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.490258: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage1/big1/batch_normalization_1/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.490297: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage1/little1/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.490327: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage1/little1/batch_normalization_1/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.490388: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage1/big1/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.490445: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage1/big1/batch_normalization_2/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.490476: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage1/little1/batch_normalization_2/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.490622: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage1/big1/sk_block/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.490662: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage1/little1/sk_block/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.490889: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage1/little1/batch_normalization_3/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.490948: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage1/big1/batch_normalization_3/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.491019: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage1/little1/batch_normalization_4/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.491069: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage1/big1/batch_normalization_4/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.491115: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage1/big1/batch_normalization_5/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.491212: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage1/big1/sk_block_1/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.491339: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage1/big1/batch_normalization_6/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.491434: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage1/merge1/batch_normalization_1/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.491489: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage1/merge1/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.491537: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage1/merge1/batch_normalization_2/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.491623: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage1/merge1/sk_block/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.491770: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage1/merge1/batch_normalization_3/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.491887: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage2/big2/batch_normalization_1/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.491919: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage2/little2/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.491956: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage2/little2/batch_normalization_1/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.492018: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage2/big2/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.492076: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage2/big2/batch_normalization_2/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.492113: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage2/little2/batch_normalization_2/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.492256: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage2/big2/sk_block/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.492288: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage2/little2/sk_block/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.492518: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage2/little2/batch_normalization_3/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.492576: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage2/big2/batch_normalization_3/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.492643: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage2/little2/batch_normalization_4/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.492694: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage2/big2/batch_normalization_4/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.492742: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage2/big2/batch_normalization_5/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.492844: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage2/big2/sk_block_1/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.492974: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage2/big2/batch_normalization_6/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.493037: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage2/big2/batch_normalization_7/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.493083: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage2/big2/batch_normalization_8/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.493178: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage2/big2/sk_block_2/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.493306: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage2/big2/batch_normalization_9/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.493398: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage2/merge2/batch_normalization_1/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.493453: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage2/merge2/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.493501: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage2/merge2/batch_normalization_2/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.493589: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage2/merge2/sk_block/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.493741: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage2/merge2/batch_normalization_3/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.493833: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/little3/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.493865: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/little3/batch_normalization_1/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.493918: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/big3/batch_normalization_1/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.493989: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/big3/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.494030: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/little3/batch_normalization_2/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.494066: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/big3/batch_normalization_2/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.494210: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/little3/sk_block/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.494250: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/big3/sk_block/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.494464: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/little3/batch_normalization_3/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.494539: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/big3/batch_normalization_3/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.494593: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/little3/batch_normalization_4/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.494648: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/big3/batch_normalization_4/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.494702: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/big3/batch_normalization_5/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.494791: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/big3/sk_block_1/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.494924: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/big3/batch_normalization_6/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.494988: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/big3/batch_normalization_7/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.495042: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/big3/batch_normalization_8/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.495131: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/big3/sk_block_2/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.495262: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/big3/batch_normalization_9/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.495326: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/big3/batch_normalization_10/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.495381: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/big3/batch_normalization_11/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.495471: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/big3/sk_block_3/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.495598: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/big3/batch_normalization_12/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.495660: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/big3/batch_normalization_13/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.495716: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/big3/batch_normalization_14/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.495803: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/big3/sk_block_4/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.495932: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/big3/batch_normalization_15/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.496042: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/merge3/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.496075: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/merge3/batch_normalization_1/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.496128: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/merge3/batch_normalization_2/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.496215: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/merge3/sk_block/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.496341: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage3/merge3/batch_normalization_3/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.496420: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage4/batch_normalization_1/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.496475: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage4/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.496524: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage4/batch_normalization_2/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.496617: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage4/sk_block/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.496771: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage4/batch_normalization_3/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.496827: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage4/batch_normalization_4/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.496881: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage4/batch_normalization_5/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.496979: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage4/sk_block_1/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.497104: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage4/batch_normalization_6/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.497165: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage4/batch_normalization_7/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.497212: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage4/batch_normalization_8/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.497306: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage4/sk_block_2/batch_normalization/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.497433: W tf_adapter/util/infershape_util.cc:337] The shape of node resnet_model/stage4/batch_normalization_9/FusedBatchNormV3 output 5 is ?, unknown shape.
2021-08-26 02:42:38.497768: W tf_adapter/util/infershape_util.cc:313] The InferenceContext of node group_deps is null.
2021-08-26 02:51:35.020087: W tf_adapter/kernels/geop_npu.cc:758] [GEOP] Out of range: End of sequence.
2021-08-26 02:51:35.271 I: Stop log output thread controller
2021-08-26 02:51:35.271 I: Shutting down LogOutfeedController thread.
2021-08-26 02:51:35.282 I: Log outfeed thread finished
2021-08-26 02:51:35.452219: W tf_adapter/util/infershape_util.cc:313] The InferenceContext of node _SOURCE is null.
2021-08-26 02:51:35.452299: W tf_adapter/util/infershape_util.cc:313] The InferenceContext of node _SINK is null.
2021-08-26 02:51:41.111 I: Finished evaluation at 2021-08-26-02:51:41
2021-08-26 02:51:41.112 I: Saving dict for global step 220038: accuracy = 0.9220532, accuracy_top_5 = 0.98609793, ece = 0.047271796, global_step = 220038, loss = 0.49775454
2021-08-26 02:51:55.599 I: Saving 'checkpoint_path' summary for global step 220038: /home/sunshk/food_ckpt/model.ckpt-220038
```