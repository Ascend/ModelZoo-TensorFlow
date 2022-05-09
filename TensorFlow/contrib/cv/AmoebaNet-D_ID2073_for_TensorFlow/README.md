-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [快速上手](#快速上手.md)
-   [迁移学习指导](#迁移学习指导.md)
-   [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）： Image Classification**

**修改时间（Modified） ：2022.5.9**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的AmoebaNet-D图像分类网络训练代码** 

<h2 id="概述.md">概述</h2>

AmoebaNet-D是由AmoebaNet演化神经架构搜索算法搜索出的一个图像分类神经网络。

- 参考论文：

    [Real, E., Aggarwal, A., Huang, Y., & Le, Q. V. (2019, July). Regularized evolution for image classifier architecture search. In Proceedings of the aaai conference on artificial intelligence (Vol. 33, No. 01, pp. 4780-4789).](https://arxiv.org/pdf/1802.01548.pdf) 


- 参考实现：

    

- 适配昇腾 AI 处理器的实现：
  
  [https://gitee.com/zero167/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/AmoebaNet-D_ID2073_for_TensorFlow](https://gitee.com/zero167/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/AmoebaNet-D_ID2073_for_TensorFlow)      



## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

脚本已默认开启混合精度，设置precision_mode参数的脚本参考如下。

  ```
  
    npu_config = NPURunConfig(
        model_dir=FLAGS.model_dir,
        save_checkpoints_steps=save_checkpoints_steps,
        save_summary_steps=0,
        # dump_config=dump_config,
        # fusion_switch_file="/home/test_user03/tpu-master/models/official/amoeba_net/fusion_switch.cfg",
        session_config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False),
        #precision_mode="allow_mix_precision")
        precision_mode="allow_fp32_to_fp16")
	#precision_mode="force_fp32")
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
1. 模型训练使用ImageNet2012数据集，数据集请用户自行获取。

2. 数据集训练前需要做预处理操作，请用户参考[Tensorflow-Slim](https://github.com/tensorflow/models/tree/master/research/slim),将数据集封装为tfrecord格式。

3. 数据集处理后，放入模型目录下，在训练脚本中指定数据集路径，可正常使用。
   

## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。

- 启动训练之前，首先要配置程序运行相关环境变量。

  环境变量配置信息参见：

     [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

- 单卡训练 

  1. 配置训练参数。

     首先在脚本train_performance_1p.sh中，配置训练数据集路径和checkpoint保存路径，请用户根据实际路径配置，示例如下所示：

     ```
        DATA_DIR=/home/test_user03/tf_records/
        MODEL_DIR=/home/test_user03/hh
      
     ```

  2. 启动训练。（脚本为train_performance_1p.sh） 

     ```
     bash train_performance_1p.sh
     ```


- 验证。

  1. 配置验证参数。

     首先在脚本train_full_1p.sh中，配置训练数据集路径和checkpoint保存路径，请用户根据实际路径配置，示例如下所示：

     ```
        DATA_DIR=/home/test_user03/tf_records/
        MODEL_DIR=/home/test_user03/hh
      
     ```

  2. 启动验证。（脚本为train_full_1p.sh） 

     ```
     bash train_full_1p.sh
     ```          





<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码<a name="section08421615141513"></a>

```
├── amoeba_net.py                          //训练与测试入口
├── README.md                               //代码说明文档
├── amoeba_net_model.py                    //模型功能
├── model_builder.py                       //根据用户传入的参数构建模型
├── model_specs.py                         //Amoeba_net架构配置
├── network_utils.py                       //Amoeba-net使用的常见操作的自定义模块
├── network_utils_test.py                  //对network_utils自定义模块的测试
├── tf_hub.py                               //模型导出和评估
├── inception_preprocessing.py            //图像预处理
├── common
│    ├──imagenet.py                         //为ImageNet ILSVRC 2012数据集提供数据帮助程序
│    ├──inference_warmup.py                //inference warmup实现```

```
## 脚本参数<a name="section6669162441511"></a>
```
--use_tpu              是否使用tpu，默认：False（由于该代码从tpu版本迁移过来，在晟腾910上只能是False）
--mode                 运行模式，默认train_and_eval；可选：train，eval
--data_dir             数据集目录
--mmodel_dir           保存模型输出的目录
--num_cells             网络结构中cell的数量，默认：6
--image_size            图像尺寸，默认：224
--num_epochs           训练迭代次数，默认：35
--train_batch_size     训练的batch size，默认：64
--eval_batch_size      验证的batch size， 默认：64    
--lr                     初始学习率，默认：2.56
--lr_decay_value        学习率指数衰减 默认：0.88
--lr_warmup_epochs      初始学习率从0增长到指定学习率的迭代数，默认：0.35
```


## 训练过程<a name="section1589455252218"></a>

1.  通过“模型训练”中的训练指令启动单卡训练。

2.  训练脚本log中包括如下信息。

```
I0420 23:35:31.718360 281472996956240 basic_session_run_hooks.py:692] global_step/sec: 4.13978
INFO:tensorflow:global_step...700647
I0420 23:35:31.722282 281472996956240 npu_hook.py:132] global_step...700647
INFO:tensorflow:global_step...700648
I0420 23:35:31.963601 281472996956240 npu_hook.py:132] global_step...700648
...
INFO:tensorflow:Saving checkpoints for 700662 into /home/test_user03/ckpt5/model.ckpt.
I0420 23:35:35.366074 281472996956240 basic_session_run_hooks.py:606] Saving checkpoints for 700662 into /home/test_user03/ckpt5/model.ckpt.
INFO:tensorflow:global_step...700663
I0420 23:36:39.784266 281472996956240 npu_hook.py:132] global_step...700663
INFO:tensorflow:global_step...700664
I0420 23:36:40.024840 281472996956240 npu_hook.py:132] global_step...700664
INFO:tensorflow:global_step...700665
I0420 23:36:40.267009 281472996956240 npu_hook.py:132] global_step...700665
INFO:tensorflow:NPUCheckpointSaverHook end...
I0420 23:36:40.267664 281472996956240 npu_hook.py:137] NPUCheckpointSaverHook end...
INFO:tensorflow:Saving checkpoints for 700665 into /home/test_user03/ckpt5/model.ckpt.
I0420 23:36:40.269501 281472996956240 basic_session_run_hooks.py:606] Saving checkpoints for 700665 into /home/test_user03/ckpt5/model.ckpt.
INFO:tensorflow:Loss for final step: 4.1664658.
I0420 23:38:08.704852 281472996956240 estimator.py:371] Loss for final step: 4.1664658.
```

## 推理/验证过程<a name="section1465595372416"></a>

1.  通过“模型训练”中的验证指令启动验证。

2.  当前只能针对该工程训练出的checkpoint进行推理测试。

3.  测试结束后会打印测试集的top1 accuracy和top5 accuracy，如下所示。

```
ow:Evaluation [78/781]
I0420 23:40:00.258475 281472996956240 evaluation.py:167] Evaluation [78/781]
INFO:tensorflow:Evaluation [156/781]
I0420 23:40:07.798311 281472996956240 evaluation.py:167] Evaluation [156/781]
INFO:tensorflow:Evaluation [234/781]
I0420 23:40:15.336721 281472996956240 evaluation.py:167] Evaluation [234/781]
INFO:tensorflow:Evaluation [312/781]
I0420 23:40:22.876575 281472996956240 evaluation.py:167] Evaluation [312/781]
INFO:tensorflow:Evaluation [390/781]
I0420 23:40:30.432068 281472996956240 evaluation.py:167] Evaluation [390/781]
INFO:tensorflow:Evaluation [468/781]
I0420 23:40:38.020324 281472996956240 evaluation.py:167] Evaluation [468/781]
INFO:tensorflow:Evaluation [546/781]
I0420 23:40:45.564076 281472996956240 evaluation.py:167] Evaluation [546/781]
INFO:tensorflow:Evaluation [624/781]
I0420 23:40:53.106832 281472996956240 evaluation.py:167] Evaluation [624/781]
INFO:tensorflow:Evaluation [702/781]
I0420 23:41:00.634234 281472996956240 evaluation.py:167] Evaluation [702/781]
INFO:tensorflow:Evaluation [780/781]
I0420 23:41:08.236136 281472996956240 evaluation.py:167] Evaluation [780/781]
INFO:tensorflow:Evaluation [781/781]
I0420 23:41:08.331177 281472996956240 evaluation.py:167] Evaluation [781/781]
2022-04-20 23:41:08.749352: I /home/phisik3/jenkins/workspace/work_code/tmp/host-prefix/src/host-build/asl/tfadaptor/CMakeFiles/tf_adapter.dir/compiler_depend.ts:805] The model has been compiled on the Ascend AI processor, current graph id is: 71
INFO:tensorflow:Finished evaluation at 2022-04-20-23:41:13
I0420 23:41:13.806376 281472996956240 evaluation.py:275] Finished evaluation at 2022-04-20-23:41:13
INFO:tensorflow:Saving dict for global step 700665: global_step = 700665, loss = 1.8883309, top_1_accuracy = 0.75600195, top_5_accuracy = 0.9269366
I0420 23:41:13.807576 281472996956240 estimator.py:2049] Saving dict for global step 700665: global_step = 700665, loss = 1.8883309, top_1_accuracy = 0.75600195, top_5_accuracy = 0.9269366
INFO:tensorflow:Saving 'checkpoint_path' summary for global step 700665: /home/test_user03/ckpt5/model.ckpt-700665
I0420 23:41:13.810555 281472996956240 estimator.py:2109] Saving 'checkpoint_path' summary for global step 700665: /home/test_user03/ckpt5/model.ckpt-700665
INFO:tensorflow:Evaluation results: {'loss': 1.8883309, 'top_1_accuracy': 0.75600195, 'top_5_accuracy': 0.9269366, 'global_step': 700665}
I0420 23:41:13.813197 281472996956240 amoeba_net.py:467] Evaluation results: {'loss': 1.8883309, 'top_1_accuracy': 0.75600195, 'top_5_accuracy': 0.9269366, 'global_step': 700665}
```
