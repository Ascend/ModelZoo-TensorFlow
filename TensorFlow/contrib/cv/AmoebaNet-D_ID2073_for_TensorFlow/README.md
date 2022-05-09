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

     首先在脚本xx.sh中，配置训练数据集路径和checkpoint保存路径，请用户根据实际路径配置，数据集参数如下所示：

     ```
        DATA_DIR=/home/test_user03/tf_records/
        MODEL_DIR=/home/test_user03/hh
      
     ```

  2. 启动训练。

     （脚本为xx.sh） 

     ```
     bash xx.sh
     ```


- 验证。

  1. 配置验证参数。

     首先在脚本xx.sh中，配置训练数据集路径和checkpoint保存路径，请用户根据实际路径配置，数据集参数如下所示：

     ```
        DATA_DIR=/home/test_user03/tf_records/
        MODEL_DIR=/home/test_user03/hh
      
     ```

  2. 启动验证。

     （脚本为xx.sh） 

     ```
     bash xx.sh
     ```          ```
- 测试用例。

  2. 测试用例测试指令（脚本位于xx.sh）

      ```
      bash xx.sh
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

1.  通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡、8卡网络训练。

2.  参考脚本的模型存储路径为results/1p或者results/8p，训练脚本log中包括如下信息。

```
2020-06-20 22:25:48.893067: I tf_adapter/kernels/geop_npu.cc:64] BuildOutputTensorInfo, num_outputs:1
2020-06-20 22:25:48.893122: I tf_adapter/kernels/geop_npu.cc:93] BuildOutputTensorInfo, output index:0, total_bytes:8, shape:, tensor_ptr:140670893455168, output140653543141408
2020-06-20 22:25:48.893165: I tf_adapter/kernels/geop_npu.cc:745] [GEOP] RunGraphAsync callback, status:0, kernel_name:GeOp133_0[ 1330us]step:150120  epoch: 30.0  FPS: 4216.5  loss: 3.373  total_loss: 4.215  lr:0.09106
2020-06-20 22:25:48.897526: I tf_adapter/kernels/geop_npu.cc:545] [GEOP] Begin GeOp::ComputeAsync, kernel_name:GeOp133_0, num_inputs:0, num_outputs:1
2020-06-20 22:25:48.897593: I tf_adapter/kernels/geop_npu.cc:412] [GEOP] tf session direct5649af5909132193, graph id: 51 no need to rebuild
2020-06-20 22:25:48.897604: I tf_adapter/kernels/geop_npu.cc:753] [GEOP] Call ge session RunGraphAsync, kernel_name:GeOp133_0 ,tf session: direct5649af5909132193 ,graph id: 51
2020-06-20 22:25:48.897656: I tf_adapter/kernels/geop_npu.cc:767] [GEOP] End GeOp::ComputeAsync, kernel_name:GeOp133_0, ret_status:success ,tf session: direct5649af5909132193 ,graph id: 51 [0 ms]
2020-06-20 22:25:48.898088: I tf_adapter/kernels/geop_npu.cc:64] BuildOutputTensorInfo, num_outputs:1
2020-06-20 22:25:48.898118: I tf_adapter/kernels/geop_npu.cc:93] BuildOutputTensorInfo, output index:0, total_bytes:8, shape:, tensor_ptr:140650333523648, output140653566153952
2020-06-20 22:25:48.898135: I tf_adapter/kernels/geop_npu.cc:745] [GEOP] RunGraphAsync callback, status:0, kernel_name:GeOp133_0[ 529us]
2020-06-20 22:25:48.898456: I tf_adapter/kernels/geop_npu.cc:545] [GEOP] Begin GeOp::ComputeAsync, kernel_name:GeOp133_0, num_inputs:0, num_outputs:1
2020-06-20 22:25:48.898475: I tf_adapter/kernels/geop_npu.cc:412] [GEOP] tf session direct5649af5909132193, graph id: 51 no need to rebuild
2020-06-20 22:25:48.898485: I tf_adapter/kernels/geop_npu.cc:753] [GEOP] Call ge session RunGraphAsync, kernel_name:GeOp133_0 ,tf session: direct5649af5909132193 ,graph id: 51
```

## 推理/验证过程<a name="section1465595372416"></a>

1.  通过“模型训练”中的测试指令启动测试。

2.  当前只能针对该工程训练出的checkpoint进行推理测试。

3.  推理脚本的参数eval_dir可以配置为checkpoint所在的文件夹路径，则该路径下所有.ckpt文件都会根据进行推理。

4.  测试结束后会打印验证集的top1 accuracy和top5 accuracy，如下所示。

```
2020-06-20 19:06:09.349677: I tf_adapter/kernels/geop_npu.cc:338] [GEOP] GeOp Finalize start, tf session: direct24135e275a110a29, graph_id_: 1
2020-06-20 19:06:09.349684: I tf_adapter/kernels/geop_npu.cc:342] tf session: direct24135e275a110a29, graph id: 1
2020-06-20 19:06:09.397087: I tf_adapter/kernels/geop_npu.cc:347] [GEOP] GE Remove Graph success. tf session: direct24135e275a110a29 , graph id: 1
2020-06-20 19:06:09.397105: I tf_adapter/kernels/geop_npu.cc:368] [GEOP] GeOp Finalize success, tf session: direct24135e275a110a29, graph_id_: 1
2020-06-20 19:06:09.398108: I tf_adapter/kernels/geop_npu.cc:338] [GEOP] GeOp Finalize start, tf session: direct24135e275a110a29, graph_id_: 31
2020-06-20 19:06:09.398122: I tf_adapter/kernels/geop_npu.cc:368] [GEOP] GeOp Finalize success, tf session: direct24135e275a110a29, graph_id_: 31
2020-06-20 19:06:09.398247: I tf_adapter/kernels/host_queue_dataset_op.cc:71] Start destroy tdt.
2020-06-20 19:06:09.412269: I tf_adapter/kernels/host_queue_dataset_op.cc:77] Tdt client close success.
2020-06-20 19:06:09.412288: I tf_adapter/kernels/host_queue_dataset_op.cc:83] dlclose handle finish.
2020-06-20 19:06:09.412316: I tf_adapter/kernels/geop_npu.cc:338] [GEOP] GeOp Finalize start, tf session: direct24135e275a110a29, graph_id_: 51
2020-06-20 19:06:09.412323: I tf_adapter/kernels/geop_npu.cc:342] tf session: direct24135e275a110a29, graph id: 51
2020-06-20 19:06:09.553281: I tf_adapter/kernels/geop_npu.cc:347] [GEOP] GE Remove Graph success. tf session: direct24135e275a110a29 , graph id: 51
2020-06-20 19:06:09.553299: I tf_adapter/kernels/geop_npu.cc:368] [GEOP] GeOp Finalize success, tf session: direct24135e275a110a29, graph_id_: 51
2020-06-20 19:06:10.619514: I tf_adapter/kernels/host_queue_dataset_op.cc:172] HostQueueDatasetOp's iterator is released.
2020-06-20 19:06:10.620037: I tf_adapter/kernels/geop_npu.cc:338] [GEOP] GeOp Finalize start, tf session: direct24135e275a110a29, graph_id_: 41
2020-06-20 19:06:10.620054: I tf_adapter/kernels/geop_npu.cc:342] tf session: direct24135e275a110a29, graph id: 41
2020-06-20 19:06:10.621564: I tf_adapter/kernels/geop_npu.cc:347] [GEOP] GE Remove Graph success. tf session: direct24135e275a110a29 , graph id: 41
2020-06-20 19:06:10.622904: I tf_adapter/util/session_manager.cc:50] find ge session connect with tf session direct24135e275a110a29
2020-06-20 19:06:10.975070: I tf_adapter/util/session_manager.cc:55] destory ge session connect with tf session direct24135e275a110a29 success.
2020-06-20 19:06:11.380491: I tf_adapter/kernels/geop_npu.cc:388] [GEOP] Close TsdClient.
2020-06-20 19:06:11.664666: I tf_adapter/kernels/geop_npu.cc:393] [GEOP] Close TsdClient success.
2020-06-20 19:06:11.665011: I tf_adapter/kernels/geop_npu.cc:368] [GEOP] GeOp Finalize success, tf session: direct24135e275a110a29, graph_id_: 41 step  epoch  top1    top5     loss   checkpoint_time(UTC)85068    3.0  50.988   76.99    3.09  
2020-06-20 18:06:0690072    3.0  51.569   77.51    3.03  
2020-06-20 18:11:1495076    3.0  51.689   77.33    3.00  
2020-06-20 18:16:22100080    3.0  51.426   77.04    3.08  
2020-06-20 18:25:11105084    3.0  51.581   77.50    3.03  
2020-06-20 18:34:23Finished evaluation
```
