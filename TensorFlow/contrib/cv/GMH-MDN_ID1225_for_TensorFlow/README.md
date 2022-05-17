-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [快速上手](#快速上手.md)
-   [迁移学习指导](#迁移学习指导.md)
-   [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：cv**

**版本（Version）：1.1**

**修改时间（Modified） ：2021.12.14**

**大小（Size）：249M**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Benchmark**

**描述（Description）：基于TensorFlow框架的基于多峰混合密度网络生成多个可行的 3D 姿态假设的网络**

<h2 id="概述.md">概述</h2>

-   GMH-MDN：一种基于多峰混合密度网络生成多个可行的 3D 姿态假设的网络

-   参考论文：

    ```
    https://arxiv.org/pdf/1904.05547.pdf
    ```

-   参考实现：
        
    ```
    https://github.com/chaneyddtt/Generating-Multiple-Hypotheses-for-3D-Human-Pose-Estimation-with-Mixture-Density-Network
    ```

-   适配昇腾 AI 处理器的实现：
    ```
    https://gitee.com/ascend/modelzoo/tree/master/built-in/TensorFlow/Benchmark/cv/image_classification/Shufflenet_ID0645_for_TensorFlow
    branch=master
    commit_id= 477b07a1e95a35885b3a9a569b1c8ccb9ad5d7af
    ```


-   通过Git获取对应commit\_id的代码方法如下：

    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    ```

## 默认配置 <a name="section91661242121611"></a>
-   网络结构
    -   初始学习率为0.001，对学习率learning_rate应用指数衰减。
    -   优化器：ADAM
    -   学习率衰减速度 decay_steps：100000
    -   学习率衰减系数 decay_rate：0.96
    -   单卡batchsize：64
    -   总Epoch数：200
    -   dropout：0.5

-   训练超参（单卡）：
    -   Batch size: 64
    -   LR scheduler: exponential decay
    -   Learning rate\(LR\): 0.001
    -   Train epoch: 200
    -   dropout：0.5
    -   linear_size：1024 \#ps: size of each layer(每一层神经元的个数)


## 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 否       |
| 混合精度   | 是       |
| 数据并行   | 否       |


## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>
相关代码示例。

```
config = tf.ConfigProto()
custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name =  "NpuOptimizer" 
custom_op.parameter_map["use_off_line"].b = True
custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
custom_op.parameter_map["modify_mixlist"].s = tf.compat.as_bytes("/home/test/ops_info.json")
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
with tf.Session(config=config) as sess:
  print(sess.run(cost))
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

### 数据集准备<a name="section361114841316"></a>

1. 模型预训练使用 [Human3.6M]数据集  ，需用户自行申请。因申请较慢，故可在[此处](https://github.com/MendyD/human36m) 下载

2. 数据集下载后，放入模型目录下，在训练脚本中指定数据集路径，可正常使用。

### 模型训练<a name="section715881518135"></a>
- 下载训练脚本。

- 开始训练。

    1.启动训练之前，首先要配置程序运行相关环境变量。

       环境变量配置信息参见：
          [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)
    
    2.单卡训练
    
    2.1设置单卡训练参数（脚本位于./GMH—MDN_ID1225_for_TensorFlow/test/train_full_1p.sh），示例如下。请确保下面例子中的“data_dir，batch_size，epochs”修改为用户数据集的路径。
        
        data_dir="../data/h36m/"
        batch_size=64
        epochs=200
    2.2 单卡训练指令（脚本位于./GMH—MDN_ID1225_for_TensorFlow/test/train_performance_1p.sh）

        bash train_performance_1p.sh --train_dir <specify your training folder>


<h2 id="开始测试.md">开始测试</h2>

- 预训练模型下载

    [地址](https://drive.google.com/open?id=1ndJyuVL-7fbhw-G654m5U8tHogcQIftT)
- 参数配置

    1.修改脚本启动参数（脚本位于test/train_full_1p.sh），将test设置为True，如下所示：

        data_dir="../data/h36m/"
        batch_size=64
        epochs=200

    2.增加checkpoints的路径，请用户根据checkpoints实际路径进行配置。使用预训练模型或者自己训练的模型
       
        checkpoint_dir=../Models/mdm_5_prior/

- 执行测试指令
  
    1.上述文件修改完成之后，执行测试指令
       
        bash test/train_performance.sh

<h2 id="迁移学习指导.md">迁移学习指导</h2>

- 数据集准备。

    1.获取数据。
        请参见“快速上手”中的数据集准备。
    
    2.数据目录结构如下：

            human36m/
            ├── h36m/
                ├── cameras.h5
                ├── S1/
                ├── S11/
                ├── S5/
                ├── S6/
                ├── S7/
                ├── S8/
                ├── S9/
                └── logging.conf/
- 修改训练脚本。

   1.加载预训练模型。
   修改**load_dir**的路径以及load参数，其中**load**为checkpoint-4874200.index 中的数字部分 4874200

- 模型训练。

    请参考“快速上手”章节。

- 模型评估。
   
    可以参考“模型训练”中训练步骤。

<h2 id="高级参考.md">高级参考【深加工】</h2>

### 脚本和示例代码<a name="section08421615141513"></a>

    ├── README.md                                //说明文档
    ├── requirements.txt						 //依赖
    ├── LICENSE
    ├── Models
    ├── experiments
    ├── src_npu_20211208155957
    │   ├── cameras.py
    │   ├── data_utils.py
    │   ├── logging.conf
    │   ├── mix_den_model.py
    │   ├── predict_3dpose_mdm.py
    │   ├── procrustes.py
    │   └── viz.py


### 脚本参数<a name="section6669162441511"></a>

```
--learning_rate		Learning rate	default:0.001
--dropout		Dropout keep probability 1 means no dropout	default:0.5	
--batch_size		batch size to use during training	default:64	
--epochs		How many epochs we should train for	default:200	
--camera_frame		Convert 3d poses to camera coordinates	default:TRUE	
--max_norm		Apply maxnorm constraint to the weights	default:TRUE	
--batch_norm		Use batch_normalization	default:TRUE	
# Data loading				
--predict_14		predict 14 joints	default:FALSE	
--use_sh		Use 2d pose predictions from StackedHourglass	default:TRUE	
--action		The action to train on 'All' means all the actions	default:All	
# Architecture				
--linear_size		Size of each model layer	default:1024	
--num_layers		Number of layers in the model	default:2	
--residual		Whether to add a residual connection every 2 layers	default:TRUE	
# Evaluation				
--procrustes		Apply procrustes analysis at test time	default:FALSE	
--evaluateActionWise		The dataset to use either h36m or heva	default:TRUE	
# Directories				
--cameras_path		Directory to load camera parameters	default:/data/h36m/cameras.h5	
--data_dir		Data directory	default:   /data/h36m/	
--train_dir		Training directory	default:/experiments/test_git/	
--load_dir		Specify the directory to load trained model	default:/Models/mdm_5_prior/	
# Train or load				
--sample		Set to True for sampling	default:FALSE	
--test		Set to True for sampling	default:FALSE	
--use_cpu		Whether to use the CPU	default:FALSE	
--load		Try to load a previous checkpoint	default:0	
--miss_num		Specify how many missing joints	default:1	
```

## 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡训练，通过运行脚本训练。
将训练脚本（test/train_full_1p.sh）中的data_dir设置为训练数据集的路径。具体的流程参见“模型训练”的示例。
模型存储路径为{train_dir}，包括训练的log以及checkpoints文件。以单卡训练为例，loss信息在文件.{train_dir}/log/log.txt中，示例如下。

```
Epoch:               1
Global step:         48742
Learning rate:       9.80e-04
Train loss avg:      10.5440
=============================
2021-12-10 08:44:00,731 [INFO] root - ===Action=== ==mm==
2021-12-10 08:44:14,404 [INFO] root - Directions    67.16
2021-12-10 08:44:41,598 [INFO] root - Discussion    69.08
2021-12-10 08:44:58,180 [INFO] root - Eating        64.17
2021-12-10 08:45:11,033 [INFO] root - Greeting      70.90
2021-12-10 08:45:34,378 [INFO] root - Phoning       84.17
2021-12-10 08:45:46,680 [INFO] root - Photo         86.36
2021-12-10 08:45:57,517 [INFO] root - Posing        63.92
2021-12-10 08:46:05,577 [INFO] root - Purchases     68.64
2021-12-10 08:46:22,047 [INFO] root - Sitting       82.77
2021-12-10 08:46:35,970 [INFO] root - SittingDown  107.23
2021-12-10 08:46:59,066 [INFO] root - Smoking       75.12
2021-12-10 08:47:14,754 [INFO] root - Waiting       71.51
2021-12-10 08:47:26,528 [INFO] root - WalkDog       78.11
2021-12-10 08:47:38,442 [INFO] root - Walking       59.05
2021-12-10 08:47:49,315 [INFO] root - WalkTogether  63.24
2021-12-10 08:47:49,323 [INFO] root - Average       74.09
2021-12-10 08:47:49,325 [INFO] root - ===================
```


### 推理/验证过程<a name="section1465595372416"></a>

#### 推理验证

在200 epoch训练执行完成后，请参见“模型训练”中的测试流程，需要修改脚本启动参数（脚本位于test/train_performance.sh）将test设置为True，修改load_dir的路径以及load参数，其中load_dir 为模型ckpt目录,load为ckpt 文件 checkpoint-4874200.index 中的数字部分 4874200，然后执行脚本。

`bash train_full_1p.sh --test=True`

该脚本会自动执行验证流程，验证结果若想输出至文档描述文件，则需修改启动脚本参数，否则输出至默认log文件（./experiments/test_git/log/log.txt）中。

```
2021-12-10 07:29:31,061 [INFO] root - Logs will be written to ../experiments/test_git/log
2021-12-10 07:32:14,597 [INFO] root - ===Action=== ==mm==
2021-12-10 07:32:33,258 [INFO] root - Directions    50.76
2021-12-10 07:32:59,096 [INFO] root - Discussion    61.78
2021-12-10 07:33:14,707 [INFO] root - Eating        56.20
2021-12-10 07:33:26,797 [INFO] root - Greeting      60.24
2021-12-10 07:33:49,975 [INFO] root - Phoning       78.02
2021-12-10 07:34:02,201 [INFO] root - Photo         74.15
2021-12-10 07:34:13,259 [INFO] root - Posing        52.02
2021-12-10 07:34:21,237 [INFO] root - Purchases     67.17
2021-12-10 07:34:37,670 [INFO] root - Sitting       78.90
2021-12-10 07:34:50,829 [INFO] root - SittingDown  101.50
2021-12-10 07:35:13,391 [INFO] root - Smoking       66.54
2021-12-10 07:35:28,320 [INFO] root - Waiting       60.78
2021-12-10 07:35:39,677 [INFO] root - WalkDog       68.80
2021-12-10 07:35:51,568 [INFO] root - Walking       52.74
2021-12-10 07:36:02,067 [INFO] root - WalkTogether  57.69
2021-12-10 07:36:02,660 [INFO] root - Average       65.82
2021-12-10 07:36:02,671 [INFO] root - ===================
```
    