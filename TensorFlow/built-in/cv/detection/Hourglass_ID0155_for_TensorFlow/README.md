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

**修改时间（Modified） ：2021.11.05**

**大小（Size）：500K**

**框架（Framework）：TensorFlow_1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Research**

**描述（Description）：基于TensorFlow框架的Hourglass网络训练代码**

<h2 id="概述.md">概述</h2>

-   Stacked Hourglass是falnewell等人提出应用于人体姿态估计的算法，通过生成heatmaps方式预测人体的关键点。作者认为重复使用 bottom-up/top-down和intermediate supervision能够提升网络性能。2016年在FLIC和MPII数据集上，本文取得了超越当前所有方法的性能。 

    - 参考论文：

        https://arxiv.org/abs/1603.06937


    - 参考实现：
        
        ```
        https://github.com/wbenbihi/hourglasstensorlfow
        ```

    - 适配昇腾 AI 处理器的实现：
    
        ```
        https://gitee.com/ascend/modelzoo/tree/master/built-in/TensorFlow/Research/cv/detection/Hourglass_ID0155_for_TensorFlow
        branch=master
        commit_id= 477b07a1e95a35885b3a9a569b1c8ccb9ad5d7af
        ```


    - 通过Git获取对应commit_id的代码方法如下:
    
        ```
        git clone {repository_url}    # 克隆仓库的代码
        cd {repository_name}    # 切换到模型的代码仓目录
        git checkout  {branch}    # 切换到对应分支
        git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
        cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
        ```

## 默认配置<a name="section91661242121611"></a>

-   网络结构
    -   优化器：RMSProp
    -   单卡batchsize：4
    -   总Epoch数：200

-   训练数据集预处理：
    -   模型使用 MPII DataSet 数据集，请用户自行下载

-   测试数据集预处理：
    -   模型使用 MPII DataSet 数据集，请用户自行下载

-   训练超参（单卡）：
    -   Batch size: 4
    -   Train epoch: 200

## 支持特性 <a name="section1899153513554"></a>

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
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
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
    <td class="cellrowborder" valign="top" width="25.52%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001074498056_p1450714271532"><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><em id="i72359495125"><a name="i72359495125"></a><a name="i72359495125"></a>21.0.2</em></p>
    </td>
    <td class="cellrowborder" valign="top" width="27.16%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001074498056_p18244640152312"><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><em id="i162363492129"><a name="i162363492129"></a><a name="i162363492129"></a><a href="https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software" target="_blank" rel="noopener noreferrer">5.0.2</a></em></p>
    </td>
    </tr>
    </tbody>
    </table>


<h2 id="快速上手.md">快速上手</h2>

## 数据集准备<a name="section361114841316"></a>

1. 模型使用 MPII DataSet 数据集，请用户自行下载。

2. 放入模型目录下，在训练脚本中指定数据集路径，可正常使用。

## 模型训练<a name="section715881518135"></a>
- 单击“立即下载”，并选择合适的下载方式下载源码包
- 开始训练
  
    1. 启动训练之前，首先要配置程序运行相关环境变量。

       环境变量配置信息参见：

          [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)
    

    2. 单卡训练
       
        2.1 设置单卡训练参数（脚本位于config.cgf），示例如下。请确保下面例子中的“img_directory”修改为用户对应的路径。

        ```
        [DataSetHG]
		training_txt_file: 'dataset.txt'
		img_directory: 'F:/Cours/DHPE/mpii_human_pose_v1/images'
		img_size: 256
		hm_size: 64
		num_joints: 16
		remove_joints: None
		joint_list = ['r_anckle', 'r_knee', 'r_hip', 'l_hip', 'l_knee', 'l_anckle', 'pelvis', 'thorax', 'neck', 'head', 'r_wrist', 'r_elbow', 'r_shoulder', 'l_shoulder', 'l_elbow', 'l_wrist']
		[Network]
		name: 'hg_refined_200'
		nFeats: 256
		nStacks: 4
		nModules: 1
		tiny: False
		nLow: 4
		dropout_rate: 0.2
		mcam: False
		[Train]
		batch_size: 4
		nEpochs: 200
		epoch_size: 1000
		learning_rate: 0.00025
		learning_rate_decay: 0.96
		decay_step: 2000
		weighted_loss: False
		[Validation]
		valid_iteration: 10
		[Saver]
		log_dir_train: 'F:/Cours/DHPE/DHPE/hourglass_tiny/'
		log_dir_test: 'F:/Cours/DHPE/DHPE/hourglass_tiny/'
		saver_step: 500
		saver_directory: ''
        ```
        

        2.2 单卡训练指令（脚本位于./Hourglass_ID0155_for_TensorFlow/test/train_full_1p.sh） 

```
        `bash train_full_1p.sh`
```


<h2 id="迁移学习指导.md">迁移学习指导</h2>

- 数据集准备。

    1.  获取数据。
        请参见“快速上手”中的数据集准备。


-  模型训练。

    请参考“快速上手”章节。


<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码<a name="section08421615141513"></a>

    ├── LICENSE
    ├── README.md
    ├── modelzoo_level.txt
    ├── requirements.txt
    ├── test
    │   ├── train_full_1p.sh                            // 执行训练脚本
    │   └── train_performance_1p.sh			                // 执行训练脚本
    └── train_launcher.py						                    // 训练入口脚本


## 脚本参数<a name="section6669162441511"></a>

```
    --data_dir                        train data dir, default : path/to/data
    --num_classes                     number of classes for dataset. default : 1000
    --batch_size                      mini-batch size ,default: 128 
    --lr                              initial learning rate,default: 0.06
    --max_epochs                      total number of epochs to train the model:default: 150
    --warmup_epochs                   warmup epoch(when batchsize is large), default: 5
    --weight_decay                    weight decay factor for regularization loss ,default: 1e-4
    --momentum                        momentum for optimizer ,default: 0.9
    --label_smoothing                 use label smooth in CE, default 0.1
    --save_summary_steps              logging interval,dafault:100
    --log_dir                         path to save checkpoint and log,default: ./model_1p
    --log_name                        name of log file,default: alexnet_training.log
    --save_checkpoints_steps          the interval to save checkpoint,default: 1000
    --mode                            mode to run the program (train, evaluate), default: train
    --checkpoint_dir                  path to checkpoint for evaluation,default : None
    --max_train_steps                 max number of training steps ,default : 100
    --synthetic                       whether to use synthetic data or not,default : False
    --version                         weight initialization for model,default : he_uniorm
    --do_checkpoint                   whether to save checkpoint or not, default : True
    --rank_size                       number of npus to use, default : 1
```

## 训练过程<a name="section1589455252218"></a>

1. 通过“模型训练”中的训练指令启动单卡训练。
2. 将训练脚本（train_full_1p.sh）中的data_path设置为训练数据集的路径。具体的流程参见“模型训练”的示例。
3. 模型存储路径为“curpath/output/$ASCEND_DEVICE_ID”，包括训练的log以及checkpoints文件。
4. 以单卡训练为例，loss信息在文件curpath/output/{ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。