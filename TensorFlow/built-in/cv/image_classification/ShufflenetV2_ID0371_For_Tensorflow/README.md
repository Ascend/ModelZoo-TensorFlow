-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [快速上手](#快速上手.md)
-   [迁移学习指导](#迁移学习指导.md)
-   [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Image Classification **

**版本（Version）：1.1**

**修改时间（Modified） ：2021.8.15**

**大小（Size）：299KB**

**框架（Framework）：TensorFlow_1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：FP32**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Research**

**描述（Description）：基于TensorFlow框架ShuffleNet升级版本学习算法训练代码**

<h2 id="概述.md">概述</h2>

-    目前，神经网络体系结构设计大多以计算复杂度的间接度量为指导，即FLOPs。然而，直接度量（如速度）也取决于其他因素，如内存访问成本和平台特性。因此，这项工作建议评估目标平台上的直接指标，而不仅仅是考虑故障。在一系列受控实验的基础上，得出了有效网络设计的几种实用指导原则。因此，提出了一种新的体系结构，称为ShuffleNet V2。

- -   参考论文：

      https://arxiv.org/abs/1807.11164

  -   参考实现：
      
      ```
      https://github.com/TropComplique/shufflenet-v2-tensorflow
      ```

-   适配昇腾 AI 处理器的实现：
    
        ```
        https://gitee.com/ascend/ModelZoo-TensorFlow/blob/master/TensorFlow/built-in/cv/image_classification/ShufflenetV2_ID0371_For_Tensorflow

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

-   训练超参（单卡）：
    -   Batch size: 128
    -   iterations_per_loop: 10
    -   Learning rate\(LR\): 0.0625
    -   Weight decay: 4e-5
    -   Train epoch: 133


## 支持特性 <a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 是       |
| 混合精度   | 是       |
| 数据并行   | 是       |


## 混合精度训练 <a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度【深加工】<a name="section20779114113713"></a>
拉起脚本中，传入--precision_mode='allow_mix_precision'


相关代码示例。



```
run_config = NPURunConfig(
        model_dir=self.config.model_dir,
        session_config=session_config,
        keep_checkpoint_max=5,
        save_checkpoints_steps=5000,
        enable_data_pre_proc=True,
        iterations_per_loop=iterations_per_loop,
        precision_mode='allow_mix_precision',
        hcom_parallel=True
      ）
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

## 数据集准备<a name="section361114841316"></a>

1. 模型训练使用imagenet2012数据集，数据集请用户自行获取。

2. 这里是列表文本数据集训练前需要做预处理操作。

   ```
   python explore_imagenet.py
   python create_tfrecords.py \
       --metadata_file=training_metadata.csv \
       --output=/path/to/your/dataset/ \
       --labels=integer_encoding.json \
       --boxes=boxes.npy \
       --num_shards=1000
   python create_tfrecords.py \
       --metadata_file=validation_metadata.csv \
       --output=/path/to/your/dataset/\
       --labels=integer_encoding.json \
       --num_shards=100
   
   (说明：数据集预处理部分numpy的版本不能高于1.6.2或者在create_tfrecords.py里np.load()里添加参数allow_pickle=True）
   ```

3. 这里是列表文本数据集处理后，放入模型目录下，在训练脚本中指定数据集路径，可正常使用。

## 模型训练 <a name="section715881518135"></a>
- 下载训练脚本。
- 检查是否有存在8卡IP的json配置文件“8p.json”。
  
```
 {"group_count": "1","group_list":     
                [{"group_name": "worker","device_count": "8","instance_count": "1", "instance_list":      
                                         [{"devices":                               
                                         [{"device_id":"0","device_ip":"192.168.100.101"},                
                                         {"device_id":"1","device_ip":"192.168.101.101"},                   
                                         {"device_id":"2","device_ip":"192.168.102.101"},                  
                                         {"device_id":"3","device_ip":"192.168.103.101"},                
                                         {"device_id":"4","device_ip":"192.168.100.100"},                 
                                         {"device_id":"5","device_ip":"192.168.101.100"},                  
                                         {"device_id":"6","device_ip":"192.168.102.100"},                   
                                         {"device_id":"7","device_ip":"192.168.103.100"}],                 
                                     "pod_name":"npu8p",        "server_id":"127.0.0.1"}]}],"status": "completed"}
```

- 开始训练。
  
    1. 启动训练之前，首先要配置程序运行相关环境变量。

       环境变量配置信息参见：

          [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)
    

    2. 单卡训练
       
        2.1 单卡训练指令（脚本位于ShuffleNetV2_ID0371_For_Tensorflow/test/train_full_1p.sh），示例如下。
        
        ```
        bash test/train_full_1p.sh 
        ```
        

   3. 8卡训练 

        3.1 8卡训练指令（脚本位于ShuffleNetV2_ID0371_For_Tensorflow/test/train_full_8p.sh），示例如下。
        
        ```
        bash test/train_full_8p.sh 
        ```
        




<h2 id="高级参考.md">高级参考 </h2>

## 脚本和示例代码<a name="section08421615141513"></a>

    ├-- README.md                            #说明文档
    ├-- requirements.txt						 #依赖
    ├-- test			#训练脚本目录	
    ├     |--train_full_1p.sh
          |--train_full_8p.sh
          |--......


   


## 脚本参数 <a name="section6669162441511"></a>

```
    --data_dir                        train data dir, default : path/to/data
    --num_classes                     number of classes for dataset. default : 1000
    --batch_size                      mini-batch size ,default: 128 
    --lr                              initial learning rate,default: 0.06
    --max_epochs                      total number of epochs to train the model:default: 133
    --warmup_epochs                   warmup epoch(when batchsize is large), default: 5
    --weight_decay                    weight decay factor for regularization loss ,default: 4e-5
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

## 训练过程 <a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动单卡或者多卡训练。单卡和多卡通过运行不同脚本，支持单卡，8卡网络训练。模型存储路径为curpath/output/ASCEND_DEVICE_ID，包括训练的log以及checkpoints文件。以8卡训练为例，loss信息在文件curpath/output/{ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。
