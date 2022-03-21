-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [快速上手](#快速上手.md)
-   [迁移学习指导](#迁移学习指导.md)
-   [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>
**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Image Classification**

**版本（Version）：1.1**

**修改时间（Modified） ：2021.04.25**

**大小（Size）**_**：152KB** 

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的用于移动设备的极其高效ShuffleNet卷积神经网络 **

<h2 id="概述.md">概述</h2>
ShuffleNet：一种用于移动设备的极其高效的卷积神经网络

-   参考论文：

        https://arxiv.org/abs/1707.01083

-   参考实现：
        
        https://github.com/TropComplique/ShuffleNet-tensorflow

    
-   适配昇腾 AI 处理器的实现：
    
        https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/built-in/cv/image_classification/Shufflenet_ID0645_for_TensorFlow

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
    -   初始学习率为1e-1
    -   优化器：Momentum
    -   单卡batchsize：200
    -   总Epoch数设置为 35
    -   Weight decay为5e-3，Momentum为0.9

## 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 否       |
| 混合精度   | 是       |
| 数据并行   | 是       |


## 混合精度训练 <a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度 <a name="section20779114113713"></a>
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

1.  硬件环境准备请参见各硬件产品文档"[驱动和固件安装升级指南](https://support.huawei.com/enterprise/zh/category/ai-computing-platform-pid-1557196528909)"。需要在硬件设备上安装与CANN版本配套的固件与驱动。
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
    <td class="cellrowborder" valign="top" width="25.52%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001074498056_p1450714271532"><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><em id="i72359495125"><a name="i72359495125"></a><a name="i72359495125"></a>21.2.0</em></p>
    </td>
    <td class="cellrowborder" valign="top" width="27.16%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001074498056_p18244640152312"><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><em id="i162363492129"><a name="i162363492129"></a><a name="i162363492129"></a><a href="https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software" target="_blank" rel="noopener noreferrer">5.0.2</a></em></p>
    </td>
    </tr>
    </tbody>
    </table>



<h2 id="快速上手.md">快速上手</h2>
## 数据集准备<a name="section361114841316"></a>

1. 模型预训练使用 [Tiny ImageNet]数据集  ，需用户自行下载。

2. 数据集下载后，执行image_dataset_to_tfrecords.py转换成tfrecords格式，放入模型目录下，在训练脚本中指定数据集路径，可正常使用。

## 模型训练 <a name="section715881518135"></a>

- 开始训练。
  
    1. 启动训练之前，首先要配置程序运行相关环境变量。

       环境变量配置信息参见：

          [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)
    

    2. 单卡训练
       
        2.1 设置单卡训练参数（脚本位于./ShuffleNet_ID0645_for_TensorFlow/test/train_performance_1p.sh），示例如下。请确保下面例子中的“--data_path”修改为用户数据集的路径。
        
        ```
        `nohup python3 train.py \
        --train_tfrecords=${data_path}/train.tfrecords \
        --val_tfrecords=${data_path}/val.tfrecords \
        --batch_size=${batch_size} \
        --num_epochs=${train_epochs} \
        --steps_per_epoch=${train_steps} \
        --initial_lr=${learning_rate} `
        ```
        
        
        2.2 单卡训练指令（脚本位于./ShuffleNet_ID0645_for_TensorFlow/test/train_performance_1p.sh） 

```
        `bash train_performance_1p.sh --data_path=../tiny-imagenet-200`
```


<h2 id="迁移学习指导.md">迁移学习指导</h2>
- 数据集准备。

    可以参考快速上手“数据集准备”中步骤。

- 模型训练。

    可以参考快速上手“模型训练”中步骤。

<h2 id="高级参考.md">高级参考</h2>
## 脚本和示例代码<a name="section08421615141513"></a>

    ├──shufflenet										 
    │    ├──CONSTANTS.py			
    │    ├──__init__.py				 
    │    ├──architecture.py	
    │    ├──get_shufflenet.py
    │    ├──architecture.py
    │    ├──input_pipeline.py		
    ├── README.md                                               //说明文档
    ├── requirements.txt					//依赖
    ├──test										 
    │    ├──train_performance_1p.sh				 //单卡功能、性能训练脚本
    │    ├──train_full_1p.sh				         //单卡全量训练脚本
    │    ├──env.sh						 //环境变量
    ├──tiny_imagenet
    │    ├──move_data.py
    ├──LICENSE           	 
    ├──image_dataset_to_tfrecords.py          	     
    ├──prediction_utils.py      
    ├──train.py    
    ├──prediction_utils.py             		    


## 脚本参数 <a name="section6669162441511"></a>

```
    --data_path                       train data dir, default : ../tiny-imagenet-200
    --batch_size                      mini-batch size ,default: 200
    --learning_rate                   initial learning rate,default: 1e-1
    --train_epochs                    total number of epochs to train the model:default: 35
    --train_steps                     steps per epoch, default: 500
```

