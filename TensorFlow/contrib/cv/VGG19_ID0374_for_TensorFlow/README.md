-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [快速上手](#快速上手.md)
-   [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：huawei**

**应用领域（Application Domain）：Classification**

**版本（Version）：1.0**

**修改时间（Modified） ：2021.8.23**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Research**

**描述（Description）：基于TensorFlow框架的VGG19图像分类网络训练代码**

**准确度（accuracy）：top-1    0.729   **

<h2 id="概述.md">概述</h2>

VGG16是一个经典的目标分类网络。整个网络都使用（3x3）的卷积核和（2x2）最大池化层，既可以保证感受视野，又能够减少卷积层的参数。Ascend提供的VGG16是基于TensorFlow实现的版本。
-   参考论文：

       [Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).](https://arxiv.org/pdf/1409.1556.pdf)

-   参考实现：



-   适配昇腾 AI 处理器的实现：
  
  
    https://gitee.com/ascend/modelzoo/tree/master/built-in/TensorFlow/Official/cv/image_classification/VGG16_for_TensorFlow
    
-   通过Git获取对应commit\_id的代码方法如下：
  
    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    ```

## 默认配置<a name="section91661242121611"></a>

-   训练数据集预处理（以ImageNet2012的Train数据集为例，仅作为用户参考示例）：
    -   图像的输入尺寸为224\*224
    -   图像输入格式：TFRecord 随机裁剪图像尺寸
    -   随机水平翻转图像
    -   根据ImageNet2012数据集通用的平均值和标准偏差对输入图像进行归一化
-   测试数据集预处理（以ImageNet2012的Validation数据集为例，仅作为用户参考示例）：
    -   图像的输入尺寸为224*224 （将图像缩放到256 * 256，然后在中央区域裁剪图像）
    -   图像输入格式：TFRecord 根据ImageNet2012数据集通用的平均值和标准偏差对输入图像进行归一化

-   训练超参（单卡）：
    -   Batch size: 256    
    -   Momentum: 0.9
    -   Learning rate\(LR\): 0.01 -> 0.001 -> 0.0001 -> 0.00001
    -   Optimizer: MomentumOptimizer
    -   Weight decay: 0.0001
    -   Label smoothing: 0.1
    -   Train epoch: 150


## 支持特性<a name="section1899153513554"></a>

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 是    |
| 混合精度  | 是    |
| 数据并行  | 是    |

## 混合精度训练<a name="section168064817164"></a>

混合精度训练昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

设置precision_mode参数的脚本参考如下。 

```
run_config = NPURunConfig( model_dir=flags_obj.model_dir, 
                            session_config=session_config, 
                            keep_checkpoint_max=5, 
                            save_checkpoints_steps=5000, 
                            enable_data_pre_proc=True, 
                            iterations_per_loop=iterations_per_loop, 
                            log_step_count_steps=iterations_per_loop, 
                            precision_mode='allow_mix_precision', 
                            hcom_parallel=True 
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
1. 模型训练使用ImageNet2012数据集，数据集请用户自行获取。

2. 数据集训练前需要做预处理操作，请用户参考[Tensorflow-Slim](https://github.com/tensorflow/models/tree/master/research/slim),将数据集封装为tfrecord格式。

3. 数据集处理后，放入模型目录下，在训练脚本中指定数据集路径，可正常使用。

## 模型训练<a name="section715881518135"></a>

-  单击“立即下载”，并选择合适的下载方式下载源码包。
-  启动训练之前，首先要配置程序运行相关环境变量。

   环境变量配置信息参见：

    [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)
-  单卡训练
  
    1. 配置训练参数。 

       在脚本scripts/train_1p.sh中，配置训练数据集路径，请用户根据实际路径配置，数据集参数如下所示：
       
         `−−datadir=`

    2. 执行训练指令（脚本为scripts/run_1p.sh）。
      
        `bash run_1p.sh`

-   验证。

    1. 测试的时候，需要修改脚本启动参数（脚本位于scripts/train_1p.sh），配置mode为evaluate并在eval_dir中配置checkpoint文件所在路径，请用户根据实际路径配置，参数如下所示：
      
       ```
       −−mode=evaluate
        −−datadir=
        --eval_dir=
       ```
2. 测试指令（脚本位于scripts/run_1p.sh）.
  
    `bash run_1p.sh`

<h2 id="高级参考.md">高级参考</h2>

 脚本和示例代码


```
├── train.py                 //网络训练与测试代码 
├── README.md                //代码说明文档 
├── vgg16 
│ ├──vgg.py                  //网络构建 
│ ├──create_session.py       //sess参数配置 
│ ├──data_loader.py          //数据加载 
│ ├──layers.py               //计算accuracy 
│ ├──logger.py               //打印logging信息 
│ ├──model.py                //model estimator 
│ ├──train_helper.py         //ckpt排序 
│ ├──hyper_param.py          //配置学习率策略 
│ ├──trainer.py              //训练器配置 
│ ├──preprocessing.py        //数据预处理 
├── scripts 
│ ├──run_1p.sh               //单卡运行启动脚本 
│ ├──train_1p.sh             //单卡训练执行脚本 
├── test 
│ ├──run_1p.sh               //单卡推理启动脚本 
│ ├──evaluate.sh             //推理执行脚本

```


## 脚本参数<a name="section6669162441511"></a>


```
--rank_size                                    使用NPU卡数量，默认：1 
--mode                                         运行模式，可选：train，evaluate，train_and_evaluate 
--max_train_steps                              训练次数
--iterations_per_loop                          NPU运行时，device端下沉次数
--max_epochs                                   训练epoch次数
--epochs_between_evals                         train_and_evaluate模式时训练和推理的间隔
--data_dir                                     数据集路径，默认：path/data 
--eval_dir                                     推理时checkpoint文件所在路径
--dtype                                        网络输入数据类型，默认：tf.float32 
--use_nesterov                                 是否使用Nesterov，默认：True 
--label_smoothing                              label smooth系数，默认：0.1 
--weight_decay                                 权重衰减，默认：0.0001 
--batch_size                                   每个NPU的batch size，默认：256
--lr                                           初始学习率
--T_max cosine_annealing                       学习率策略中的T_max值，默认：150 
--momentum                                     动量，默认：0.9 
--display_every                                打屏间隔，默认：1 
--log_name                                     log文件名，默认：vgg19.log 
--log_dir                                      ckpt文件存放路径，默认：./model_1p
```

说明：当前默认模式为train_and_evaluate，每训练epochs_between_evals个epoch测试1次，共训练max_epochs个epoch；可选模式：train，训练max_train_steps次；evaluate模式，对eval_dir目录下的ckpt进行测试。

## 训练过程<a name="section1589455252218"></a>

1. 通过“模型训练”中的训练指令启动单卡训练。
2. 参考脚本的模型存储路径为results/1p 。

## 训练性能<a name="section1589455252218"></a>

| 平台     | BatchSize | FPS        |
| -------- | --------- | ---------- |
| NPU      | 256       | 1120(5.6x) |
| NPU      | 64        | 860(4.3x)  |
| GPU V100 | 64        | 199        |

## 训练精度<a name="section1589455252218"></a>

|          | top1-acc |
| -------- | -------- |
| NPU      | 0.729    |
| GPU V100 | 0.705    |



## 推理/验证过程<a name="section1465595372416"></a>

1. 通过“模型训练”中的测试指令启动测试。 
2. 当前只能针对该工程训练出的checkpoint进行推理测试。 
3. 推理脚本的参数eval_dir可以配置为checkpoint所在的文件夹路径，则该路径下所有.ckpt文件都会进行验证。 
4. 测试结束后会打印验证集的top1 accuracy和top5 accuracy。



