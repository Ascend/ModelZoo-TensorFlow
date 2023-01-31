-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [快速上手](#快速上手.md)
-   [训练结果](#训练结果.md)
-   [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Natural Language Processing** 

**版本（Version）：**

**修改时间（Modified） ：2022.3.16**

**大小（Size）：**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：h5**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Research**

**描述（Description）：基于TensorFlow框架的OSNet网络训练代码** 

<h2 id="概述.md">概述</h2>

	经过多版本改进的目标检测网络。

- 参考论文（无论文，仅有github仓库）：

    https://github.com/ultralytics/yolov5

- 参考实现：

    https://github.com/bubbliiiing/yolov5-keras

- 适配昇腾 AI 处理器的实现：
    
        
  https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/YOLOV5_ID0378_for_TensorFlow
        


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

        数据增强：
        (1)将图像随机裁剪出256128的尺寸然后按1.25的因子重新缩放；
        (2)随机水平翻转；
        (3)随机擦除
  - 图像的输入尺寸为128×64×3
  - 图像输入格式：.jpg

- 测试数据集预处理：

  - 图像的输入尺寸：多样
  - 图像输入格式：.jpg

- 训练超参

  - Batch size： 8
  - Train epoch: 120
  - initial learning rate：0.01


## 支持特性<a name="section1899153513554"></a>

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 否    |
| 混合精度  | 是    |
| 并行数据  | 否   |

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
1. 模型训练使用VOC2007+2012数据集（参考文献https://github.com/bubbliiiing/yolov5-keras）

2. 根据文献里面的readme自行下载数据集（https://pan.baidu.com/s/19Mw2u_df_nBzsC2lg20fQA，提取码: j5ge）
    
## 模型训练<a name="section715881518135"></a>

1.解压下载需要的数据集

2.数据集需要生成对应的txt文件存储位置信息

    将voc_annotation.py文件中VOCdevkit_path修改为../../../VOCdevkit格式，指定到数据集路径

    python voc_annotation.py

    得到2007_train.txt、2007_val.txt文件，在train.py中配置路径

- 启动训练之前，首先要配置程序运行相关环境变量。

  环境变量配置信息参见：

     [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)

- 单卡训练 

  1. 配置训练参数。

     在train.py中，配置参数。

     ```
     classes_path               指向model_data下的voc_classes.txt
     anchors_path               指向model_data下的yolo_anchors.txt'
     train_annotation_path      指向2007_train.txt
     val_annotation_path        指向2007_val.txt'
     ```

  2. 配置测试参数。

     在yolo.py中，配置参数。
     ```
        model_path                 指向训练好的模型
     ```
     在get_map.py中，配置参数。
     ```
        VOCdevkit_path             指向VOC数据集位置
     ```

  3. 启动训练和测试。
     ```
     bash train_full_1p.sh
     ```

<h2 id="训练结果.md">训练结果</h2>

- 精度结果比对

|精度指标项|论文发布|GPU实测|NPU实测|
|---|---|---|---|
|mAP|xxx|66.0%|64.65%|

- 性能结果比对  (暂无)

|性能指标项|论文发布|GPU实测|NPU实测|
|---|---|---|---|
|FPS|XXX|YYY|ZZZ|


<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码<a name="section08421615141513"></a>

```
├── train.py                                    //网络训练代码
├── README.md                                   //代码说明文档
├── get_map.py                                  //精度测试代码
├── yolo.py                                     //yolo网络创建代码
├── nets                                        //模型网络结构代码
├── requirements.txt                            //训练python依赖列表
├── train_full_1p.sh                            //训练启动脚本
├── utils                                       //训练与测试流程工具代码
├── 2007_train.txt                              //VOC训练集信息
├── 2007_val.txt                                //VOC测试集信息
│    、、、                 
```

## 脚本参数<a name="section6669162441511"></a>

```
在train.py中设置参数

```

## 训练过程<a name="section1589455252218"></a>

1.  通过“模型训练”中的训练指令启动单卡卡训练。