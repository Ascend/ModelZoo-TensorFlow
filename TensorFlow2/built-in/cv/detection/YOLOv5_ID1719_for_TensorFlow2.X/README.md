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

**修改时间（Modified） ：2021.12.01**

**大小（Size）**_**：245M**

**框架（Framework）：TensorFlow 2.4.1**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Research**

**描述（Description）：基于TensorFlow框架的yolov5目标检测算法训练代码**

<h2 id="概述.md">概述</h2>

-    yolov5是yolo系列的最新一作，相较于前作，yolov5在网络结构以及数据增强方面都有些许差异，虽然该作未得到原作者认可，但其使用到的多种优化技巧仍具有一定的学习价值。

- 参考论文：

    [https://arxiv.org/abs/2004.10934](https://arxiv.org/abs/2004.10934)

- 参考实现：

    [https://github.com/hunglc007/tensorflow-yolov4-tflite](https://github.com/hunglc007/tensorflow-yolov4-tflite)

- 适配昇腾 AI 处理器的实现：
    
    [https://gitee.com/ascend/modelzoo/tree/master/built-in/TensorFlow/Research/cv/detection/YOLOv5_ID1719_for_TensorFlow2.X](https://gitee.com/ascend/modelzoo/tree/master/built-in/TensorFlow/Research/cv/detection/YOLOv5_ID1719_for_TensorFlow2.X)

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
    
    - 初始学习率为0.001，使用cosine_decay
    
    -   优化器：adam
    -   单卡batchsize：8
    -   总epoch数为38*8

-   训练超参（多卡）：
    -   Batch size: 8*8
    -   LR scheduler: cosine decay
    -   Learning rate\(LR\): 0.001
    -   stage2_epoch:38*8


## 支持特性<a name="section1899153513554"></a>

| 特性列表   | 是否支持 |
| ---------- | -------- |
| 分布式训练 | 是       |
| 混合精度   | 是       |
| 数据并行   | 是       |


## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>
相关代码示例。

```
import npu_device
npu_device.global_options().precision_mode = 'allow_mix_precision'
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

1. 模型训练使用COCO2017数据集，数据集请用户自行获取。
2. 数据集标注文件需要先后使用scripts目录下coco_convert.py及coco_annotation.py生成。标注文件生成后即内含图片路径及box信息，故数据集图片文件不可随意移动位置。

## 模型训练<a name="section715881518135"></a>
- 单击“立即下载”，并选择合适的下载方式下载源码包。
- 开始训练。
  
    1. 启动训练之前，首先要配置程序运行相关环境变量。

       环境变量配置信息参见：

          [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)
    

    2. 多卡训练
       
        2.1 设置多卡训练参数（脚本位于YOLOv5_ID1719_for_TensorFlow2.X/test/train_full_8p.sh），示例如下。
            
        
        ```
        #经脚本转换后的验证集标注文件，需用户根据实际情况自行配置
        anno_converted='/npu/traindata/COCO2017/val2017.txt'
        #原始验证集标注文件，需用户根据实际情况自行配置
        gt_anno_path='/npu/traindata/COCO2017/annotations/instances_val2017.json'
        #此处为平均到单卡的batchsize
        batch_size=8
        #训练epoch，此处为平均到单卡的epoch数
        stage2_epoch=38
        #精度模式，可选混合精度及fp32模式
        precision_mode='allow_mix_precision' #'allow_fp32_to_fp16'
        ```
        
        
        
        2.2 多卡训练指令（脚本位于YOLOv5_ID1719_for_TensorFlow2.X/test） 

        ```
        bash train_full_8p.sh --data_path=xx
        假设用户生成的标注文件有如下结构，配置data_path时需指定为COCO这一层，例：--data_path=/home/COCO。具体标注文件生成参考数据集准备章节
        ├─COCO
           ├─train2017.txt
           ├─val2017.txt
        ```

<h2 id="迁移学习指导.md">迁移学习指导</h2>

- 数据集准备。

    1.  获取数据。
        请参见“快速上手”中的数据集准备
    
- 模型训练

    请参考“快速上手”章节

<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码<a name="section08421615141513"></a>

    ├── README.md                                //说明文档
    ├── requirements.txt                         //依赖
    ├── train.py                                 //训练入口脚本
    ├── test
    |    |—— train_full_8p.sh                    //多卡训练脚本
    |    |—— train_performance_8p.sh             //多卡训练脚本
    ├── core
    │    ├──yolov4.py                            //网络结构定义脚本
    │    ├──config.py                            //全量网络配置脚本
    ├── scripts
         ├──coco_convert.py                      //数据集标注缓存文件生成脚本
         ├──coco_annotation.py                   //数据集标注文件生成脚本


## 脚本参数<a name="section6669162441511"></a>

```
batch_size                                       训练batch_size
stage2_epoch                                     平均单卡训练epoch数
precision_mode                                   训练精度模式
anno_converted                                   转换后验证集标注文件路径
gt_anno_path                                     原始coco验证集标注文件路径
其余参数请在core/config.py中配置
```

## 训练过程<a name="section1589455252218"></a>

通过“模型训练”中的训练指令启动多卡训练。
将训练脚本（train_full_8p.sh）中的data_path设置为训练数据集的路径。具体的流程参见“模型训练”的示例。