-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [快速上手](#快速上手.md)
-   [迁移学习指导](#迁移学习指导.md)
-   [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Aesthetics Assessment**

**版本（Version）：1.1**

**修改时间（Modified） ：2021.08.25**

**大小（Size）：1008k**

**框架（Framework）：TensorFlow_1.15.0**

**模型格式（Model Format）：h5**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Research**

**描述（Description）：基于TensorFlow+keras框架的NIMA图像质量评估训练代码** 

<h2 id="概述.md">概述</h2>

NIMA由两个模型组成，是一个美学和技术图像质量评估的网络。该模型通过迁移学习进行训练，即在使用ImageNet数据集进行预训练的CNN网络模型（MobileNet）基础上，针对分类任务进行了finetune。

- 参考论文：

    [Hossein Talebi and Peyman Milanfar “NIMA: Neural Image Assessment”](https://arxiv.org/pdf/1709.05424.pdf)


- 参考实现：

    https://github.com/titu1994/neural-image-assessment


- 适配昇腾 AI 处理器的实现：

    https://gitee.com/ascend/modelzoo/tree/master/built-in/TensorFlow/Research/cv/detection/NIMA_for_TensorFlow
        


- 通过Git获取对应commit\_id的代码方法如下：
    
    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    ```

## 默认配置<a name="section91661242121611"></a>

- 训练数据集（AVA-dataset：美学质量评估的数据集）：

  - 数据集介绍：来自DPChallenge.com，包括25w张照片，每张照片均有三个标注：
    - 美学质量标注：投票分数0~9，分数越高则质量越高
    - 语义标注：66个tag，接近20w张图至少包含1个tags，15w张图包含2个tags
    - 摄影风格标注（14个分类）
  - 文件信息介绍：
    - AVA.txt：2列->图片ID，3至12列->美学打分分布，13至14列->语义ID，15列->challenge_id
    - aesthetics_images_lists：部分语义类别的文件集合
    - style_image_lists：风格类别文件  
    - challenges.txt：challenge_id对应的意义
    - images：图片文件夹
    - tags.txt：语义ID对应的意义
  
- 测试数据集（AVA-dataset中选取5000张：美学质量评估的数据集）：

  - 数据集介绍：来自DPChallenge.com，包括25w张照片，每张照片均有三个标注：
    - 美学质量标注：投票分数0~9，分数越高则质量越高
    - 语义标注：66个tag，接近20w张图至少包含1个tags，15w张图包含2个tags
    - 摄影风格标注（14个分类）
  - 文件信息介绍：
    - AVA.txt：2列->图片ID，3至12列->美学打分分布，13至14列->语义ID，15列->challenge_id
    - aesthetics_images_lists：部分语义类别的文件集合
    - style_image_lists：风格类别文件  
    - challenges.txt：challenge_id对应的意义
    - images：图片文件夹
    - tags.txt：语义ID对应的意义  

- 训练超参

  - Batch size: 200
  - Learning rate(LR): 0.001
  - Optimizer: adam
  - epoch: 20


## 支持特性<a name="section1899153513554"></a>

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 否    |
| 混合精度  | 是    |
| 并行数据  | 是    |

## 混合精度训练<a name="section168064817164"></a>

昇腾910 AI处理器提供自动混合精度功能，可以针对全网中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

## 开启混合精度<a name="section20779114113713"></a>

脚本已默认开启混合精度，设置precision_mode参数的脚本参考如下。

  ```
    sess_config = tf.ConfigProto()
    custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
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
    <td class="cellrowborder" valign="top" width="25.52%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001074498056_p1450714271532"><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><a name="zh-cn_topic_0000001074498056_p1450714271532"></a><em id="i72359495125"><a name="i72359495125"></a><a name="i72359495125"></a>20.2.0</em></p>
    </td>
    <td class="cellrowborder" valign="top" width="27.16%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001074498056_p18244640152312"><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><a name="zh-cn_topic_0000001074498056_p18244640152312"></a><em id="i162363492129"><a name="i162363492129"></a><a name="i162363492129"></a><a href="https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software" target="_blank" rel="noopener noreferrer">20.2</a></em></p>
    </td> 
    </tr>
    </tbody>
    </table>


<h2 id="快速上手.md">快速上手</h2>

- 数据集准备
1. 模型训练使用AVA数据集，数据集请用户自行获取。


2. 数据集压缩包解压后，放入模型目录下，在训练脚本中指定数据集路径，可正常使用。
   

## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。

- 启动训练之前，首先要配置程序运行相关环境变量。

  环境变量配置信息参见：

     [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

- 单卡训练 

  1. 配置训练参数。

     将25w图片数据集images放到AVA_dataset目录下，将mobilenet_1_0_224_tf_no_top.h5放到环境上/root/.keras/models/目录

     ```
     ```

  2. 启动训练。

     启动单卡训练 （脚本为NIMA_for_TensorFlow/run_npu_1p.sh） 

     ```
     bash run_npu_1p.sh
     ```

- 验证。

    1. 测试的时候，无需修改代码，默认一个epoch训练完成后eval一次。
    

<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码<a name="section08421615141513"></a>

```
├── README.md                                 //代码说明文档
├── AVA_dataset                               //数据集路径
│    ├──aesthetics_image_lists                //部分语义类别的文件集合
│    ├──style_image_lists                     //风格类别文件             
│    ├──challenges.txt                        //challenge_id对应的意义
│    ├──tags.txt                              //语义ID对应的意义
├── utils  
│    ├──check_dataset.py                      //数据集校验
│    ├──data_loader.py                        //数据集加载
├── train_mobilenet.py                        //训练主入口
├── requirements.txt                          //需要安装的依赖文件列表
├── run_npu_1p.sh                             //1卡执行脚本
```

## 脚本参数<a name="section6669162441511"></a>

```
--batch_size              默认200
--epochs                  训练epoch次数，默认：20
```




