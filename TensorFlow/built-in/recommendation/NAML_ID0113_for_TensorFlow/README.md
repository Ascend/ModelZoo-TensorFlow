-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [快速上手](#快速上手.md)
-   [迁移学习指导](#迁移学习指导.md)
-   [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：Recommendation**

**版本（Version）：1.1**

**修改时间（Modified） ：2021.08.25**

**大小（Size）：6.3M**

**框架（Framework）：TensorFlow_1.15.0**

**模型格式（Model Format）：h5**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Research**

**描述（Description）：基于TensorFlow+keras框架的NAML推荐网络训练代码** 

<h2 id="概述.md">概述</h2>

NAML是一个新闻推荐网络，核心是新闻编码器和用户编码器。在新闻编码器中，使用一种多视图学习模型(Attentive Multi-view)，即通过将标题，正文和主题类别视为新闻的不同视图来学习统一的新闻表示形式。此外，还将单词级别和视图级别的注意力机制(Word-level & View-level)应用于新闻编码器，以选择重要的单词和视图来学习信息性新闻表示。在用户编码器中，基于用户浏览的新闻来学习用户的表示，并应用注意力机制来选择信息性新闻以进行用户表示学习。

- 参考论文：

    [Chuhan Wu, Fangzhao Wu, Mingxiao An, Jianqiang Huang, Yongfeng Huang, Xing Xie. “Neural News Recommendation with Attentive Multi-View Learning” IJCAI2019NAML](https://wuch15.github.io/paper/IJCAI19NAML.pdf)


- 参考实现：

    https://github.com/microsoft/recommenders/blob/main/examples/00_quick_start/naml_MIND.ipynb


- 适配昇腾 AI 处理器的实现：

    https://gitee.com/hxxhl88/modelzoo/tree/master/built-in/TensorFlow/Research/recommendation/NAML_for_TensorFlow
        


- 通过Git获取对应commit\_id的代码方法如下：
    
    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    ```

## 默认配置<a name="section91661242121611"></a>

- 训练数据集（新闻推荐的MIND数据集是从微软新闻网站的匿名行为日志中收集的，以MIND-large训练集为例，仅作为用户参考示例）：

  - 用户的点击历史记录和印象日志：behaviors.tsv
  - 新闻文章的信息：news.tsv
  - 从WikiData知识图中提取的新闻实体的embedding向量：entity_embedding.vec
  - 从WikiData知识图中提取的实体之间关系的embedding向量：relation_embedding.vec

- 测试数据集（新闻推荐的MIND数据集是从微软新闻网站的匿名行为日志中收集的，以MIND-large验证集为例，仅作为用户参考示例）

  - 用户的点击历史记录和印象日志：behaviors.tsv
  - 新闻文章的信息：news.tsv
  - 从WikiData知识图中提取的新闻实体的embedding向量：entity_embedding.vec
  - 从WikiData知识图中提取的实体之间关系的embedding向量：relation_embedding.vec

- 训练超参

  - Batch size: 32
  - Learning rate(LR): 0.0001
  - Optimizer: adam
  - loss: cross_entropy_loss
  - epoch: 10
  - support_quick_scoring: true


## 支持特性<a name="section1899153513554"></a>

| 特性列表  | 是否支持 |
|-------|------|
| 分布式训练 | 是    |
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
1. 模型训练使用MIND-large数据集，数据集请用户自行获取。


2. 数据集压缩包解压后，放入模型目录下，在训练脚本中指定数据集路径，可正常使用。
   

## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。

- 启动训练之前，首先要配置程序运行相关环境变量。

  环境变量配置信息参见：

     [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/modelzoo/wikis/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=3148819)

- 单卡训练 

  1. 配置训练参数。

     首先在脚本tests/train_performance_1p.sh中，配置训练数据集路径，请用户根据实际路径配置，数据集参数如下所示：

     ```
      --data_path=${currentDir}/data
     ```

  2. 启动训练。

     启动单卡训练 （脚本为NAML_for_TensorFlow/tests/train_performance_1p.sh） 

     ```
     bash train_performance_1p.sh
     ```

- 8卡训练

  1. 配置训练参数。

     首先在脚本tests/train_performance_8p.sh中，配置训练数据集路径，请用户根据实际路径配置，数据集参数如下所示：

     ```
      --data_path=${currentDir}/data
     ```

  2. 启动训练。

     启动单卡训练 （脚本为NAML_for_TensorFlow/tests/train_performance_8p.sh） 

     ```
     bash train_performance_8p.sh
     ```


- 验证。

    1. 测试的时候，无需修改代码，默认一个epoch训练完成后eval一次。
    

<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码<a name="section08421615141513"></a>

```
├── README.md                                 //代码说明文档
├── recommenders-master
│    ├──examples                               //网络训练示例
│    │    ├──00_quick_start                    //模型训练主入口
│    │    │    ├──naml_MIND.py                 //naml训练入口    
│    ├──reco_utils                                  
│    │    ├──recommender                            //推荐
│    │    │    ├──newsrec                           //新闻推荐
│    │    │    │    ├── io
│    │    │    │    │    ├──mind_all_iterator.py    //数据加载和处理
│    │    │    │    ├── models
│    │    │    │    │    ├──base_model.py           //模型基类构建
│    │    │    │    │    ├──naml.py                 //模型naml构建
│    │    │    │    ├── newsrec_utils.py            //公共函数
├── test
│    ├──env.sh                               //环境变量配置文件
│    ├──hccl_1p.json                         //1卡运行配置文件
│    ├──hccl_8p.json                         //8卡运行配置文件
│    ├──train_performance_1p.sh              //8卡执行脚本
│    ├──train_performance_8p.sh              //1卡执行脚本
```

## 脚本参数<a name="section6669162441511"></a>

```
--batch_size              默认32
--model_path              使用NPU卡数量，默认：./
--data_path               数据集路径，默认：./
--max_steps               训练迭代次数，默认：None
--epochs                  训练epoch次数，默认：1
--MIND_type               数据集类型，默认'small'，可选["demo", "small", "large"]
```




