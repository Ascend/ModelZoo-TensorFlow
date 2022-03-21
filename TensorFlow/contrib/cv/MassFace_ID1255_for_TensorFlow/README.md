-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [快速上手](#快速上手.md)
-   [迁移学习指导](#迁移学习指导.md)
-   [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：yule-li**

**应用领域（Application Domain）： Image Classification**

**修改时间（Modified） ：2021.12.24**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的MassFace网络训练代码** 

<h2 id="概述.md">概述</h2>

This project provide an efficient implementation for deep face recognition using Triplet Loss. When trained on [CASIA-Webface](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html) and tested on on [LFW](http://vis-www.cs.umass.edu/lfw/),this code can achieve an 98.3% accuracy with softmax pretrain and [98.6%](models/model-20190214-150620.ckpt-600000) with CosFace pretrain. The framework using triplet loss can be seen as the following figure. It contrains Data Sample, Feature Extractor, Triplet Selection and Triplet Loss modules. The details can be seen as our technical report: [**MassFace: an effecient implementation using triplet loss for face recognition**](https://arxiv.org/abs/1902.11007)


| *Framework using triplet loss to train face recognition model: total P ∗ K images are sampled for P persons with K images each person. The sampled images are mapped into feature vectors through deep convolutional network. The indexs of triplet pairs are computed by a hard example mining process based on the feature vectors and the responding triplet feature pairs can be gathered. Finally, the features of triplet pairs are inputed into triplet loss to train the CNN.* |

**Data Sample**: we sample total P*K images for P persons with K images each in an iteration. We implement it by tensorflow ```tf.data``` api.

**Feature Extractor**: We use MobileFacenets [1] to extract the feature of the input image as a deep representation. It only has about 6.0MB parameters and can be infered very fast as well.

**Triplet Selection**: Triplet selection aims to choice the valid triplet (i, j, k) which is used as input of triplet loss. The valid triplet menas that i, j have the identity and i, k have different identity. We implement serveral mining strategy to select triplet pairs.
- Batch All
- Batch Random
- Batch Min Min
- Batch Min Max
- Batch Hardest
- 参考论文：

  [Li Y . MassFace: an efficient implementation using triplet loss for face recognition[J].  2019.
- 参考实现：

    

- 适配昇腾 AI 处理器的实现：
  
  [https://gitee.com/ascend/modelzoo/tree/master/built-in/TensorFlow/Official/cv/image_classification/DenseNet121_ID0067_for_TensorFlow](https://gitee.com/ascend/modelzoo/tree/master/built-in/TensorFlow/Official/cv/image_classification/DenseNet121_ID0067_for_TensorFlow)      


- 通过Git获取对应commit\_id的代码方法如下：
  
    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
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
将casia-112x112数据集分为两个数据集（webface-112x112，casia-112x112）用于预训练和训练；lfw-112x112用于测试。

 1. 训练预训练模型train_softmax.py的数据集webface-112x112。

 2. 训练预训练模型train_triplet.py的数据集casia-112x112。

 3. 测试集test.py的数据集lfw-112x112。

 数据集obs：URL:
https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=2Rd/lItddR2BAnf7eD0qCKZruW9F9UAhLoc715aav4nSFjDbbYSlGe/LAwGbhEG9fPmPdrImmVNlnnlkarzsN6aFwZ/KTJ1dnwiyFtqmq+Ano8a9knTw0JbNKGN11S/uzDVGVeK/tM5O5Ber6Kv7hMT3yLpAOoBBC/3UL/RZrANyng1zplORCfRAh/FGijwh+r71p2YnJ+0/0I6z3KsIqXIClomkCQuZJ8hz6WsY5Auc04926cKC+SI/ohPTQqFvILggQ61xcnXzTI4eptCUunV+iJK8kRAHWgtJWb6WYgxbvL9hgYvWefXfwTvQP2Jmkv9QWD+HtBfcQQtGdZiACuukBtEu+NyVj1N4vy+R9ZamDCnkPG5Nsd8Y+HiwyiKAB0obHd6nnnxtgzf4ngJiHtaiF0XwousDNER456cqPHXfHBKqrD8srXLW0hj8+gQyDUFp7jiEg6xs5N2Wxb4ta1aV1Fi6DWe1wYIPfIUH37Eq+0lai7DVgmEsz6SPlRdYpQ/aexTrl2qmUcWkftQeCQ==

提取码:
111111

*有效期至: 2022/06/22 10:44:14 GMT+08:00

## 模型训练<a name="section715881518135"></a>
- Ⅰ 预训练Pretrain with softmax loss（100个epoch），所使用的数据集名称为webface-112x112，执行命令为: python train_softmax.py,得到预训练模型存于/home/qyy/MassFac/models/facenet_ms_mp(新更新的代码将模型存于/home/qyy/MassFac/models-softmax）

- Ⅱ 训练Train with triplet loss（由于时间问题训练了50个epoch），所使用的数据集名称为casia-112x112，模型存于/home/qyy/MassFac/models；镜像：swr.cn-north-4.myhuaweicloud.com/ascend-share/5.0.4.alpha002_tensorflow-ascend910-cp37-euleros2.8-aarch64-training:1.15.0-21.0.2_1217

   1、  将/home/ma-user/miniconda3/envs/TensorFlow-1.15-arm/lib/python3.7/site-packages/tensorflow_core/python/ops/nn_grad.py替换为上传的nn_grad.py文件。

   2、  修改819行的预训练模型参数的路径为训练好的路径（--pretrained_model）。

   3、  然后执行:python train_triplet.py。
 

- 训练执行流程总结  
 （1） python train_softmax.py。

 （2）  修改train_triplet.py第819行的预训练模型路径参数pretrained_model，使用第一步得到的预训练模型。

 （3）  "cp -r /home/qyy/MassFac/nn_grad.py /home/ma-user/miniconda3/envs/TensorFlow-1.15-arm/lib/python3.7/site-packages/tensorflow_core/python/ops/nn_grad.py"。

 （4）  "python /home/qyy/MassFac/train_triplet.py"。

## Test测试
- 在30-32行设置数据集（lfw-112x112）路径和model路径，然后执行：python test.py（测试时数据集需完整的测试数据集，否则会报错）
### npu0.671（triplet50个epoch）
### gpu0.682（triplet50个epoch）

## 训练过程<a name="section1589455252218"></a>

1.  通过“模型训练”中的训练指令启动训练。

2.  参考脚本的预训练模型存储路径为/home/qyy/MassFac/models/facenet_ms_mp(新更新的代码将模型存于/home/qyy/MassFac/models-softmax），训练模型存储路径为/home/qyy/MassFac/models。

## 推理
calAcc.py首先将推理得到的output存入emb_array，调用源码中的lfw.evaluate计算精度，总的来说是通过询问测试系统两张照片是否是同一个人，将系统给出的答案和真实标签actual_issame.txt比较得到精度值。最终精度为Accuracy: 0.679+-0.016