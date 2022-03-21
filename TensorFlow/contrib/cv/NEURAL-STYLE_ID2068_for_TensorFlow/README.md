-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [快速上手](#快速上手.md)
-   [迁移学习指导](#迁移学习指导.md)
-   [高级参考](#高级参考.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：CV**

**版本（Version）：**

**修改时间（Modified） ：2022.2.14**

**大小（Size）：160k**

**框架（Framework）：TensorFlow 1.15.0**

**处理器（Processor）：昇腾910**

**描述（Description）：直接使用训练好的VGG19分类网络，将艺术画风迁移到普通的图片上，使机器也可以画名画。** 

<h2 id="概述.md">概述</h2>

风格迁移的核心思路：使用现成的识别网络，提取图像不同层级的特征。其中低层次响应描述图像的风格，高层次响应描述图像的内容。使用梯度下降方法，可以调整输入响应，在特定层次获得特定的响应。多次迭代之后，输入响应即为特定风格和内容的图像。

特点：常见的深度学习问题利用输入、输出样本训练网络的权重，得到网络模型后使用测试数据推理。而这篇文章中，是利用已经训练好的权重，每次输入两幅图像，通过迭代过程达到风格迁移，并不涉及推理过程。

使用的预训练网络：直接使用训练好的VGG19分类网络。

迁移过程：使用分类网络中卷积层的响应来表达图像的风格和内容。输入内容图与风格图，并以高斯噪声为初始迁移图像，计算它们之间的内容损失和风格损失，并将二者加权得到总损失。优化内容+风格的总损失，多次执行前向/后向迭代使用L-BFGS方法优化，即可实现风格迁移。

- 开源代码：

    https://github.com/anishathalye/neural-style。

- 参考论文：

    [K. Simonyan and A. Zisserman. Very Deep Convolutional Networks for Large-Scale Image Recognition.]

- 参考实现：

    obs://cann-id2068/gpu/

- 适配昇腾 AI 处理器的实现：
  

- 通过Git获取对应commit\_id的代码方法如下：
  
    ```
    git clone {repository_url}    # 克隆仓库的代码
    cd {repository_name}    # 切换到模型的代码仓目录
    git checkout  {branch}    # 切换到对应分支
    git reset --hard ｛commit_id｝     # 代码设置到对应的commit_id
    cd ｛code_path｝    # 切换到模型代码所在路径，若仓库下只有该模型，则无需切换
    ```

## 默认配置<a name="section91661242121611"></a>

- 数据：内容图与风格图见https://github.com/anishathalye/neural-style下examples文件夹


- 预训练的网络：

  - 使用了19层VGG网络的16个卷积层和5个池化层所提供的特征空间，不使用任何一个完全连接的图层。
  - 该模型是公开可用的。下载链接：https://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat。

- 训练超参

    - CONTENT_WEIGHT = 5e0
    - CONTENT_WEIGHT_BLEND = 1
    - STYLE_WEIGHT = 5e2
    - TV_WEIGHT = 1e2
    - STYLE_LAYER_WEIGHT_EXP = 1
    - LEARNING_RATE = 1e1
    - BETA1 = 0.9
    - BETA2 = 0.999
    - EPSILON = 1e-08
    - STYLE_SCALE = 1.0
    - ITERATIONS = 1000
    - VGG_PATH = 'imagenet-vgg-verydeep-19.mat'
    - POOLING = 'max'
    

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


## 模型训练<a name="section715881518135"></a>

        

<h2 id="迁移学习指导.md">迁移学习指导</h2>



<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码<a name="section08421615141513"></a>



## 脚本参数<a name="section6669162441511"></a>

```
--content <content file> 
--styles <style file> 
--output <output file>
--checkpoint-output  --checkpoint-iterations  保存检查点图像。
--iterations   更改迭代的次数(默认为1000)。500-2000 次迭代似乎会产生不错的结果。
--content-weight，--style-weight和--learning-rate 对于某些图像或输出大小，需要一些超参数调整。
--style-layer-weight-exp调整样式转换的“抽象”程度。
--content-weight-blend指定内容传输层的系数。默认值1.0，风格转换尝试保留更细致的内容细节。值应该在[0.0,1.0]。
--pooling允许去选择使用平均池化层还是最大池化层，原始 VGG 使用最大池化，但风格迁移论文建议将其替换为平均池化。
--preserve colors保留内容图颜色选项
```


## 训练过程<a name="section1589455252218"></a>


## 推理/验证过程<a name="section1465595372416"></a>

## 训练结果

论文无明确的精度指标，具体情况可以看二者输出的图片：
GPU：obs://cann-id2068/gpu/1-output.jpg
NPU：obs://cann-id2068/npu/MA-new-pr1-02-14-15-24/output/11.jpg

若要用数据体现，可以看迭代1000次后的损失，也就是输出图片与内容图风格图之间的差距

|                     | content loss | style loss | tv loss | total loss |
| ------------------- | ---------- | ----- | ------ | ----- |
| GPU精度数据         | 847266     | 468498 | 72813.6  | 1.38858e+06 |
| NPU训练精度         | 874779      | 240435 | 77285.6  | 1.1925e+06 |

|                     | sec/step | 
| ------------------- | ---------- | 
| GPU性能数据         | 0.057018280  |     
| NPU性能数据         | 4.555015087  |       




