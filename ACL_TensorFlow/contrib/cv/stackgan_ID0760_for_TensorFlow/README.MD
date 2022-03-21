-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [模型性能](#模型性能.md)
-   [高级参考](#高级参考.md)

<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：image generation** 

**版本（Version）：1.2**

**修改时间（Modified） ：2021.10.23**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt,pb,om**


**处理器（Processor）：昇腾910A, 昇腾310**

**应用级别（Categories）：Demo**

**描述（Description）：基于TensorFlow框架的stackgan图像生成，训练推理代码** 

<h2 id="概述.md">概述</h2>

根据文字描述，人工生成高质量图片的任务是计算机视觉领域一个挑战，并且有很多应用场景。现有的文字转图像方式很难展现文字的含义，并且细节部分缺失严重，不够生动具体。stackgan采用了两阶段训练：首先是 Stage-I，根据给定的文字描述，勾勒初始的形状和色彩，生成低分辨率的图像，然后 Stage-II 根据 Stage-I 生成的低分辨率图像以及原始文字描述，生成具有更多细节的高分辨率图像。这个阶段可以重新捕获被 Stage-I 忽略的文字描述细节，修正 Stage-I 的的结果的缺陷，并添加改良的细节。

- 参考论文：

    [StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks](https://arxiv.org/pdf/1612.03242v1.pdf) by Han Zhang, Tao Xu, Hongsheng Li, Shaoting Zhang, Xiaogang Wang,   Xiaolei Huang, Dimitris Metaxas.


- 参考实现：https://github.com/hanzhanggit/StackGAN

## 默认配置<a name="section91661242121611"></a>

- resize图像的输入尺寸为256*256
- 随机水平翻转图像


- 训练超参

  - Batch size: 64
  - DISCRIMINATOR_LR = 2e-4
  - GENERATOR_LR = 2e-4
  - Optimizer: ADAM
  - Weight decay: 0.00001
  - Train epoch: 600


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

- 模型固化
  * 将pickle转换成bin文件
    ```
    python3 file2bin.py
    ```

  * 将freeze_graph.py中model_path指定为stageII训练得到的模型的绝对路径，执行得到pb模型
       ```
      python3 stageII/freeze_graph.py
      #tensorflow的freeze_graph.freeze_graph有问题。stackgan的模型是两个输出，但是freeze成两个输出的pb报错。因此freeze成了两个单输出的模型
      ```
  * 借助ATC工具将pb模型转换成能够在310芯片上进行推理的om模型，需执行如下的atc命令：

      ``` shell
        atc --model=./frozen_model.pb --framework=3 --output=./model --soc_version=Ascend310 --input_shape='Placeholder:64,1024' 
        atc --model=./frozen_model2.pb --framework=3 --output=./model2 --soc_version=Ascend310 --input_shape='Placeholder:64,1024'
        #将两个单输出的模型转换om
      ```
  * 使用msame工具测试
      ```
      msame --model /root/yzx/model.om --input /root/yzx/offline_inference/source_binfile --output /root/yzx/offline_inference/output_binfile
      msame --model /root/yzx/model2.om --input /root/yzx/offline_inference/source_binfile --output /root/yzx/offline_inference/output_binfile2
      ```
  * 后处理
      ```
      python3 stageII/run_exp.py --cfg stageII/cfg/birds.yml --gpu 0 --bin
      #首先将stageII/cfg/birds.yml中的train.flag改为false，然后运行上述代码。bin参数意味着不对代码其中的模型进行推理，直接从bin文件读取数据进行后续推理。图像保存后依据在线推理的说明，使用脚本对保存的图像进行测试。

      ```
      
  * 网盘链接：[pb模型、om模型](https://disk.pku.edu.cn:443/link/AF1FD3B3DFEC436F320FB723013F41B6)
[GPU版本网盘链接](https://disk.pku.edu.cn:443/link/B5A5FE6DB805E1E7ABFABEBF6682D050
)
<h2 id="模型性能.md">模型性能</h2>

