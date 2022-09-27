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

## 数据准备

1. 模型训练使用char-CNN-RNN text embeddings和Caltech-UCSD Birds-200-2011数据集，从[此处](https://pan.baidu.com/s/1j3xH9bSTUbW9Ji379LVI3A)(密码：f587)获取。

2. 将bird.zip解压到Data/，将CUB_200_2011.tgz解压到Data/birds/。
3. 预处理：`python misc/preprocess_birds.py`


## 模型训练<a name="section715881518135"></a>

- 单击“立即下载”，并选择合适的下载方式下载源码包。

- 启动训练之前，首先要配置程序运行相关环境变量。

  环境变量配置信息参见：

     [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)

- 单卡训练 
    * 启动单卡训练
     ```
     python stageI/run_exp.py --cfg stageI/cfg/birds.yml --gpu 0
     #首先需要进行第一阶段的训练，训练的模型文件会保存在ckpt/下。
     python stageII/run_exp.py --cfg stageII/cfg/birds.yml --gpu 0
     #然后进行第二阶段的训练，需要在stageII/cfg/birds.yml中的PRETRAINED_MODEL指定第一阶段训练模型文件的目录，训练得到的模型会保存在ckpt/下
     ```
    

- 推理
  * 启动910单卡测试 （脚本为scripts/test.sh）先将[npu_checkpoint](https://disk.pku.edu.cn:443/link/F36FE90B91ED4792239F90B4DDD1569B)下载到MobileFaceNet_Tensorflow/output/ckpt_best/目录下，然后使用如下脚本进行测试。
    ```
    python stageII/run_exp.py --cfg stageII/cfg/birds.yml --gpu 0
    #需要将stageII/cfg/birds.yml中的PRETRAINED_MODEL指定第二阶段得到的训练模型，FLAG设置为false
    ```
- 验证
  * 模型结果验证需要通过预训练的inception model来进行。先将[验证项目](https://github.com/hanzhanggit/StackGAN-inception-model)下载到本地，将[预训练权重](https://disk.pku.edu.cn:443/link/3477A5479337F89E801BCFC75A69C0C3)下载放到验证项目根目录中。
  * 执行
     ```
    python inception_score.py --image_folder IMAGE_FOLDER_PATH
    #其中IMAGE_FOLDER_PATH为推理过程输出的图片所在文件夹，位于ckpt/.../下
     ```

- 模型固化
  * 将freeze.py中model_path指定为stageII训练得到的模型的绝对路径，执行得到pb模型
       ```
      python freeze.py
      ```
  * 借助ATC工具将pb模型转换成能够在310芯片上进行推理的om模型，需执行如下的atc命令：

      ``` shell
        atc --model=./model.pb --framework=3 --output=./model --soc_version=Ascend310 --input_shape='Placeholder:1,64,1024' 
      ```
  * 使用msame工具测试
      ```
      msame --model frozen_model.om --output output/ --loop 100
      ```
      ![avatar](pic/omtest.png)
  * 网盘链接：[pb模型、om模型](https://disk.pku.edu.cn:443/link/AF1FD3B3DFEC436F320FB723013F41B6)
[GPU版本网盘链接](https://disk.pku.edu.cn:443/link/B5A5FE6DB805E1E7ABFABEBF6682D050
)
<h2 id="模型性能.md">模型性能</h2>

## 1. 昇腾910A芯片模型性能

### 训练精度与性能

| Parameters                 |    NPU                  | GPU                           |
| -------------------------- | ----------------------- |------------------------------ |
| Resource                   | Ascend 910              |GPU                            |
| Tensorflow Version         | 1.15.0                  |1.15.0                         |
| Dataset                    | Caltech-UCSD Birds-200-2011                     |Caltech-UCSD Birds-200-2011                            |
| Training Parameters        | epoch=600, batch_size=64 |epoch=600, batch_size=64       |
| Optimizer                  | ADAM                    |ADAM                           |
| Validation Accuracy        | 1.37,0.03(mean,var)     |1.38,0.01(mean,var)         |
| speed                      | 23s/step     |18s/step         |





<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码<a name="section08421615141513"></a>

```
.
├── License
├── README.MD
├── demo
│   ├── birds_demo.sh
│   ├── birds_skip_thought_demo.py
│   ├── cfg
│   │   ├── birds-demo.yml
│   │   ├── birds-eval.yml
│   │   ├── birds-skip-thought-demo.yml
│   │   ├── flowers-demo.yml
│   │   └── flowers-eval.yml
│   ├── demo.py
│   ├── flowers_demo.sh
│   └── get_embedding.lua
├── misc
│   ├── __init__.py
│   ├── config.py
│   ├── custom_ops.py
│   ├── datasets.py
│   ├── preprocess_birds.py
│   ├── preprocess_flowers.py
│   ├── skipthoughts.py
│   ├── tf_upgrade.py
│   └── utils.py
├── models
│   └── README.md
├── modelzoo_level.txt
├── requirements.txt
├── stageI
│   ├── __init__.py
│   ├── cfg
│   │   ├── birds.yml
│   │   └── flowers.yml
│   ├── model.py
│   ├── run_exp.py
│   └── trainer.py
└── stageII
    ├── __init__.py
    ├── cfg
    │   ├── birds.yml
    │   └── flowers.yml
    ├── model.py
    ├── run_exp.py
    └── trainer.py
```

说明：当前代码仅支持单卡训练与验证。

## 训练过程<a name="section1589455252218"></a>

1. 通过“模型训练”中的训练指令启动单卡训练

2. 下面是训练过程中的部分日志输出

```
Epoch 0 | g_loss: 1.2287179; d_loss: 1.66146
epoch #1|  0%|                                                  |ETA:  --:--:--
epoch #1|  1%|                                                   |ETA:  0:00:19
epoch #1|  2%|#                                                  |ETA:  0:00:19
epoch #1|  4%|##                                                 |ETA:  0:00:19
epoch #1|  5%|##                                                 |ETA:  0:00:18
epoch #1|  7%|###                                                |ETA:  0:00:18
epoch #1|  8%|####                                               |ETA:  0:00:18
epoch #1| 10%|#####                                              |ETA:  0:00:17
epoch #1| 11%|#####                                              |ETA:  0:00:17
epoch #1| 13%|######                                             |ETA:  0:00:17
epoch #1| 14%|#######                                            |ETA:  0:00:17
epoch #1| 15%|########                                           |ETA:  0:00:16
epoch #1| 17%|########                                           |ETA:  0:00:16
epoch #1| 18%|#########                                          |ETA:  0:00:16
epoch #1| 20%|##########                                         |ETA:  0:00:15
epoch #1| 21%|###########                                        |ETA:  0:00:15
epoch #1| 23%|###########                                        |ETA:  0:00:15
epoch #1| 24%|############                                       |ETA:  0:00:15
epoch #1| 26%|#############                                      |ETA:  0:00:14
epoch #1| 27%|##############                                     |ETA:  0:00:14
epoch #1| 28%|##############                                     |ETA:  0:00:14
epoch #1| 30%|###############                                    |ETA:  0:00:13
epoch #1| 31%|################                                   |ETA:  0:00:13
epoch #1| 33%|#################                                  |ETA:  0:00:13
epoch #1| 34%|#################                                  |ETA:  0:00:13
epoch #1| 36%|##################                                 |ETA:  0:00:12
epoch #1| 37%|###################                                |ETA:  0:00:12
epoch #1| 39%|###################                                |ETA:  0:00:12
epoch #1| 40%|####################                               |ETA:  0:00:11
epoch #1| 42%|#####################                              |ETA:  0:00:11
epoch #1| 43%|######################                             |ETA:  0:00:11
epoch #1| 44%|######################                             |ETA:  0:00:11
epoch #1| 46%|#######################                            |ETA:  0:00:10
epoch #1| 47%|########################                           |ETA:  0:00:10
epoch #1| 49%|#########################                          |ETA:  0:00:10
epoch #1| 50%|#########################                          |ETA:  0:00:09
epoch #1| 52%|##########################                         |ETA:  0:00:09
epoch #1| 53%|###########################                        |ETA:  0:00:09
epoch #1| 55%|############################                       |ETA:  0:00:09
epoch #1| 56%|############################                       |ETA:  0:00:08
epoch #1| 57%|#############################                      |ETA:  0:00:08
epoch #1| 59%|##############################                     |ETA:  0:00:08
epoch #1| 60%|###############################                    |ETA:  0:00:07
epoch #1| 62%|###############################                    |ETA:  0:00:07
epoch #1| 63%|################################                   |ETA:  0:00:07
epoch #1| 65%|#################################                  |ETA:  0:00:06
epoch #1| 66%|##################################                 |ETA:  0:00:06
epoch #1| 68%|##################################                 |ETA:  0:00:06
epoch #1| 69%|###################################                |ETA:  0:00:06
epoch #1| 71%|####################################               |ETA:  0:00:05
epoch #1| 72%|####################################               |ETA:  0:00:05
epoch #1| 73%|#####################################              |ETA:  0:00:05
epoch #1| 75%|######################################             |ETA:  0:00:04
epoch #1| 76%|#######################################            |ETA:  0:00:04
epoch #1| 78%|#######################################            |ETA:  0:00:04
epoch #1| 79%|########################################           |ETA:  0:00:04
epoch #1| 81%|#########################################          |ETA:  0:00:03
epoch #1| 82%|##########################################         |ETA:  0:00:03
epoch #1| 84%|##########################################         |ETA:  0:00:03
epoch #1| 85%|###########################################        |ETA:  0:00:02
epoch #1| 86%|############################################       |ETA:  0:00:02
epoch #1| 88%|#############################################      |ETA:  0:00:02
epoch #1| 89%|#############################################      |ETA:  0:00:02
epoch #1| 91%|##############################################     |ETA:  0:00:01
epoch #1| 92%|###############################################    |ETA:  0:00:01
epoch #1| 94%|################################################   |ETA:  0:00:01
epoch #1| 95%|################################################   |ETA:  0:00:00
epoch #1| 97%|#################################################  |ETA:  0:00:00
epoch #1| 98%|################################################## |ETA:  0:00:00
Lossy conversion from float32 to uint8. Range [-1.0, -0.7576702833175659]. Convert image to uint8 prior to saving to suppress this warning.
Lossy conversion from float32 to uint8. Range [-1.0, -0.7708669900894165]. Convert image to uint8 prior to saving to suppress this warning.
Epoch 1 | g_loss: 0.77000165; d_loss: 1.432099
epoch #2|  0%|                                                  |ETA:  --:--:--
... ...
```