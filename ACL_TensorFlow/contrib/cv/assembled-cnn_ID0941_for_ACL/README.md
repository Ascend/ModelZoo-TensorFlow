-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [推理环境准备](#推理环境准备.md)
-   [模型固化](#模型固化.md)
-   [ATC转换](#ATC转换.md)
-   [离线推理](#离线推理.md)
<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：** Image Classification 

**版本（Version）：1.2**

**修改时间（Modified） ：2021.8.26**

**大小（Size）：154M(pb) 81M(om)**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：pb/om**

**精度（Precision）：fp32**

**处理器（Processor）：昇腾310**

**应用级别（Categories）：Official**

**描述（Description）：基于TensorFlow框架的图像分类网络训练代码** 

<h2 id="概述.md">概述</h2>

Assembled-CNN是一个经典的图像分类网络，主要特点是通过改进并集成不同CNN中诸如Selective Kernel, Anti Alias, Big Little Networks, Residual Blocks等方案，提升模型效率。该方法取得2019年iFood图像细粒度分类大赛第一名。 

- 参考论文：

    [Lee, Jungkyu, et al. "Compounding the performance improvements of assembled techniques in a convolutional neural network." arXiv preprint arXiv:2001.06268 (2020).](https://arxiv.org/pdf/2001.06268v2.pdf) 

- [参考实现](https://github.com/clovaai/assembled-cnn)

<h2 id="推理环境准备.md">推理环境准备</h2>

- 离线推理数据集
    - 样本输入[下载链接](https://pan.baidu.com/s/17LwxwzxoxEGnatUHoCls2g), 提取码：fqcv
    - ground truth [下载链接](https://pan.baidu.com/s/19lMswr9AZ7AnhCMH6dWCtA), 提取码：uh7d
    - 输入与ground truth均为BIN格式

- 待固化的ckpt模型
    - [下载链接](https://pan.baidu.com/s/1WN-eixkTJv1TDhUtuUxWpQ), 提取码：00iu



<h2 id="模型固化.md">模型固化</h2>
使用前，将代码中的ckpt_path换为待固化的模型路径前缀。

运行固化命令
```
cd code
python freeze_graph.py
```
所生成模型会在```./pb_model/npu.pb```下。[下载链接](https://pan.baidu.com/s/1aOs7vpnZRTvvDH3tUWkxtQ)，提取码：t373

<h2 id="ATC转换.md">ATC转换</h2>
配置好ATC环境后，运行ATC转换命令
```
atc --model=$PATH_TO_PB_MODEL --framework=3 --output=$OUTPUT_ROOT/$YOUR_OM_MODEL_NAME --soc_version=Ascend310  --input_shape="input:1,256,256,3"
```
得到om模型。[下载链接](https://pan.baidu.com/s/1021h7AIB1fPnx_j6Y69e8A), 提取码：meuy


<h2 id="离线推理.md">离线推理</h2>

- 启动推理
    在310上配置好msame后，执行
    ```
    ./msame --model $OM_MODEL_PATH --input $BIN_INPUT_PATH --output $OUTPUT_PATH --outfmt BIN
    ```

- 结果比较
    910环境中运行下列命令，将解析ground truth与predict的BIN文件并进行比较，输出Top1精度
    ```
    cd code
    python offline_infer.py --predict_path=$BIN_PREDICT_PATH --gt_path=$BIN_GT_PATH
    ```

    | |Top1 Accuracy |
    |----|:----:|
    | Online  |92.22% |
    | Offline | 92.18%|



