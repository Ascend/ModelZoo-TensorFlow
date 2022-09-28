-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [训练环境准备](#训练环境准备.md)
-   [模型性能](#模型性能.md)
-   [高级参考](#高级参考.md)

<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：** Image Classification 

**版本（Version）：1.1**

**修改时间（Modified） ：2021.08.20**

**大小（Size）：60M**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt,pb,om**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾910A, 昇腾310**

**应用级别（Categories）：Demo**

**描述（Description）：基于TensorFlow框架的DilatedRNN图像分类网络训练,推理代码** 

<h2 id="概述.md">概述</h2>

DilatedRNN提出了多分辨率扩张型递归跳跃连接，可以与各种RNN一起使用。此外，DilatedRNN可以有效提高训练效率，在参数较少的情况下保持较为不错的性能。此外，作者提出了一种RNN的度量方法，称为平均递归长度。

本文的主要贡献有三个:

1. 提出了一种新的因果循环跳跃链，以更少的参数缓解了梯度问题，扩大了依赖范围。
2. 将多个膨胀递归层叠加，构造膨胀RNN，使多层可以学习多维相关性。
3. 提出了一种新的度量方法，即平均循环长度，来衡量之前提出的循环跳跃连接与本文提出的扩展RNN版本之间的差异。 

参考论文：
[Shiyu Chang, Yang Zhang, et al. "Dilated Recurrent Neural Networks." *Advances in Neural Information Processing Systems 30 (NIPS 2017)*](https://arxiv.org/abs/1703.05175)

模型架构：
  * DilatedRNN体系结构基本由多个RNN模块组成，但在经典RNN模块中加入多分辨率扩张型跳跃连接，构成扩张型RNN单元。

  * 膨胀RNN由多尺度隐层组成，这些隐层均由基本的膨胀RNN单元构成，参数“膨胀”呈指数增长。通过引入跳跃循环连接，扩张RNN可以有效提高训练效率，在参数更少的情况下保持下游任务的state性能。

  ![img](./imgs/1.png) 


## 默认配置<a name="section91661242121611"></a>

- 数据集：MNIST，由Tensorflow原生支持，用户不需要额外时间下载

- 训练超参

  - seed = 92916
  - hidden_structs = [20] * 9
  - dilations = [1, 2, 4, 8, 16, 32, 64, 128, 256]

  - batch_size = 128
  - learning_rate = 1e-3
  - training_iters = batch_size * 30000
  - testing_iters = batch_size * 2


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

- 单击“立即下载”，并选择合适的下载方式下载源码包。

- 启动训练之前，首先要配置程序运行相关环境变量。

  环境变量配置信息参见：

     [Ascend 910训练平台环境变量设置](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/01.%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC%E8%BF%81%E7%A7%BB%E6%A1%88%E4%BE%8B/Ascend%20910%E8%AE%AD%E7%BB%83%E5%B9%B3%E5%8F%B0%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE)

- 单卡训练 
    * 启动单卡训练 （脚本为DilatedRNN_for_TensorFlow/scripts/train_1p_full.sh）
     同时，脚本运行可以输入对应的参数，包括当前的RNN Cell类型(支持GRU与LSTM)，以及训练/验证过程的轮数等等。
     ```
     bash train_1p_full.sh
     ```
     对于只需要执行一些step的训练过程的用户，可以执行只训练些许step的脚本
     ```
     bash train_1p_less.sh
     ```

- 验证
  * 启动910单卡测试 （脚本为./scripts/test.sh）
     同时，脚本运行可以输入对应的参数，包括当前的RNN Cell类型(支持GRU与LSTM)，以及训练/验证过程的轮数等等。

     ```
     bash test.sh
     ```
  * 启动310单卡测试（脚本为./scripts/om_eval.sh）,借助msame工具验证om模型在310芯片上的推理性能，结合print_om_acc.py脚本，即可查看om模型推理精度

     ```
     bash om_eval.sh
     python3 print_om_acc.py
     ```

<h2 id="模型性能.md">模型性能</h2>

## 1. 昇腾910A芯片模型性能

### 训练性能与精度表现

| Parameters                 |                                               |
| -------------------------- | --------------------------------------------- |
| Resource                   | Ascend 910; CPU 2.60GHz, 24cores; Memory, 72G |
| Update Date                | 12/01/2021 (month/day/year)                   |
| Tensorflow Version         | 1.15.0                                        |
| Dataset                    | MNIST                                         |
| Training Parameters        | epoch=30000, batch_size=128, lr=0.001         |
| Optimizer                  | RMSProp                                       |
| Loss Function              | Softmax Cross Entropy With Logits Loss        |
| outputs                    | probability                                   |
| Loss                       | 0.02 (gru)  0.03 (lstm)                       |
| Validation Accuracy        | 95.82% (gru)  95.70% (lstm)                   |
| Speed                      | 0.13 s / step                                 |
| Total time                 | 80 mins                                       |
| Checkpoint for Fine tuning | 51.7 MB (gru) 46.4 MB (lstm) (.ckpt file)     |


### 测试性能与精度表现

| Parameters          |                                               |
| ------------------- | --------------------------------------------- |
| Resource            | Ascend 910; CPU 2.60GHz, 24cores; Memory, 72G |
| UpdateDate          | 12/01/2021 (month/day/year)                   |
| Tensorflow Version  | 1.15.0                                        |
| Dataset             | MNIST                                         |
| batch_size          | 128                                           |
| outputs             | probability                                   |
| Evaluation Accuracy | 95.27% (lstm)  95.58%(gru)                    |
| Speed               | 3.02 s / test iteration                       |
| Total time          | 5 min                                         |


## 2. 昇腾310芯片模型推理性能

1. 我们需要将原本在910A芯片上训练所得的参数固化pb模型，转换成能够在310芯片上进行推理的om模型，需要执行如下的atc命令：
```
atc --model=/home/HwHiAiUser/cml/lstm.pb --framework=3 --output=/home/HwHiAiUser/cml/tf_lstm_310 --soc_version=Ascend310 --input_shape='Placeholder:1,784,1' --precision_mode=allow_fp32_to_fp16 --op_select_implmode=high_precision
```
经过atc工具转换之后，成功由原先的Tensorflow框架下的pb模型，转换成能够在NPU架构下进行推理的om模型


2. 借助msame工具，我们将输入图片数据转换成bin格式，执行以下命令进行om模型推理：
```
# Ascend310 om model evaluation script
# input dir: ./mnist/data/input
# output dir: ./out 

msame --model './om/tf_lstm_310.om' --input './mnist/data/input' --output './out' 
```

3. 结合所得om模型推理所得输出，编写脚本计算om模型推理所得精度：
```
import numpy as np

input_num = 10000
cell_type = 'lstm'
correct = 0.0

prefix = './data/om_310_output/tf_lstm_310/'

# read ground-truth labels
labels = np.load("./data/gt/labels.npy", allow_pickle=True)

# calculate accuracy
for idx in range(input_num):
    pred_path = prefix + '{0:05d}_output_0.bin'.format(idx)
    pred = np.fromfile(pred_path, dtype=np.float16)
    gt = labels[idx]

    # argmax to get the max value's index of pred and gt
    idx_p, idx_g = np.argmax(pred), np.argmax(gt)
    # judge if prediction match the ground truth label
    if idx_p == idx_g:
        correct += 1.0

# final accuracy
acc = correct / input_num * 100

# print final evaluation accuracy
print("======= Final Eval Accuracy =======")
print("Current Environment: Ascend310")
print("Current Om Output Dir Prefix: %s" % prefix)
print("Om Model Cell Type: %s " % cell_type)
print("MNIST Test Set Evaluation Accuracy: %.2f%%" % acc)
```

4. 推理结果截图
![img](./imgs/gru_om_eval_310.png)
![img](./imgs/lstm_om_eval_310.png)

### 310芯片推理性能与精度表现

| Parameters          |                                               |
| ------------------- | --------------------------------------------- |
| Resource            | Ascend 310; CPU 2.60GHz, 24cores; Memory, 72G |
| UpdateDate          | 12/01/2021 (month/day/year)                   |
| Tensorflow Version  | 1.15.0                                        |
| Dataset             | MNIST                                         |
| batch_size          | 1                                             |
| outputs             | probability                                   |
| Evaluation Accuracy | 93.14% (lstm)  93.16%(gru)                    |
| Speed               | 3.91 s / test iteration                       |
| Total time          | 1 min                                         |

### GPU, Ascend310, Ascend910A推理性能与精度对比

| Environment          | Evaluation Accuracy     | Evaluation Speed          | Train Speed     | Trained Model Size        |
| -------------------- | ----------------------- | ------------------------  | --------------- | ------------------------- |
| GPU (Tesla V100)     | 94.52%(lstm) 94.30%(gru)| 3.17 s / test iteration   | 0.20 s / step   | 16.3MB(lstm) 51.4MB(gru)  |
| Ascend910A           | 95.27%(lstm) 95.58%(gru)| 3.02 s / test iteration   | 0.13 s / step   | 46.4MB(lstm) 51.7MB(gru)  |
| Ascend310            | 93.14%(lstm) 93.16%(gru)| 3.91 s / test iteration   |         -       | 49.7MB(lstm) 104.8MB(gru) |




<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码<a name="section08421615141513"></a>

```
├── README.md
├── requirements.txt
├── LICENSE
├── modelzoo_level.txt
├── Dockerfile
├── scripts
│   ├── test.sh
│   ├── om_eval.sh
│   ├── train_1p_less.sh
│   └── train_1p_full.sh
├── dilated_rnn
│   ├── classification_models.py
│   └── drnn.py
├── imgs
├── log
|   ├── train
|       ├── loss+perf_gru.txt
|       ├── loss+perf_lstm.txt
|   └── eval
|       ├── perf_gru.txt
|       ├── perf_lstm.txt
├── author.txt
├── test.py
├── train.py
├── freeze_graph.py
├── print_om_acc.py
```

## 脚本参数<a name="section6669162441511"></a>

```
-type 当前使用的RNN Cell类型，默认为LSTM
-epoch 训练过程/验证过程的轮数，默认为30000
-batch_size 训练批大小，默认为128
-lr 学习率，默认为0.001
```

说明：当前代码仅支持单卡训练与验证。

## 训练过程<a name="section1589455252218"></a>

1. 通过“模型训练”中的训练指令启动单卡训练

2. 完整训练过程的输出日志存储在./log/train文件夹下，分别包含GRU单元与LSTM单元在整体训练过程中输出的loss日志
   
3. 训练过程中会间隔执行推理过程，同时输出validation accuracy日志

4. 下面是训练过程中的部分日志输出

```
================ Training Process ================
[2021-11-13 15:18:46,213] logger {logger.py:28} INFO - ### Init. Logger logger ###
Extracting ./dataset/train-images-idx3-ubyte.gz
Extracting ./dataset/train-labels-idx1-ubyte.gz
Extracting ./dataset/t10k-images-idx3-ubyte.gz
Extracting ./dataset/t10k-labels-idx1-ubyte.gz
==> Building a dRNN with GRU cells
Building layer: multi_dRNN_dilation_1, input length: 784, dilation rate: 1, input dim: 1.
=====> Input length for sub-RNN: 784
Building layer: multi_dRNN_dilation_2, input length: 784, dilation rate: 2, input dim: 20.
=====> Input length for sub-RNN: 392
Building layer: multi_dRNN_dilation_4, input length: 784, dilation rate: 4, input dim: 20.
=====> Input length for sub-RNN: 196
Building layer: multi_dRNN_dilation_8, input length: 784, dilation rate: 8, input dim: 20.
=====> Input length for sub-RNN: 98
Building layer: multi_dRNN_dilation_16, input length: 784, dilation rate: 16, input dim: 20.
=====> Input length for sub-RNN: 49
Building layer: multi_dRNN_dilation_32, input length: 784, dilation rate: 32, input dim: 20.
=====> 16 time points need to be padded. 
=====> Input length for sub-RNN: 25
Building layer: multi_dRNN_dilation_64, input length: 784, dilation rate: 64, input dim: 20.
=====> 48 time points need to be padded. 
=====> Input length for sub-RNN: 13
Building layer: multi_dRNN_dilation_128, input length: 784, dilation rate: 128, input dim: 20.
=====> 112 time points need to be padded. 
=====> Input length for sub-RNN: 7
Building layer: multi_dRNN_dilation_256, input length: 784, dilation rate: 256, input dim: 20.
=====> 240 time points need to be padded. 
=====> Input length for sub-RNN: 4
[2021-11-30 20:28:48,356] ml_logger {train.py:142} INFO - Iter 100, Minibatch Loss: 1.213610, Training Accuracy: 0.617188
[2021-11-30 20:29:01,167] ml_logger {train.py:142} INFO - Iter 200, Minibatch Loss: 0.631044, Training Accuracy: 0.765625
[2021-11-30 20:29:13,914] ml_logger {train.py:142} INFO - Iter 300, Minibatch Loss: 0.551664, Training Accuracy: 0.882812
[2021-11-30 20:29:26,735] ml_logger {train.py:142} INFO - Iter 400, Minibatch Loss: 0.549120, Training Accuracy: 0.867188
[2021-11-30 20:29:39,694] ml_logger {train.py:142} INFO - Iter 500, Minibatch Loss: 0.454275, Training Accuracy: 0.898438
[2021-11-30 20:29:52,496] ml_logger {train.py:142} INFO - Iter 600, Minibatch Loss: 0.379688, Training Accuracy: 0.875000
[2021-11-30 20:30:05,311] ml_logger {train.py:142} INFO - Iter 700, Minibatch Loss: 0.412862, Training Accuracy: 0.859375
[2021-11-30 20:30:18,100] ml_logger {train.py:142} INFO - Iter 800, Minibatch Loss: 0.425813, Training Accuracy: 0.851562
[2021-11-30 20:30:30,965] ml_logger {train.py:142} INFO - Iter 900, Minibatch Loss: 0.254271, Training Accuracy: 0.906250
[2021-11-30 20:30:43,681] ml_logger {train.py:142} INFO - Iter 1000, Minibatch Loss: 0.312116, Training Accuracy: 0.906250
[2021-11-30 20:30:56,431] ml_logger {train.py:142} INFO - Iter 1100, Minibatch Loss: 0.272671, Training Accuracy: 0.906250
[2021-11-30 20:31:09,216] ml_logger {train.py:142} INFO - Iter 1200, Minibatch Loss: 0.405168, Training Accuracy: 0.914062
[2021-11-30 20:31:22,102] ml_logger {train.py:142} INFO - Iter 1300, Minibatch Loss: 0.311698, Training Accuracy: 0.921875
[2021-11-30 20:31:34,838] ml_logger {train.py:142} INFO - Iter 1400, Minibatch Loss: 0.291747, Training Accuracy: 0.914062
[2021-11-30 20:31:47,606] ml_logger {train.py:142} INFO - Iter 1500, Minibatch Loss: 0.243242, Training Accuracy: 0.921875
[2021-11-30 20:32:00,349] ml_logger {train.py:142} INFO - Iter 1600, Minibatch Loss: 0.280477, Training Accuracy: 0.921875
[2021-11-30 20:32:13,037] ml_logger {train.py:142} INFO - Iter 1700, Minibatch Loss: 0.290714, Training Accuracy: 0.921875
[2021-11-30 20:32:25,828] ml_logger {train.py:142} INFO - Iter 1800, Minibatch Loss: 0.381600, Training Accuracy: 0.882812
[2021-11-30 20:32:38,571] ml_logger {train.py:142} INFO - Iter 1900, Minibatch Loss: 0.145469, Training Accuracy: 0.960938
[2021-11-30 20:32:51,289] ml_logger {train.py:142} INFO - Iter 2000, Minibatch Loss: 0.177678, Training Accuracy: 0.929688
[2021-11-30 20:33:03,945] ml_logger {train.py:142} INFO - Iter 2100, Minibatch Loss: 0.253772, Training Accuracy: 0.906250
[2021-11-30 20:33:16,768] ml_logger {train.py:142} INFO - Iter 2200, Minibatch Loss: 0.229234, Training Accuracy: 0.937500
[2021-11-30 20:33:29,520] ml_logger {train.py:142} INFO - Iter 2300, Minibatch Loss: 0.205244, Training Accuracy: 0.953125
[2021-11-30 20:33:42,204] ml_logger {train.py:142} INFO - Iter 2400, Minibatch Loss: 0.251065, Training Accuracy: 0.929688
[2021-11-30 20:33:54,873] ml_logger {train.py:142} INFO - Iter 2500, Minibatch Loss: 0.244628, Training Accuracy: 0.937500
[2021-11-30 20:34:07,749] ml_logger {train.py:142} INFO - Iter 2600, Minibatch Loss: 0.281366, Training Accuracy: 0.906250
[2021-11-30 20:34:20,482] ml_logger {train.py:142} INFO - Iter 2700, Minibatch Loss: 0.228194, Training Accuracy: 0.937500
[2021-11-30 20:34:33,139] ml_logger {train.py:142} INFO - Iter 2800, Minibatch Loss: 0.145047, Training Accuracy: 0.953125
[2021-11-30 20:34:45,825] ml_logger {train.py:142} INFO - Iter 2900, Minibatch Loss: 0.317225, Training Accuracy: 0.898438
[2021-11-30 20:34:58,568] ml_logger {train.py:142} INFO - Iter 3000, Minibatch Loss: 0.153843, Training Accuracy: 0.937500
[2021-11-30 20:35:11,371] ml_logger {train.py:142} INFO - Iter 3100, Minibatch Loss: 0.157880, Training Accuracy: 0.929688
[2021-11-30 20:35:24,044] ml_logger {train.py:142} INFO - Iter 3200, Minibatch Loss: 0.264375, Training Accuracy: 0.929688
[2021-11-30 20:35:36,784] ml_logger {train.py:142} INFO - Iter 3300, Minibatch Loss: 0.277210, Training Accuracy: 0.914062
[2021-11-30 20:35:49,537] ml_logger {train.py:142} INFO - Iter 3400, Minibatch Loss: 0.174850, Training Accuracy: 0.953125
[2021-11-30 20:36:02,298] ml_logger {train.py:142} INFO - Iter 3500, Minibatch Loss: 0.169467, Training Accuracy: 0.960938
[2021-11-30 20:36:14,964] ml_logger {train.py:142} INFO - Iter 3600, Minibatch Loss: 0.193126, Training Accuracy: 0.929688
[2021-11-30 20:36:27,694] ml_logger {train.py:142} INFO - Iter 3700, Minibatch Loss: 0.084758, Training Accuracy: 0.984375
[2021-11-30 20:36:40,386] ml_logger {train.py:142} INFO - Iter 3800, Minibatch Loss: 0.180610, Training Accuracy: 0.945312
[2021-11-30 20:36:53,199] ml_logger {train.py:142} INFO - Iter 3900, Minibatch Loss: 0.231963, Training Accuracy: 0.921875
[2021-11-30 20:37:05,920] ml_logger {train.py:142} INFO - Iter 4000, Minibatch Loss: 0.103248, Training Accuracy: 0.960938
[2021-11-30 20:37:18,646] ml_logger {train.py:142} INFO - Iter 4100, Minibatch Loss: 0.191707, Training Accuracy: 0.953125
[2021-11-30 20:37:31,344] ml_logger {train.py:142} INFO - Iter 4200, Minibatch Loss: 0.168183, Training Accuracy: 0.953125
[2021-11-30 20:37:44,135] ml_logger {train.py:142} INFO - Iter 4300, Minibatch Loss: 0.267030, Training Accuracy: 0.890625
[2021-11-30 20:37:56,834] ml_logger {train.py:142} INFO - Iter 4400, Minibatch Loss: 0.131411, Training Accuracy: 0.976562
[2021-11-30 20:38:09,525] ml_logger {train.py:142} INFO - Iter 4500, Minibatch Loss: 0.142892, Training Accuracy: 0.953125
[2021-11-30 20:38:22,207] ml_logger {train.py:142} INFO - Iter 4600, Minibatch Loss: 0.195141, Training Accuracy: 0.937500
[2021-11-30 20:38:34,867] ml_logger {train.py:142} INFO - Iter 4700, Minibatch Loss: 0.153777, Training Accuracy: 0.953125
[2021-11-30 20:38:47,663] ml_logger {train.py:142} INFO - Iter 4800, Minibatch Loss: 0.113819, Training Accuracy: 0.968750
[2021-11-30 20:39:00,328] ml_logger {train.py:142} INFO - Iter 4900, Minibatch Loss: 0.301655, Training Accuracy: 0.890625
[2021-11-30 20:39:12,972] ml_logger {train.py:142} INFO - Iter 5000, Minibatch Loss: 0.274004, Training Accuracy: 0.929688
[2021-11-30 20:39:25,649] ml_logger {train.py:142} INFO - Iter 5100, Minibatch Loss: 0.210297, Training Accuracy: 0.929688
[2021-11-30 20:39:38,470] ml_logger {train.py:142} INFO - Iter 5200, Minibatch Loss: 0.105259, Training Accuracy: 0.960938
[2021-11-30 20:39:51,185] ml_logger {train.py:142} INFO - Iter 5300, Minibatch Loss: 0.246521, Training Accuracy: 0.921875
[2021-11-30 20:40:03,850] ml_logger {train.py:142} INFO - Iter 5400, Minibatch Loss: 0.135715, Training Accuracy: 0.945312
[2021-11-30 20:40:16,551] ml_logger {train.py:142} INFO - Iter 5500, Minibatch Loss: 0.236003, Training Accuracy: 0.890625
```

## 验证过程<a name="section1465595372416"></a>

1. 通过“模型训练”中的测试指令启动测试。

2. 完整验证过程的输出日志存储在./log/eval文件夹下，分别包含GRU单元与LSTM单元在完成训练之后，读取ckpt文件进行验证的accuracy日志

3. 下面是验证过程中的部分日志输出

```
================ Evaluation Process ================
Extracting ./dataset/train-images-idx3-ubyte.gz
Extracting ./dataset/train-labels-idx1-ubyte.gz
Extracting ./dataset/t10k-images-idx3-ubyte.gz
Extracting ./dataset/t10k-labels-idx1-ubyte.gz
==> Building a dRNN with GRU cells
Building layer: multi_dRNN_dilation_1, input length: 784, dilation rate: 1, input dim: 1.
=====> Input length for sub-RNN: 784
Building layer: multi_dRNN_dilation_2, input length: 784, dilation rate: 2, input dim: 20.
=====> Input length for sub-RNN: 392
Building layer: multi_dRNN_dilation_4, input length: 784, dilation rate: 4, input dim: 20.
=====> Input length for sub-RNN: 196
Building layer: multi_dRNN_dilation_8, input length: 784, dilation rate: 8, input dim: 20.
=====> Input length for sub-RNN: 98
Building layer: multi_dRNN_dilation_16, input length: 784, dilation rate: 16, input dim: 20.
=====> Input length for sub-RNN: 49
Building layer: multi_dRNN_dilation_32, input length: 784, dilation rate: 32, input dim: 20.
=====> 16 time points need to be padded. 
=====> Input length for sub-RNN: 25
Building layer: multi_dRNN_dilation_64, input length: 784, dilation rate: 64, input dim: 20.
=====> 48 time points need to be padded. 
=====> Input length for sub-RNN: 13
Building layer: multi_dRNN_dilation_128, input length: 784, dilation rate: 128, input dim: 20.
=====> 112 time points need to be padded. 
=====> Input length for sub-RNN: 7
Building layer: multi_dRNN_dilation_256, input length: 784, dilation rate: 256, input dim: 20.
=====> 240 time points need to be padded. 
=====> Input length for sub-RNN: 4
[2021-11-30 21:39:47,657] ml_logger {test.py:55} INFO - ==> Building a dRNN with LSTM cells
[2021-11-30 21:40:11,806] tensorflow {saver.py:1284} INFO - Restoring parameters from ./checkpoints_npu/LSTM/best_model.ckpt
===== Start Testing =====
[2021-11-30 21:45:39,999] ml_logger {test.py:98} INFO - ========> Testing Accuarcy: 0.952700
[2021-11-30 21:50:10,851] ml_logger {test.py:98} INFO - ========> Testing Accuarcy: 0.952700
```