-   [基本信息](#基本信息.md)
-   [概述](#概述.md)
-   [模型性能](#模型性能.md)
-   [高级参考](#高级参考.md)

<h2 id="基本信息.md">基本信息</h2>

**发布者（Publisher）：Huawei**

**应用领域（Application Domain）：** Image Classification 

**版本（Version）：1.2**

**修改时间（Modified） ：2021.08.20**

**大小（Size）：60M**

**框架（Framework）：TensorFlow 1.15.0**

**模型格式（Model Format）：ckpt,pb,om**

**精度（Precision）：Mixed**

**处理器（Processor）：昇腾310**

**应用级别（Categories）：Demo**

**描述（Description）：基于TensorFlow框架的DilatedRNN图像分类网络推理代码** 

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

<h2 id="模型性能.md">模型性能</h2>

## 昇腾310芯片模型推理性能

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
| UpdateDate          | 08/20/2021 (month/day/year)                   |
| Tensorflow Version  | 1.15.0                                        |
| Dataset             | MNIST                                         |
| batch_size          | 1                                           |
| outputs             | probability                                   |
| Evaluation Accuracy | 93.14% (lstm)  93.16%(gru)                    |
| Speed               | 150 ms / test iteration                        |
| Total time          | 30 mins                                        |

### 提交模型网盘链接
- [网盘主链接](https://disk.pku.edu.cn/#/link/31C0C6776BC8256580B7744B9E7B240D?gns=EC2CFB9BD41C4D84934837420C6184C4%2FD5F05CDF31F6410EB1854AFE91B36DAB)
- [Ascend310环境 -- om模型](https://disk.pku.edu.cn/#/link/31C0C6776BC8256580B7744B9E7B240D?gns=EC2CFB9BD41C4D84934837420C6184C4%2FD5F05CDF31F6410EB1854AFE91B36DAB%2FF1F2252BC84C4BE58AEB4395CBEBBCF4)
- [数据集](https://disk.pku.edu.cn/#/link/31C0C6776BC8256580B7744B9E7B240D?gns=EC2CFB9BD41C4D84934837420C6184C4%2FD5F05CDF31F6410EB1854AFE91B36DAB%2F9D387891B5334529BE14592FDC6DD601)

<h2 id="高级参考.md">高级参考</h2>

## 脚本和示例代码<a name="section08421615141513"></a>

```
├── README.md
├── LICENSE
├── modelzoo_level.txt
├── author.txt
├── om_eval.sh
├── imgs
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