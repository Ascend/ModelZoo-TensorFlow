# Prototypical-Networks: Prototypical Networks for Few-shot Learning

本文为小样本分类问题提出了原型网络。原型网络的思路非常简单：对于分类问题，原型网络将其看做在语义空间中寻找每一类的原型中心。针对Few-shot Learning的任务定义，原型网络训练时学习如何拟合中心。学习一个度量函数，该度量函数可以通过少量的几个样本找到所属类别在该度量空间的原型中心。测试时，Support Set中的样本来计算新的类别的聚类中心，再利用最近邻分类器的思路进行预测。本文主要针对Few-Show／Zero-Shot任务中过拟合的问题进行研究，将原型网络和聚类联系起来，和目前的一些方法进行比较，取得了不错的效果。

<img src="https://gitee.com/phoebe0507/img_gallery/raw/master/readme/prototypical-networks.png" style="zoom:67%;" />

原型模型请参考[GitHub链接](https://github.com/abdulfatir/prototypical-networks-tensorflow/blob/master)，NPU迁移代码请参考[Gitee链接] (https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/PROTOTYPICAL-NETWORKS_ID1286_for_TensorFlow)




## 代码路径解释

```
.
|-- LICENSE
|-- __init__.py
|-- checkpoint            ----存放ckpt文件
|   |-- log               ----存放pb模型文件
|-- data                  ----数据集目录
|   |-- omniglot          ----omniglot数据集
|   |   |-- data          ----数据集存放
|   |   |-- splits        ----数据集读取序列
|-- input                 ----输入Bin文件目录
|   |-- inputx            ----inputx输入数据(Support Set)
|   |-- inputq            ----inputq输入数据(Query Set)
|   |-- inputy            ----inputy输入数据(Labels)
|-- data.py               ----数据处理
|-- pre_process.py        ----数据预处理成bin文件
|-- model.py              ----网络模型
|-- frozen_graph.py       ----ckpt转pb模型代码
|-- pb_test.py            ----pb模型测试代码(使用data.py输入数据)
|-- pb_test_bin.py        ----pb模型测试代码(使用bin文件输入数据)
|-- avg_acc.py            ----推理平均精度
```



## 冻结pb模型

将自己训练的ckpt生成pb模型：

```
python frozen_graph.py
```



## om模型

使用ATC模型转换工具进行模型转换时可以参考如下指令:

```
atc --model=$HOME/pb_model/proto_model.pb --framework=3 --output=$HOME/pb_model/out/tf_prototypical --input_shape="inputx:20,5,28,28,1;inputq:20,15,28,28,1;inputy:20,15" --output_type=FP32 --soc_version=Ascend310 
```

具体操作详情和参数设置可以参考

- [ATC快速入门_昇腾CANN社区版(5.0.3.alpha002)(推理)_ATC模型转换_华为云](https://support.huaweicloud.com/atctool-cann503alpha2infer/atlasatc_16_0005.html)



## 模型获取

ominglot数据集获取链接：[百度网盘](https://pan.baidu.com/s/1l7gAEWIGryn1WxIccexVPg)  提取码：7bo6

ckpt文件、pb模型、om模型及输入bin文件获取链接：[百度网盘](https://pan.baidu.com/s/1N1RpiVH26sj04t7lcEvwVQ)  提取码：l6jz



## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，将待检测om文件放在model文件夹，然后进行性能测试。



## 性能测试

使用msame推理工具，参考如下命令，发起推理性能测试：

```
./msame --model $HOME/pb_model/tf_prototypical.om --output $HOME/pb_model/out/output1 --outfmt TXT
```

Batch: 1 

inputx_shape: 20,5,28,28,1  

inputq_shape: 20,15,28,28,1

inputy_shape: 20,15

推理性能 18.975ms

![](https://gitee.com/phoebe0507/img_gallery/raw/master/readme/offline.png)

## 精度测试

- 生成数据

  生成的bin文件存储在当前目录的input/inputx、input/inputq、input/inputy文件夹下；或从模型获取部分提到的网盘链接中直接下载处理好的输入bin文件”input.tar.gz“进行解压。

```
python pre_process.py
```

- om模型推理

```
./msame --model $HOME/pb_model/out/tf_prototypical.om --input $HOME/pb_model/input/inputx,$HOME/pb_model/input/inputq,$HOME/pb_model/input/inputy --output $HOME/pb_model/out/output1 --outfmt TXT
```

- 推理结果分析

  修改avg_acc.py文件中filename参数为上一步om推理输出的文件目录——$HOME/pb_model/out/output1/编号，执行如下指令：

```
python avg_acc.py
```



## 推理精度

|                   | 论文指标 | GPU V100 | NPU Ascend910 | 离线推理 |
| ----------------- | -------- | -------- | ------------- | -------- |
| 20-way 5-shot Acc | 98.9%    | 98.18%   | 98.27%        | 97.4%    |
