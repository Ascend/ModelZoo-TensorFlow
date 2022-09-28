### **模型参考文献**

《HybridSN: Exploring 3-D–2-D CNN Feature Hierarchy for Hyperspectral Image Classification》

### **参考实现**

[modelzoo: Ascend Model Zoo - Gitee.com](https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/HybridSN_ID1160_for_TensorFlow)

### **训练环境准备**

硬件环境：Ascend: 1*Ascend 910(32GB) | ARM: 24 核 96GB，

软件环境：TensorFlow1.15

### **数据集准备**

自行下载以下数据集，均为开源可下载数据集Indian Pines (IP) dataset，University of Pavia (PU) dataset，Salinas Scene (SA) dataset

其中，代码中自行划分训练集和测试集。

### **脚本和事例代码**

```
--run_file.py                           //pycharm在ModelArts训练的定义脚本
--npu_train.sh                         //单卡运行脚本
--Hybrid-Spectral-Net.py               //HybridSN网络训练及测试代码
```

### **脚本参数**

```
--epoch       Epoch to train, default 100
--batch_size  batch size, default 256
--data_path   dataset path
```



### **训练过程**

通过变量dataset选择不同数据集进行训练。

### **训练结果**
1. GPU与NPU精度对比


| 数据集                  |           | 论文精度  |           |       | GPU精度 |       |       | NPU精度 |       |
| ----------------------- | --------- | --------- | --------- | ----- | ------- | ----- | ----- | ------- | ----- |
|                         | OA        | Kappa     | AA        | OA    | Kappa   | AA    | OA    | Kappa   | AA    |
| Indian Pines (IP)       | 99.75±0.1 | 99.71±0.1 | 99.63±0.1 | 99.69 | 99.65   | 99.79 | 99.65 | 99.61   | 99.64 |
| University of Pavia(PU) | 99.98±0.0 | 99.98±0.0 | 99.97±0.0 | 99.96 | 99.95   | 99.91 | 99.98 | 99.97   | 99.92 |
| Salinas Scene (SA)      | 100±0.0   | 100±0.0   | 100±0.0   | 100   | 100     | 100   | 100   | 100     | 100   |
2. GPU与NPU性能对比

以每个epoch耗时作为性能指标，GPU每个epoch耗时相同，单NPU由于第一个epoch需要加载数据，因此耗时较长，因此在这里我们记录的是NPU上训练的第二个epoch开始的耗时作为性能指标。
| 数据集                     | GPU性能 | NPU性能 |
|-------------------------|-------|-------|
| Indian Pines（IP）        | 70s   | 8s    |
| University of Pavia（PU） | 65s   | 36s   |
| Salinas Scene（SA）       | 80s   | 40s   |










