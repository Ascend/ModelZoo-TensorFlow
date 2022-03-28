# 概述
    mean-teacher是一种用于图像分类的半监督学习方法，能够在拥有少量有标签数据的情况下训练出分类准确率很高的网络模型。

- 论文链接: [Weight-averaged consistency targets improve semi-supervised deep learning results](https://arxiv.org/abs/1703.01780)

- 官方代码仓: [链接](https://github.com/CuriousAI/mean-teacher/)

- 精度性能比较:

|  | 论文 | GPU | Ascend |
| ------ | ------ | -- | ------ |
| error | 12.3% | 13.5% | 14.6% |
| 性能(s/steps) |  | 1.17 | 0.30 |
# 环境
    - python 3.7.5
    - Tensorflow 1.15
    - Ascend910

# 训练
## 数据集
    使用./prepare_data.sh脚本预处理数据集
## 训练超参见train_cifar10.py参数列表
## 单卡训练命令
首先在脚本test/train_full_1p.sh中，配置train_steps、data_path等参数，请用户根据实际路径配置data_path，或者在启动训练的命令行中以参数形式下发

-启动训练
```commandline
bash train_full_1p.sh --data_path=../data
```

# 功能测试
少量step运行
```commandline
bash ./test/train_performance_1p.sh
```

# 模型固化
准备checkpoint,默认为 ./ckpt/checkpoint-40000
- 执行脚本,结果将保存在
```commandline
python3 freeze_graph.py
```
# 部分脚本和示例代码
```text
├── README.md                                //说明文档
├── requirements.txt			//依赖
├──test		                         //训练脚本目录								 
│    ├──train_performance_1p.sh			 
│    ├──train_full_1p.sh
├──train_cifar10.py                 	     //训练脚本
|——freeze_graph.py              //固化脚本
```
# 输出
模型存储路径为test/output/ASCEND_DEVICE_ID，包括训练的log以及checkpoints文件。loss信息在文件test/output/{ASCEND_DEVICE_ID}/train_${ASCEND_DEVICE_ID}.log中。 模型固化输出为pb_model/milking_cowmask.pb
