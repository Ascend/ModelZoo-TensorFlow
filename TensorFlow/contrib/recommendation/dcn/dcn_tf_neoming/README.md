# DCN
模型出处：[Deep & Cross Network for Ad Click Predictions](https://github.com/ZiyaoGeng/Recommender-System-with-TF2.0/tree/master/DCN)
## 提交文件说明
```bash
├── checkpoints                                 # 保存的checkpoints文件
│   └── keras                                   # 最好的结果，提供百度云网盘链接
│       ├── checkpoint
│       ├── dcn_weights.data-00000-of-00002
│       ├── dcn_weights.data-00001-of-00002
│       └── dcn_weights.index
├── data_process                                # 数据预处理
├── dataset                                     # 数据集，提供百度云网盘链接
│   └──  Criteo
│       ├── demo.txt
│       └── train.txt
├── dcn                                         # dcn模型定义                    
├── eval.py                                     # 验证脚本
├── README.md                                   # 本文档
├── om											# 离线推理
├── pic.png                                     # 论文精度
└── train.py                                    # 训练脚本
```
### 数据集链接
模型训练使用Criteo数据集，数据集请用户自行获取。

数据集训练前需要做预处理操作，请用户参考GPU开源链接,将数据集封装为tfrecord格式。

### 模型ckpt链接
ckechpoints 链接: https://pan.baidu.com/s/19XqnkLyR50-9V7B2NL3Hsw  密码: dht4

## 代码使用说明
下载好数据集之后,使用如下命令进行训练
```bash
python train.py
```
***
下载好ckpt之后,使用如下命令进行验证

```bash
python eval.py
```
`eval.py`中`53`行会根据路径读取checkpoints并进行推理
```python
model.load_weights('checkpoints/keras/dcn_weights')
```
## 实验超参与精度

###  实验超参数

- file：Criteo文件；
- read_part：是否读取部分数据，`True`；
- sample_num：读取部分时，样本数量，`5000000`；
- test_size：测试集比例，`0.2`；
- 
- embed_dim：Embedding维度，`8`；
- dnn_dropout：Dropout, `0.5`；
- hidden_unit：DNN的隐藏单元，`[1024, 1024]`；
- 
- learning_rate：学习率，`0.001`；
- batch_size：`4096`；
- epoch：`10`；



### 实验精度
提供的checkpoints得到的精度
`AUC: 0.805587, loss: 0.4492`


论文中只提供了logloss,误差在0.01以内

![](pic.png)