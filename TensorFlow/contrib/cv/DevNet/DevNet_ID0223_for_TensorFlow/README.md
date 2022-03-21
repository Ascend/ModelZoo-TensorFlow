# DevNet: An End-to-end Anomaly Score Learning Network
模型出处：
By Guansong Pang, Chunhua Shen, Anton van den Hengel. Deep anomaly detection with deviation networks (KDD19).

## 提交文件夹说明：
```shell
├── dataset                                         #数据集
├── om                                              #离线推理
├── images                                          #README.md中使用的图片
├── MA_LOG                                          #Pycharm Toolkit在昇腾上的训练日志
├── model                                           #保存的ckpoint
├── README.md                                       #README.md
├── results                                         #模型结果输出
├── LICENSE                                         #LICENSE文件
├── train.py                                        #训练启动脚本
├── eval.py                                         #验证启动脚本
├── devnet.py                                       #网络定义脚本
├── export.py                                       #ckpt转pb
└── utils.py                                        #相关助手函数

```
## 代码使用说明
通过终端执行训练脚本
```python
python train.py 
```
通过终端执行验证脚本
```python
python eval.py
```

在使用Pycharm Toolkit在ModelArts平台上训练

![](./images/pycharm-toolkit.jpg)
```python
python main.py --data_url=/data/url/path --train_url=/train/url/path --network_depth=2 --runs=10 --known_outliers=30 --cont_rate=0.02 --data_format=0 --output=./results.csv --dataset=`annthyroid_21feat_normalised`
```

## 相关环境
* Ascend TensorFlow 1.15


## 论文来源
[ACM Portal](https://dl.acm.org/citation.cfm?id=3330871) or [arXiv](https://arxiv.org/abs/1911.08623)

## 数据集
[ADRepository](https://github.com/GuansongPang/anomaly-detection-datasets) 

## 论文引用
>Guansong Pang, Chunhua Shen, and Anton van den Hengel. "Deep anomaly detection with deviation networks." In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pp. 353-362. 2019.

## 模型精度
目标数据集:`annthyroid_21feat_normalised`

AUC-ROC和AUC-PR的值越高，代表模型的效果越好。NPU和GPU超参设置一样，10轮训练后取平均值，均比论文中效果要好
|Method |  AUC-ROC | AUC-PR |
|:-:|:-:|:-:|
|Paper|0.783|0.274|
|This code NPU|**0.9107**|**0.7333**|
|This code GPU|**0.9526**|**0.7979**|

论文中的精度：

![](./images/performance.jpg)
