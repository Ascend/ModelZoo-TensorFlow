# AFM: Attentional Factorization Machine
模型出处：
By Guansong Pang, Chunhua Shen, and Anton van den Hengel. Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks (IJCAI17).

## 提交文件夹说明：
```shell
├── code                           #代码集
    ├──LoadData.py                 #数据集载入文件
    ├──FM.py                       #FM训练及预测文件
    ├──AFM.py                      #AFM训练及预测文件
    └──ckpt2pb.py                  #ckpt转pb
├── data                           #数据集
├── om                             #离线推理
├── pretrain                       #保存的ckpoint
├── results                        #模型结果输出
├── images                         #README.md中使用的图片
├── LICENSE                        #LICENSE文件
└── README.md                      #README.md

```
## 代码使用说明
通过终端执行训练脚本
```python
python3 code/AFM.py --dataset ml-tag --epoch 100 --pretrain -1 --batch_size 4096 --hidden_factor0 8 --hidden_factor1 256 --keep0 1.0 --keep1 0.5 --lamda_attention 2 --lr 0.045 --process train --train_url pretrain/  --data_url data/
```
通过终端执行验证脚本
```python
python3 code/AFM.py --dataset ml-tag --epoch 100 --pretrain -1 --batch_size 4096 --hidden_factor0 8 --hidden_factor1 256 --lr 0.045 --keep0 1.0 --keep1 0.5 --lambda_attention 2 --train_url ../pretrain/  --data_url ../data/ --process evaluate
```

在使用Pycharm Toolkit在ModelArts平台上训练

![](./images/pycharm-toolkit.jpg)
```python
Image Path:swr.cn-north-4.myhuaweicloud.com/mat_mul_replace/roma-tensorflow-ascend910-c75-cp37-euleros2.8-aarch64-training:1.15.0-2.0.8_base
Boot Command:/bin/bash run_train.sh 's3://afm-end/afm-2021-1-22-20/code2' 'code2/AFM_2021_1_2NPU.py' '/tmp/log/demo.log'
OBS Path:/afm-end/afm-2021-1-22-20/
Data Path in OBS:/afm-end/afm-2021-1-22-20/
command: code/AFM_2021_1_2NPU.py --data_url s3://afm-end/afm-2021-1-22-20/ --train_url s3://afm-end/afm-2021-1-22-20/AFM_end/output/V0030/
```

## 相关环境
* Ascend TensorFlow 1.15


## 论文来源
[arXiv](https://arxiv.org/abs/1708.04617) or [作者个人主页](http://staff.ustc.edu.cn/~hexn/papers/ijcai17-afm.pdf)

## 数据集
[ml-tag](https://github.com/hexiangnan/attentional_factorization_machine/tree/master/data/ml-tag) 

## 论文引用
>Jun Xiao, Hao Ye, Xiangnan He, Hanwang Zhang, Fei Wu and Tat-Seng Chua (2017). Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks IJCAI, Melbourne, Australia, August 19-25, 2017.

## 模型精度
目标数据集:`ml-tag`

RMSE值越小，代表模型的效果越好。
|Method |  RMSE |
|:-:|:-:|
|Paper|0.4325|
|This code NPU|**0.4684**|
|This code CPU|**0.4684**|

论文中的精度：

![](./images/paper_rmse.jpg)