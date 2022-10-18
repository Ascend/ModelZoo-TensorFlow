# README

## 1、关于项目
OOD是一个细菌检测网络，其主要特点是提出Likelihood Ratios并用以提高网络对于分布外细菌检测的准确性

论文： [paper](https://arxiv.org/abs/1906.02845) 

论文源代码： [code](https://github.com/google-research/google-research/tree/master/genomics_ood) 

## 2、代码结构
```


├── config               
│   ├── om_in.sh        #msame推理in_val_data
│   ├── om_ood.sh        #msame推理ood_val_data
│   └── pb2om.sh         #act命令
├── log
│         ├── bin_data  #存放转化为的bin格式的数据
│             ├── in_val_data  #分布内数据 
│             └── ood_val_data #分布外数据
│         ├── ckpt_file  #存放ckpt文件
│         ├── frozen_pb_file      #存放冻结后的pb模型
│         ├── om_file    #存放om模型
│         ├── original_data #存放原始数据
│             ├── between_2011-2016_in_val  #分布内数据 
│             └── between_2011-2016_ood_val #分布外数据
│         ├── pb_file    #存放pb模型
│         └── result        #存放离线推理结果的文件夹
├── auroc.py           #根据推理结果计算模型最终指标AUROC 
├── ckpt2pb.py           #将ckpt文件转化为pb文件 
├── data2bin.py     #将原始数据转化为bin格式
├── generative.py     #网络结构，用于ckpt转pb
├── LICENSE
├── modelzoo_level.txt
├── msame                 #msame,具体的环境需要自己编译
├── README.md
├── requirements.txt     #需要安装的第三方库文件
└── utils.py      


```
## 3、关于测试集和模型

测试集放在./log/original_data/下。
测试数据集和模型文件都放在百度网盘上：[百度网盘链接](https://pan.baidu.com/s/1wDSn-rkcyE2Hjr6lQCxw9w?pwd=mqpt) 
提取码：mqpt
百度网盘中对应目录下也包含已成功转换的pb、om等模型及bin数据文件
## 4、pb模型

原始ckpt文件下载后，放在路径log/ckpt_file/目录下，使用的是ckpt-218000。执行以下命令转换为pb模型并生成冻结后的pb模型。

```
python3 ckpt2pb.py 
```
ckpt2pb.py中固定了pb模型(且用于pb转om)的输出目录：./log/pb_file/ \
冻结后的pb模型的输出目录：./log/frozen_pb_file/


## 5、生成om模型。

使用atc命令将pb模型转换为om模型，执行以下命令转换为om模型。

```
bash ./config/pb2om.sh
```

注意配置对应的文件路径,soc_version默认是Ascend910

## 6、测试集内文件转换为bin文件

执行以下命令将测试集数据经过处理转换为用于网络输入的bin文件。data2bin.py文件中固定了生成bin文件存放的目录为./log/bin_data。
测试数据集分为分布内数据（存放目录为./log/original_data/between_2011-2016_in_val）和分布外数据（存放目录为./log/original_data/between_2011-2016_in_val）
这两部分数据均包含10,000个样本，每100个样本生成一个.bin文件。
使用如下命令转换分布内数据
```
python3 data2bin.py  --in_val_data=True --out_dir ./log/bin_data/in_val_data
```
使用如下命令转换分布外数据
```
python3 data2bin.py  --in_val_data=False --out_dir ./log/bin_data/ood_val_data
```
转换得到的bin文件分别存放于目录./log/bin_data/in_val_data（分布内数据）和./log/bin_data/ood_val_data（分布外数据）
## 7、使用om模型进行推理

使用msame工具进行推理。参考[msame简介](https://gitee.com/ascend/tools/tree/master/msame)，参考命令如下。
推理分布内数据

```
bash config/om_in.sh
```
推理得到结果存放在目录./log/result,注意将结果目录更名为in_val_result,否则须在auroc.py中更改文件路径

推理分布外数据
```
bash config/om_ood.sh
```
推理得到结果存放在目录./log/result,注意将结果目录更名为ood_val_result,否则须在auroc.py中更改文件路径
注意配置对应的文件路径


## 8、om模型离线推理性能

每一百个样本推理的平均运行性能为858.35 ms。


## 9、om模型离线推理结果对比NPU及GPU

执行以下命令得到模型衡量指标AUROC数值

```
 python3 auroc.py
```
|   | 论文 | GPU | NPU | om |
|-------|------|------|------|------|
| AUROC | 0.626 | 0.677 | 0.641 | 0.665 |

