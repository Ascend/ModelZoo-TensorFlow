# LSTM-HAR
LSTM-HAR推理部分实现，模型概述详情请看LSTM-HAR_ID2084_for_TensorFlow/README.md

## 训练环境

* TensorFlow 1.15.0
* Python 3.7.5
* sklearn 1.0.2

## 代码及路径解释

```
LSTM-HAR_ID2084_for_ACL
├── ckpt_pb.py                 ckpt模型固化为pb模型
├── ckpt_pb.sh                 ckpt模型固化为pb模型脚本文件
├── atc.sh  				   act工具：pb模型转换为om模型
├── msame.sh				   msame工具：om离线推理命令
├── data_bin.py			       推理数据生成：将测试集中的数据转换存储为bin文件
├── accuarcy.py                推理结果统计
├── accuarcy.sh                推理结果统计脚本
├── UCI HAR Dataset			   数据集位置		
│   └── ..			           评估数据集
├── input		               数据集转为bin存放位置		
│   └── ..			            
├── output		               msame推理结果bin文件存放位置		
    └── ..			            
```


## 数据集
* A Public Domain Dataset for Human Activity Recognition Using Smartphones

## 模型文件
包括初始ckpt文件，固化pb模型文件，以及推理om模型文件

初始ckpt模型文件： obs://cann-id2084/acl/model/

pb模型文件以及om模型文件： obs://cann-id2084/acl/

## 将测试集转为bin文件

```shell
python3.7.5 data_bin.py
```

## pb模型

模型固化
```shell
sh ckpt_pb.sh
```
## 生成om模型

使用ATC模型转换工具进行模型转换时可参考如下指令 atc.sh:
```shell
atc --model=./lstm_har.pb --framework=3 --output=./lstm_har --soc_version=Ascend310 --input_shape="X:1,128,9" --log=info --out_nodes="output:0"
```
具体参数使用方法请查看官方文档。

## 使用msame工具推理

使用msame工具进行推理时可参考如下指令 msame.sh
```shell
./msame --model lstm_har.om --input input/ --output output/
```
参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

## 使用推理得到的bin文件进行推理
```shell
python3.7.5 accuarcy.py output/ "./UCI HAR Dataset/test/y_test.txt"
```

## 精度
```shell
sh accuarcy.sh
```
* * Ascend310推理精度：90.23%

## 推理性能：

![Image](images/result.png)