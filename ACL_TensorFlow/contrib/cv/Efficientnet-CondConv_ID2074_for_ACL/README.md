# Efficientnet-CondConv
Condconv即有条件参数化卷积，为每个样本学习专有的卷积内核。用CondConv替换正常卷积，能够增加网络的大小和容量，同时保持有效的推理。参考文章为 CondConv: Conditionally Parameterized Convolutions for Efficient Inference   参考项目：https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/condconv.


## ckpt文件
1. 包含4个文件：checkpoint, ckpt-0000.data, ckpt-0000.index, ckpt-0000.meta
2. 文件链接：obs://acl-lhz/efficientnet-condconv/ckpt/


## pb模型
1. 将ckpt文件下载到对应的文件夹。
2. 运行ckpt2pb.py, 得到pb文件。
3. 文件链接：obs://acl-lhz/efficientnet-condconv/pb_model/condconv.pb



## om模型
atc转换命令参考：
```sh
Ascend/ascend-toolkit/latest/atc/bin/atc --input_shape="input:1,224,224,3" --input_format=NHWC --output="./condconv" --soc_version=Ascend310 --framework=3 --model="condconv.pb"
```
文件链接：obs://acl-lhz/condconv.om


##  编译msame推理工具
-参考https://gitee.com/ascend/tools/tree/master/msame, 编译出msame推理工具。


## 制作数据集：
1. 将图片按照文件夹分类，文件夹名称为类别编号。

2. 运行img2bin.py, 默认图片格式为.jpeg。
文件链接：obs://acl-lhz/efficientnet-condconv/bininput/


## 离线推理和精度计算
1. 执行命令得到bin格式的推理输出。
```sh
./out/msame --model="/usr/local/condconv.om" --input="/usr/local/bininput" --output="./" --outfmt BIN
```
2. 运行precision.py, 得到推理精度。


## 离线推理精度

|     |  论文精度	| GPU精度  |  NPU精度  | 推理精度|
|  ----     |  ----     |  ----    |   ----   |  ----  |
| top1_accuarcy   | 78.3% | 80.0%|80.9%|81.7%|
