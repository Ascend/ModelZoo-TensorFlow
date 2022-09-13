# MIXNET
mixnet即混合深度卷积，在一次卷积中自然地混合多个卷积核大小，大内核提取高级语义信息，小内盒提取位置边缘信息，以此获得更好的精度和效率。参考文章为 MixConv: Mixed Depthwise Convolutional Kernels   参考项目： https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet/mixnet.


## ckpt文件
1. 包含4个文件：checkpoint, ckpt-0000.data, ckpt-0000.index, ckpt-0000.meta
2. 文件链接：obs://acl-lhz/mixnet/ckpt/


## pb模型
1. 将ckpt文件下载到对应的文件夹。
2. 运行ckpt2pb.py, 得到pb文件。
3. 文件链接：obs://acl-lhz/mixnet/pb_model/mixnet.pb



## om模型
1. 注意在操作前获得pb文件权限:
```sh
chmod 777 mixnet.pb
```
2.atc转换命令参考：
```sh
Ascend/ascend-toolkit/latest/atc/bin/atc --input_shape="input:1,224,224,3" --input_format=NHWC --output="./mixnet" --soc_version=Ascend310 --framework=3 --model="mixnet.pb"
```
3. 文件链接：obs://acl-lhz/mixnet.om


##  编译msame推理工具
-参考https://gitee.com/ascend/tools/tree/master/msame, 编译出msame推理工具。


## 制作数据集：
1. 将图片按照文件夹分类，文件夹名称为类别编号。

2. 运行img2bin.py, 默认图片格式为.jpeg。
文件链接：obs://acl-lhz/mixnet/bininput/


## 离线推理和精度计算
1. 注意在操作前获得om文件权限:
```sh
chmod 777 mixnet.om
```
2. 执行命令得到bin格式的推理输出。
```sh
./out/msame --model="/usr/local/mixnet.om" --input="/usr/local/bininput" --output="./" --outfmt BIN
```
3. 运行precision.py, 得到推理精度。


## 离线推理精度

|     |  论文精度	| GPU精度  |  NPU精度  | 推理精度|
|  ----     |  ----     |  ----    |   ----   |  ----  |
| top1_accuarcy   | 73.8% | 74.1%|74.3%|75.1%|
