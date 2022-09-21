# voxelmorph
voxelmorph推理部分实现，模型概述详情请看[TensorFlow/contrib/cv/voxelmorph_ID2120_for_TensorFlow · Ascend/ModelZoo-TensorFlow - 码云 - 开源中国 (gitee.com)](https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/voxelmorph_ID2120_for_TensorFlow)

## 训练环境

* TensorFlow 1.15.0
* Python 3.7.5

## 代码及路径解释

```
voxelmorph_ID2120_for_ACL
├─ ext						外部依赖库文件夹
├─ models					存放ckpt，pb，om模型的文件夹
├─ output					om推理输出文件夹
|
├─ ckpt2pb.py				ckpt模型固化为pb
├─ pb2om.sh					pb转om脚本
├─ data2bin.py				数据转换为bin形式批量程序
├─ omrun.sh					om推理脚本
├─ omtest.py				om推理结果测试程序
├─ datagenerators.py		测试代码依赖：导入数据程序
├─ networks.py				测试代码依赖：voxelmorph网络py程序
└─ README.md		            
```


## 数据集
* Dataset-ABIDE

下载地址：`obs://voxelmorph-zyh/Dataset-ABIDE/`

## 模型文件
包括初始ckpt文件，固化pb文件，以及推理om文件
链接：

## pb模型

模型固化
```shell
python3 ckpt2pb.py
```
## 生成om模型

使用ATC模型转换工具进行模型转换时可参考脚本 `pb2om.sh`, 指令可参考:
```shell
${atc_path}/atc --input_shape="input_src:1,160,192,224,1;input_tgt:1,160,192,224,1" \
--out_nodes="spatial_transformer/map/TensorArrayStack/TensorArrayGatherV3:0;flow/BiasAdd:0" \
--output="./models/vm" \
--soc_version=Ascend310 --framework=3 --model="./models/frozen_model.pb" 
```
具体参数使用方法请查看官方文档。

## 将测试集图片转为bin文件

```shell
python3 data2bin.py
```
## 使用msame工具推理

使用msame工具进行推理时可参考脚本`omrun.sh`，指令可参考：
```shell
${msame_path}/msame --model ./models/vm.om \
--input ${input_src},${input_tgt} --output ./output \
--outputSize "1000000000, 1000000000"  --outfmt BIN --loop 1 --debug true
```
参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

## 使用推理得到的bin文件进行推理
```shell
python3 omtest.py
```

## 推理精度
Ascend910模型精度，推理与NPU结果相同：

|                                          | 论文精度     | GPU精度      | NPU精度      | 推理精度     |
| :--------------------------------------: | ------------ | ------------ | ------------ | ------------ |
| DICE系数（[0, 1], 1 最优）/ 均值(标准差) | 0.752(0.140) | 0.708(0.133) | 0.703(0.134) | 0.703(0.134) |
