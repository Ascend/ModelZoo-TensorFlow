# NMT for ACL
## 目的
- 基于seq2seq的机器翻译
## 环境安装依赖
- 参考《驱动和开发环境安装指南》安装NPU环境
- python依赖库：numpy six
- TensorFlow:1.15
- 适用于Ascend310/Ascend310P3 
## 数据集及模型准备
NMT训练、推理代码参考如下：

[训练、推理代码](https://github.com/tensorflow/nmt/)

获取在线推理的数据集目录下的文件，拷贝到./dataset/目录下，使用的tst2013数据集用作推理数据集。

pb模型：获取NMT转pb的pb模型，将pb模型放在model目录下。

如需checkpoint转pb，请先将脚本依赖的训练、推理代码放到对应位置，checkpoint转pb参考脚本：
```shell
cd freeze_pb
python3 freeze_pb.py
```

## 前处理脚本
请将数据集单词转成对应索引，具体参考preprocess,请创建一个data目录及src_ids,src_len两个子目录用来存放输入bin文件
```
cd dataset
mkdir data
cd data
mkdir src_ids
mkdir src_len
cd ../../preprocess
python3 preprocess.py
```

## 模型转换
len表示数据集中最大句子长度，请根据数据集修改

[pb模型下载链接](https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/modelzoo/Research/nlp/nmt.pb)
```
cd model
./convert_om.sh
```

## 离线推理
首次请编译xacl，具体请见xacl_fmk中 请创建outputs目录用来存放输出bin文件，每次推理前请清空outputs目录
```
mkdir outputs
cd xacl_fmk/out
./xacl_fmk -m ../../model/nmt_1268.om -i ../../dataset/data/src_ids,../../dataset/data/src_len -o ../../outputs/ -w 496
```

## 后处理脚本
请将脚本所依赖的训练、推理代码放到对应位置,精度结果存放在accuracy_file中
```
python3 postprocess
```
