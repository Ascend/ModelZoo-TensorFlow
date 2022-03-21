# MicroExpNet

## 目录

1. [介绍](#介绍)
2. [Citation](#citation)
3. [API](#api)
4. [Models](#models)

## 介绍

MicroExpNet 是一个非常小 (不到 1MB) and 快速 (1851 FPS on i7 CPU) [TensorFlow](https://www.tensorflow.org/) 用于从正面人脸图像进行面部表情识别 (FER) 的卷积神经网络模型.其通过知识蒸馏的方式，从InceptionV3学习对于面部表情的识别。

 - 原文链接：https://arxiv.org/abs/1711.07011v4

##使用ACT转化OM模型
命令行示例：

`
atc --model=/root/microexpnet/ckmodel.pb --framework=3 
--output=/root/microexpnet/ckmodel --soc_version=Ascend310 --input_shape="input:1,7056" 
--log=info --out_nodes="ArgMax:0" --precision_mode=allow_fp32_to_fp16 
--op_select_implmode=high_precision
`

##使用msame工具推理
参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。
获取到msame可执行文件之后，将待检测om文件放在model文件夹。注意，输入文件必须为float32类型，可
在预处理完数据集之后，使用img.tofile('xxx.bin')即可

命令行示例:

`
./msame --model /root/microexpnet/ckmodel.om --input "/root/microexpnet/out" 
--output "/root/microexpnet/out/result" --outfmt TXT
`