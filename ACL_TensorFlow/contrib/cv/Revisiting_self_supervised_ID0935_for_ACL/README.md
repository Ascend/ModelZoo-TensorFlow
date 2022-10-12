# README

## 1、关于项目
自我监督技术最近成为了无监督的有效解法之一，然而目前对于自监督的研究大量集中在pretext任务，而如卷积神经网络(CNN)的选择，却没有得到同等的重视。因此，作者回顾了之前提出的许多自我监督模型，进行了一项彻底的大规模研究，结果发现了多个关键的见解。作者挑战了自我监督视觉表征学习中的一些常见做法，并大大超过了之前发表的最先进的结果。

论文： [paper](https://arxiv.org/abs/1901.09005)

论文源代码： [code](https://github.com/google/revisiting-self-supervised)

## 2、代码结构
```


├── config               
│   ├── om_inf.sh        #msame命令
│   └── pb2om.sh         #act命令
├── log
│         ├── bin_image  #存放转化为的bin格式文件夹
│         ├── ckpt_file  #存放ckpt文件
│         ├── frozen_pb_file      #存放冻结后的pb模型
│         ├── om_file    #存放om模型
│         ├── original_jpeg_image #存放ILSVRC2012_img_val图片
│         ├── pb_file    #存放pb模型
│         └── result_image        #存放离线推理结果的文件夹
├── models
│         ├── resnet.py  #骨干网络
│         ├── utils.py   #构造网络
│         └── vggnet.py
├── self_supervision
    ├── self_supervision_lib.py
    └── supervised.py
├── ckpt2pb.py           #将ckpt文件转化为pb文件
├── compare_result.txt   #pb和om对比的结果
├── fusion_result.json   
├── inception_preprocessing.py
├── inf_input2bin.py     #将原始图片转化为bin格式
├── LICENSE
├── main                 #msame,具体的环境需要自己编译
├── modelzoo_level.txt
├── offline_reasoning.py #对比pb和om结果的函数
├── README.md
└── requirements.txt     #需要安装的第三方库文件


```
## 3、关于测试集和模型

测试集采用了使用ILSVRC2012_img_val数据集，放在./log/original_jpeg_image/下。

测试数据集和模型文件都放在百度网盘上：[百度网盘链接](https://pan.baidu.com/s/1kwQIHzhefyE9TDjDuxQ9Vg?pwd=1234)
## 4、pb模型

原始ckpt文件下载后，放在路径log/ckpt_file/目录下，使用的是ckpt-895827。执行以下命令转换为pb模型并生成冻结后的pb模型。

```
python3 ckpt2pb.py --ckpt_file=log/ckpt_file  --pb_file=log/pb_file --frozen_pb_file=log/frozen_pb_file
```
ckpt2pb.py中固定了pb模型的输出目录：./log/pb_file/ \
冻结后的pb模型的输出目录：./log/frozen_pb_file/


## 5、生成om模型。

使用atc命令将pb模型转换为om模型，执行以下命令转换为om模型。

```
sh ./config/pb2om.sh
```

注意配置对应的文件路径,soc_version默认是Ascend310

## 6、测试集内文件转换为bin文件

执行以下命令将测试集内的jpg文件经过处理转换为用于网络输入的bin文件。inf_input2bin.py文件中固定了生成bin文件存放的目录为./log/bin_image。
因为imagenet验证数据集有50000张，全部转换成bin文件消耗太长时间，可以转换少量的图片，比如100张
```
python3 inf_input2bin.py --start=1 --end=100 --original_jpeg_image=./log/original_jpeg_image --bin_image=./log/bin_image
```

## 7、使用om模型进行推理

使用msame工具进行推理。参考[msame简介](https://gitee.com/ascend/tools/tree/master/msame)，参考命令如下。

```
sh config/om_inf.sh
```

注意配置对应的文件路径


## 8、om模型离线推理性能

推理的平均运行性能为4.35 ms。


## 9、om模型离线推理结果对比pb模型的结果

执行以下命令查看推理生成的结果对比。result_image_dir必须得是om离线推理结果存放的文件夹的地址。

```
 python3 -u offline_reasoning.py --start=1 --end=100 --original_jpeg_image=log/original_jpeg_image --result_image_dir=log/result_image/20221011_12_21_54_254035 > compare_result.txt
```
在compare_result.txt中对比了100张图片，发现pb模型和om推理得到的结果是100%一样的。

