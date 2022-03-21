## README

## 1、关于项目

本项目的目的是复现“Monocular Total Capture: Posing Face, Body, and Hands in the Wild ”论文算法。

论文链接为：[paper](https://arxiv.org/abs/1812.01598)

开源代码链接为：[code](https://github.com/CMU-Perceptual-Computing-Lab/MonocularTotalCapture/)

该文章提出了一种从单目照片输入中捕获目标人体的三维整体动作姿态的方法。效果如图所示。

![示例图](https://images.gitee.com/uploads/images/2021/1108/200620_2c36b961_5720652.png "3.png")

## 2、关于依赖库

见requirements.txt，需要安装numpy>=1.18.1 scipy 等库。

## 3、关于测试集

测试集采用了Human3.6M的测试集，分享链接见download.md。下载后请在pre_process.py和post_process.py中修改测试集相关数据的路径。处理后的测试集目录为：

+ h36m
  + images
    + S9_Directions_1.54138969_000001.jpg
    + S9_Directions_1.54138969_000006.jpg
    + ...
  + annot
    + camera.h5
    + valid.txt
    + valid.h5

## 4、pb模型

原始ckpt文件已经上传obs桶，分享链接见download.md。下载后在freeze_graph.py中修改ckpt文件的路径。执行以下命令转换为pb模型。

```
python3 freeze_graph.py
```

freeze_graph.py中固定了pb模型的输出目录为./pb_model。

## 5、生成om模型。

使用atc命令将pb模型转换为om模型，执行以下命令转换为om模型。

```
sh pb2om.sh
```

## 6、测试集内文件转换为bin文件

执行以下命令将测试集内的jpg文件经过处理转换为用于网络输入的bin文件。pre_process.py文件中固定了生成bin文件存放的目录为./TestData。

```
python3 pre_process.py
```

## 7、使用om模型进行推理

使用msame工具进行推理。参考命令如下。

```
./msame --model /home/HwHiAiUser/usr/om_model.om --input /home/HwHiAiUser/usr/input --output /home/HwHiAiUser/usr/output
```

## 8、om模型离线推理性能

推理的平均运行性能为57.03ms。

![dd](https://gitee.com/wwxgitee/pictures/raw/master/%E6%8E%A8%E7%90%86%E6%80%A7%E8%83%BD.png)

## 9、om模型离线推理loss值

在post_process.py中修改推理输出的bin文件的路径。执行以下命令查看推理生成的bin文件的loss值。

```
python3 post_process.py
```

![推理loss值](https://gitee.com/wwxgitee/pictures/raw/master/%E6%8E%A8%E7%90%86%E7%B2%BE%E5%BA%A6.png)