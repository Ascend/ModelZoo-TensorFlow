# YAD2K(yolov2) 推理介绍

##  概述

Redmon和Farhadi（2017）提出了YOLO9000（也被成为YOLOv2）。与YOLO相比，YOLOv2做出了如下的改进，包括batch normalization，使用高分辨率训练图像、维度聚类（K-means）以及锚框（anchor box）等。backbone使用 Darknet19，对于网络输出其预测的是相对于当前网格的偏移量而不是预测边界框的坐标。

论文中 yolov2 以 40FPS 速度在 VOC 2007 Test集 上实现了78.6%mAP。

本项目 GPU上以 69FPS 速度在 VOC 2007 Test集上 以score_threshold=0.05 与iou_threshold=0.6 复现 72.51%mAP 

+ 参考论文：

  [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242)

  


## 默认配置

+ 推理超参
  + eval  score_threshold=0.05
  + eval  iou_threshold=0.6

## Rquirements

+ Keras 2.3.1
+ TensorFlow 1.15.0
+ opencv-python 4.5.4.58
+ pillow 8.3.2
+ numpy 1.16.2
+ hdf5 1.12.1
+ h5py 2.10.0
+ math
+ shutil
+ time
+ matplotlib
+ os
+ json
+ glob

  

## 高级参考

### 示例代码结构

```
├── data_process
│    ├──check_data.py                         //数据txt文本内容测试，查看内容是否正确
│    ├──config.py                    		  //网络参数配置
│    ├──data_loader.py                        //构建dataset，方便在训练过程中读取数据
│    ├──voc_annotation.py                     //所需数据集拆分成训练和测试集两部分
│    ├──2007_test.txt                         //测试集路径地址以及标签信息
├── evalue									  //生成推理结果文件存放处
├── inference_picresult						  //om推理所需的数据、推理输出与结果图片
├── nets
│    ├──v2net.py                              //yolov2网络模型
│    ├──yololoss.py                           //网络输出编码以及损失函数设计
├── log									  	  //训练的模型保存地址
│    ├──new_2.h5                              //应用于推理的h5推理模型文件
├── test_img								  //可以使用里面的图片做一个简单的小detect
├── detect.py                                 //模型预测相关函数
├── ep100-loss1.568-val_loss18.650.h5         //初始h5参数模型，需用save_h5_main.py保存推理模型
├── get_info.py                               //生成h5模型推理 预测和实际的分类和预测框结果文件
├── get_info_fortuili2.py                     //生成om模型推理 预测和实际的分类和预测框结果文件
├── get_map.py                                //读取上两者的结果文件，生成AP、																	Precison、Recall和F1的图像
├── h5_to_pb_new.py                           //将h5文件转换成pb
├── predata.py                                //将测试图片预处理保存为bin文件
├── README.md                                 //代码说明文档
├── save_h5_main.py                           //将初始h5参数模型保存为推理h5模型文件
├── yak2k_new3.om                             //生成用于推理的om文件
```



# 推理过程

## 输入预处理

先预处理输入数据，使用 predata.py 将测试图片均转换成bin格式（注意数据格式为 float32）。

需注意代码中2007_test.txt记录的图片地址是否对应正确。

```shell
python predata.py   
```

## 保存h5推理模型

由于原有h5模型包含后面的yolo_loss部分，且引入了不必要的input输入。所以需要保存一个仅包含推理的模型部分，用save_h5_main.py来保存h5推理模型文件。

```shell
python save_h5_main.py
```

## 数据预处理

需要将测试图片进行预处理后，转换为bin格式文件

```shell
python predata.py
```

## h5模型转换成pb模型

将h5模型参数文件，使用 h5_to_pb_new.py 转换为pb文件。

```shell
python h5_to_pb_new.py
```

## pb模型转换成om模型

+ pb模型转换为om模型，需要使用atc工具

+ 将pb模型文件，使用atc 指令转换为 om模型文件。（注意pb模型文件放置的地址，om模型的输出地址）

  ```shell
  atc --model=/root/tools/yolov2/yad2k_new2.pb --framework=3 --output=/root/tools/yolov2/yak2k_new3 --soc_version=Ascend310 --input_shape="input_0:1,416,416,3" --log=info --out_nodes="Identity:0"
  ```


## 使用msame进行推理

+ 使用msame工具，将输入bin数据 输入 om 模型，得到推理的中间结果。（注意模型文件放置的位置，bin数据地址，和输出的结果地址）

  ```shell
  ./msame --model "/root/tools/yolov2/yak2k_new2.om" --input "/root/tools/yolov2/demo_bin" --output "/root/tools/yolov2/out/" --outfmt BIN
  ```

## 后处理

+ 拿到输出的结果，使用 get_info_fortuili2.py 后处理 生成推理结果与真实标签结果。 再使用 get_map.py 画出推理结果的各类AP图像，以及输出平均mAP的结果。

  ```shell
  python get_info_fortuili2.py
  python get_map.py
  ```



# 精度评估

h5推理得到的mAP结果 72.51%mAP

![实际推理结果](http://kyle-pic.oss-cn-hangzhou.aliyuncs.com/img/实际推理.png)

om推理得到的mAP结果  72.50%mAP

![om推理结果](http://kyle-pic.oss-cn-hangzhou.aliyuncs.com/img/om推理.png)



# 性能评估

h5模型推理得到的FPS为 62.81

![h5模型推理得到的FPS](http://kyle-pic.oss-cn-hangzhou.aliyuncs.com/img/h5_refer_fps.png)

om模型推理得到的结果：分为 图片预处理+ 模型结果输出的时间  + 后处理的时间

+ 图片预处理平均时间  10.49ms  

![图片预处理时间](http://kyle-pic.oss-cn-hangzhou.aliyuncs.com/img/predata_time.png)

+ om模型结果输出的平均时间为 4.67ms 

   ![om模型结果输出的时间](http://kyle-pic.oss-cn-hangzhou.aliyuncs.com/img/om_infer_time.png)

+ 后处理得到的fps 为 291.65   换算成时间为  3.43ms

![后处理得到的fps](http://kyle-pic.oss-cn-hangzhou.aliyuncs.com/img/om_refer_fps.png)

+ 合并计算得到 时间为 10.49+4.67+3.43=18.59ms  换算成 fps 为 (1000/18.59) = 53.79



