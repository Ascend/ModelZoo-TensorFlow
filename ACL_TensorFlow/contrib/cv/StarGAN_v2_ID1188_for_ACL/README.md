# StarGAN_v2_for_ACL_TensorFlow

## 原始模型

根据[训练过程](https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/StarGAN_v2_ID1188_for_TensorFlow)，得到预训练模型

## 转为om模型

以refer_test模式为例，使用如下命令进行转换，其中model路径根据实际情况配置。[PB模型](链接：https://pan.baidu.com/s/1BF2psccBuyRTNloWtrz5Nw 
提取码：t62p)

```
atc --model=./output_model/pb_model/frozen_model.pb --framework=3 --output=./model --soc_version=Ascend310 --input_shape='input_node1:1,256,256,3;input_node2:1,256,256,3'
```

## 数据预处理

参考训练过程，准备好数据集文件。执行以下命令，将样本图像数据转化为bin文件：

```
python convert.py --mode img2bin
```

## 性能

使用msame推理工具进行性能测试：

```
msame --model model.om --input "./inputs/pixabay_dog_004000.bin","./refer/pixabay_dog_004005.bin" --output output/ --loop 10
```

如下log信息：

```
[INFO] acl init success
[INFO] open device 0 success
[INFO] create context success
[INFO] create stream success
[INFO] get run mode success
[INFO] load model model.om success
[INFO] create model description success
[INFO] get input dynamic gear count success
[INFO] create model output success
output//20211128_191144
[INFO] start to process file:./inputs/pixabay_dog_004000.bin
[INFO] start to process file:./refer/pixabay_dog_004005.bin
... ...
[INFO] model execute success
Inference time: 65.768ms
[INFO] model execute success
Inference time: 65.495ms
[INFO] model execute success
Inference time: 65.494ms
[INFO] model execute success
Inference time: 65.435ms
[INFO] model execute success
Inference time: 65.544ms
[INFO] model execute success
Inference time: 65.56ms
[INFO] model execute success
Inference time: 65.554ms
[INFO] model execute success
Inference time: 65.517ms
[INFO] model execute success
Inference time: 65.522ms
[INFO] model execute success
Inference time: 65.56ms
[INFO] get max dynamic batch size success
[INFO] output data success
Inference average time: 65.544900 ms
Inference average time without first time: 65.520111 ms
[INFO] destroy model input success.
[INFO] unload model success, model Id is 1
[INFO] Execute sample success
[INFO] end to destroy stream
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl
```

平均推理性能为65.52ms

## 精度

推理完成后，执行如下命令，将bin文件转换回图像文件：

```
python convert.py --mode bin2img
```

得到生成后的结果图像后，使用FID进行评估：

TO BE FINISHED