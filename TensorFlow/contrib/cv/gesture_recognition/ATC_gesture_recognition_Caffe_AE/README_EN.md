## 模型功能

 基于该模型实现20种手势识别。

## 原始模型

原始模型权重下载地址 :

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/gesture_recognition/resnet18_gesture.caffemodel

原始模型网络下载地址 ：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/gesture_recognition/resnet18_gesture.prototxt


## om模型

om模型下载地址：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/gesture_recognition/gesture_yuv.om

使用ATC模型转换工具进行模型转换时可以参考如下指令:

```
/home/ascend/Ascend/ascend-toolkit/20.0.RC1/atc/bin/atc --output_type=FP32 --input_shape="data:1,3,224,224" --weight="/home/ascend/models/resnet18_gesture.caffemodel"  --input_format=NCHW --output="/home/ascend/modelzoo/gesture/device/gesture_yuv.om" --soc_version=Ascend310 --insert_op_conf=/home/ascend/modelzoo/gesture/device/insert_op.cfg --framework=0 --save_original_model=false --model="/home/ascend/models/resnet18_gesture.prototxt" 
```

## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，将待检测om文件放在model文件夹，然后进行性能测试。

## 性能测试

使用msame推理工具，发起推理性能测试。可以参考如下指令： 

```
./msame --model ../model/gesture_yuv.om --output output/ --loop 100
```

性能测试数据为：

```
[INFO] output data success
Inference average time: 1.404590 ms
Inference average time without first time: 1.402273 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
Test Finish!

```

Batch: 1, shape:  224 *224 *3，带AIPP，平均推理性能 1.402273ms

## 精度测试

待完善

推理效果

![输入图片说明](https://images.gitee.com/uploads/images/2021/0208/143256_abab30cc_8189215.jpeg "test1.jpg")
![输入图片说明](https://images.gitee.com/uploads/images/2021/0208/143308_57b3f0a4_8189215.jpeg "test2.jpg")