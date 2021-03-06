## 模型功能

该模型图像边缘检测

## 原始模型

参考实现 ：

 https://github.com/yun-liu/rcf

原始模型网络下载地址 ：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/dege_detection/rcf.prototxt

原始模型权重文件下载地址

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/dege_detection/rcf_bsds.caffemodel


## om模型

om模型下载地址：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/dege_detection/rcf.om

使用ATC模型转换工具进行模型转换时可以参考如下指令

```
atc --model=rcf.prototxt --weight=./rcf_bsds.caffemodel --framework=0 --output=rcf --soc_version=Ascend310 --input_fp16_nodes=data --input_format=NCHW --output_type=FP32
```

## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，将待检测om文件放在model文件夹，然后进行性能测试。

## 性能测试

使用msame推理工具，参考如下命令，发起推理性能测试： 

```
./msame --model rcf.om  --output output/ --loop 10
```

性能测试数据为：

```
[INFO] output data success
Inference average time: 37.446300 ms
Inference average time without first time: 37.429556 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
Test Finish!
```

Batch: 1, shape: 3,512,512 不带AIPP，平均推理性能37.429556 ms

## 精度测试

待完善
推理效果

![输入图片说明](https://images.gitee.com/uploads/images/2021/0202/105056_596c8382_8113712.jpeg "ori.jpg")
![输入图片说明](https://images.gitee.com/uploads/images/2021/0202/105106_6052b95e_8113712.jpeg "out_ori.jpg")