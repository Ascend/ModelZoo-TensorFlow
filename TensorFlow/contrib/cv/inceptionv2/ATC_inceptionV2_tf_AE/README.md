## 模型功能

该模型实现了对踢脚线识别的功能

## 原始模型

原始模型网络下载地址 ：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/InceptionV2/frozen_graph.pb
## om模型

om模型下载地址：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/InceptionV2/frozen_graph-inception-resnet-test1.om

使用ATC模型转换工具进行模型转换时可以参考如下指令

```
atc --input_shape="input:1,299,299,3" --input_format=NHWC --output="frozen_graph-inception-resnet-test1" --soc_version=Ascend310 --framework=3 --model="./frozen_graph.pb"
```

## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，将待检测om文件放在model文件夹，然后进行性能测试。

## 性能测试

使用msame推理工具，参考如下命令，发起推理性能测试： 

```
./msame --model frozen_graph-inception-resnet-test1.om  --output output/ --loop 10
```

性能测试数据为：

```
[INFO] create model description success
[INFO] create model output success
[INFO] model execute success
Inference time: 9.741ms
[INFO] model execute success
Inference time: 7.241ms
[INFO] model execute success
Inference time: 7.343ms
[INFO] model execute success
Inference time: 7.219ms
[INFO] model execute success
Inference time: 7.277ms
[INFO] model execute success
Inference time: 7.268ms
[INFO] model execute success
Inference time: 7.205ms
[INFO] model execute success
Inference time: 7.263ms
[INFO] model execute success
Inference time: 7.182ms
[INFO] model execute success
Inference time: 7.297ms
output//20210315_081336
[INFO] output data success
Inference average time: 7.503600 ms
Inference average time without first time: 7.255000 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
Test Finish!
```

Batch: 1, shape: 299,299,3 不带AIPP，平均推理性能7.503600 ms


### 推理效果

推理前：

![输入图片说明](https://images.gitee.com/uploads/images/2021/0315/161709_6c4062a0_5578318.jpeg "1101.jpg")

推理后：

![输入图片说明](https://images.gitee.com/uploads/images/2021/0315/161725_624aef99_5578318.jpeg "1101.jpg")
