## 模型功能

 对图像中的物体进行分类。

## 原始模型

原始模型权重下载地址 :

https://obs-9be7.obs.myhuaweicloud.com/resnet50/resnet50_tensorflow_1.7.pb


## om模型

om模型下载地址：

https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/resnet50/tf_resnet50.om

使用ATC模型转换工具进行模型转换时可以参考如下指令，具体操作详情和参数设置可以参考  [ATC工具使用指导](https://support.huaweicloud.com/ti-atc-A200dk_3000/altasatc_16_002.html) 

```
wget https://obs-9be7.obs.myhuaweicloud.com/models/resnet50_imagenet_dynamic_hw-python/aipp_resnet50.aippconfig
```

```
atc --model=./resnet50_tensorflow_1.7.pb  --framework=3 --output=./tf_resnet50 --soc_version=Ascend310  --input_shape="Placeholder:1,-1,-1,3"  --dynamic_image_size="112,112;224,224;448,448" --insert_op_conf=./aipp_resnet50.aippconfig
```

## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，将待检测om文件放在model文件夹，然后进行性能测试。

## 性能测试

使用msame推理工具，发起推理性能测试。可以参考如下指令： 

```
 ./msame --model ../../../models/tf_resnet50.om --output output/ --loop 100
```

性能测试数据为：

```
[INFO] acl init success
[INFO] open device 0 success
[INFO] create context success
[INFO] create stream success
[INFO] get run mode success
[INFO] load model ./tf_resnet50.om success
[INFO] create model description success
[INFO] get input dynamic gear count success
[INFO] create model output success
output//2022922_17_4_9_955610
[INFO] model execute success
Inference time: 8.69ms
[INFO] get max dynamic batch size success
[INFO] output data success
Inference average time: 8.690000 ms
[INFO] destroy model input success
[INFO] unload model success, model Id is 1
[INFO] Execute sample success
[INFO] end to destroy stream
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl
```

Batch: 1, shape:  448 *448 *3，带有AIPP，平均推理性能 8.69ms

## 精度测试

待完善
