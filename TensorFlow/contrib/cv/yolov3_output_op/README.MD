## 模型功能

 对yolov3的tensorflow模型的输出进行解析。

## 原始模型

参考实现 ：
```
https://pjreddie.com/darknet/yolo/
```

原始模型权重下载地址 :
```
https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/Yolov3/yolov3.caffemodel
```

模型网络下载地址 ：
```
https://c7xcode.obs.myhuaweicloud.com/models/YOLOV3_coco_detection_picture_with_postprocess_op/yolov3_modify.prototxt
```


## om模型

使用ATC模型转换工具进行模型转换时可以参考如下指令:

```
atc --model=./yolov3_modify.prototxt --weight=./yolov3.caffemodel --framework=0 --output=yolov3_output_op --soc_version=Ascend310
```

## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，将待检测om文件放在model文件夹，然后进行性能测试。

## 性能测试

使用msame推理工具，参考如下命令，发起推理性能测试： 

```
./msame --model ./yolov3_output_op.om --output ./outs --outfmt TXT --loop 100
```

```
Inference average time: 10.446380 ms
Inference average time without first time: 10.446081 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
[INFO] end to destroy stream
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl
```

平均推理性能10.45ms

## 精度测试

待完善



