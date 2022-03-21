## 模型功能

检测输入图片中的物体类别和框的坐标。

## 原始模型

参考实现 ：

https://github.com/wizyoung/YOLOv3_TensorFlow

原始模型网络下载地址 ：

```
https://c7xcode.obs.myhuaweicloud.com/models/YOLOV3_coco_detection_picture_with_postprocess_op/yolov3_tensorflow_1.5.pb
```

对应的cfg文件下载地址：

```
https://c7xcode.obs.myhuaweicloud.com/models/YOLOV3_coco_detection_picture_with_postprocess_op/aipp_nv12.cfg
```


## om模型

使用ATC模型转换工具进行模型转换时可以参考如下指令:

```
atc --model=yolov3_tensorflow_1.5.pb --framework=3 --output=yolov3_tf_aipp --output_type=FP32 --soc_version=Ascend310 --input_shape="input:1,416,416,3" --out_nodes="yolov3/yolov3_head/Conv_6/BiasAdd:0;yolov3/yolov3_head/Conv_14/BiasAdd:0;yolov3/yolov3_head/Conv_22/BiasAdd:0" --log=info --insert_op_conf=aipp_nv12.cfg
```

## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，将待检测om文件放在model文件夹，然后进行性能测试。

## 性能测试

使用msame推理工具，参考如下命令，发起推理性能测试： 

```
./msame --model ./yolov3_tf_aipp.om --output ./outs --outfmt TXT --loop 100
```

```
Inference average time: 9.438250 ms
Inference average time without first time: 9.437131 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
[INFO] end to destroy stream
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl
```

Batch: 1, shape: 416 * 416* 3，带AIPP，平均推理性能9.44ms

## 精度测试

待完善

