## 模型功能

车辆颜色分类

## 原始模型

原始模型下载地址 :

https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/YOLOV3_carColor_sample/data/color.pb

对应的cfg文件下载地址：

https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/YOLOV3_carColor_sample/data/aipp.cfg

## om模型

om模型下载地址：

https://issue-modelzoo.obs.cn-north-4.myhuaweicloud.com/Ayolov3/color_dvpp_10batch.om

使用ATC模型转换工具进行模型转换时可以参考如下指令:

```shell
atc --input_shape="input_1:-1,224,224,3" --output=./color_dynamic_batch --soc_version=Ascend310 --framework=3 --model=./color.pb --insert_op_conf=./aipp.cfg --dynamic_batch_size="1,2,4,8"
```

## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，将待检测om文件放在model文件夹，然后进行性能测试。

## 性能测试

使用msame推理工具，参考如下命令，发起推理性能测试： 

```shell
./msame --model model/color_dynamic_batch.om --output output/ --loop 100 --dymBatch 8
```

```shell
[INFO] output data success
Inference average time: 5.778290 ms
Inference average time without first time: 5.772394 ms
[INFO] destroy model input success
[INFO] unload model success, model Id is 1
[INFO] Execute sample success

```

Batch: 8, shape: 8 * 224* 224，平均推理性能5.778290ms

## 精度测试

待完善

推理效果:

![输入图片说明](https://issue-modelzoo.obs.cn-north-4.myhuaweicloud.com/Ayolov3/color.jpg "color.jpg")
