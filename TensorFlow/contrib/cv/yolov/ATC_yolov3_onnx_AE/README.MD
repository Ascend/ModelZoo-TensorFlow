## 模型功能

 对图像中的物体进行检测。

## 原始模型

参考实现 ：

https://github.com/ultralytics/yolov3
branch=master
commit_id=166a4d590f08b55cadfebc57a11a52ff2fc2b7d3

yolov3模型文件 :

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/YOLOV3_carColor_sample/data/yolov3_t.onnx


对应的cfg文件下载地址：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/YOLOV3_carColor_sample/data/aipp_onnx.cfg



## om模型

yolov3.om模型下载地址：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/Yolov3/yolov3_framework_caffe_aipp_1_batch_1_input_int8_output_FP32.om ？？？？？？？？？？？？？ 


生成yolov3的适配昇腾AI处理器的离线模型文件时可以参考如下指令:

```
atc --model=./yolov3_t.onnx --framework=5 --output=yolov3 --input_shape="images:1,3,416,416;img_info:1,4" --soc_version=Ascend310 --input_fp16_nodes="img_info" --insert_op_conf=aipp_onnx.cfg
```


## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，将待检测om文件放在model文件夹，然后进行性能测试。

## 性能测试

使用msame推理工具，参考如下命令，发起推理性能测试： 

yolov3.om推理性能测试：

```
./main --model "/home/HwHiAiUser/mytemp/yolov3.om"  --output "./out/" --outfmt TXT --loop 1
```

```
[INFO] model execute success
Inference time: 11.81ms
[INFO] get max dynamic batch size success
[INFO] output data success
Inference average time: 11.810000 ms
```

Batch: 1, shape: 624* 416 *3 ，带AIPP，平均推理性能11.81ms


## 精度测试

待完善

推理效果

![输入图片说明](https://images.gitee.com/uploads/images/2020/1116/160411_c5731771_8113712.jpeg "yolo-caffe.jpg") ？？？？？？？？？？？？？？？？？？

