## 模型功能

通过检测输入图片,识别图中不同种类物体。

## 原始模型

参考实现 ：

https://github.com/hunglc007/tensorflow-yolov4-tflite

原图是用Keras训练得到的，先转成h5文件，再将h5转成pb文件。
后处理中的where算子不支持，因此拆分掉后处理部分的算子。

拆分pb模型的命令：

python3.7.5 -m tensorflow.python.tools.optimize_for_inference --input=yolov4.pb --output=yolov4_no_postprocess.pb --input_names="Input" --output_names="functional_1/tf_op_layer_concat_13/concat_13,functional_1/tf_op_layer_concat_14/concat_14"

原始模型网络下载地址 ：

https://nkxiaolei88.obs.cn-north-1.myhuaweicloud.com/ATC%20Model/YoloV4/yolov4_no_postprocess.pb


## om模型

om模型下载地址：

https://nkxiaolei88.obs.cn-north-1.myhuaweicloud.com/ATC%20Model/YoloV4/yolov4_no_postprocess.om

使用ATC模型转换工具进行模型转换时可以参考如下指令:

```
atc --model=./yolov4_no_postprocess.pb --framework=3 --output=./yolov4_no_postprocess --soc_version=Ascend310 --input_shape="Input:1,416,416,3"
```


## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，将待检测om文件放在model文件夹，然后进行性能测试。

## 性能测试

使用msame推理工具，参考如下命令，发起推理性能测试： 

```
./msame --model /home/HwHiAiUser/tools/msame/model/yolov4_no_postprocess.om --output /home/HwHiAiUser/tools/msame/output/ --outfmt TXT --loop 100
```

```
[INFO] output data success
Inference average time: 46.967860 ms
Inference average time without first time: 46.953970 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
Test Finish!
```

Batch: 1, shape: 416 * 416 * 3，不带AIPP，平均推理性能46.967860ms

## 精度测试

待完善

推理效果

![输入图片说明](https://images.gitee.com/uploads/images/2021/0226/154600_805c7298_8070502.jpeg "kite.jpg")
![输入图片说明](https://images.gitee.com/uploads/images/2021/0226/154612_96b7e29f_8070502.jpeg "out_test.jpg")
