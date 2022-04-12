## 模型功能

行人重识别（REID)

## 原始模型

参考：

https://github.com/ultralytics/yolov5

原实现模型：

https://gitee.com/dw8023/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/YOLOV5_ID0378_for_TensorFlow

pb文件下载地址 :

链接：https://pan.baidu.com/s/1lgZmbp8SlZGSkLluzyM5mg 
提取码：ofwm

## om模型

om模型下载地址：

链接：https://pan.baidu.com/s/1SXq5KX8qZEEQi_JTDji-XQ 
提取码：214y

使用ATC模型转换工具进行模型转换时可以参考如下指令:

```
atc --model=/root/yolov5/model/yolov5.pb --framework=3 --output=/root/yolov5/yolov5 --soc_version=Ascend310 --input_shape="input:1,640,640,3" 
```

## 数据集准备

VOC原始验证集中的图像数据转换为bin文件参见img2bin.py文件：


bin格式数据集地址：(bin.zip)

obs://yolov5-id0378/dataset/



## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，进行性能测试。



## 性能测试

使用msame推理工具，参考如下命令，发起推理性能测试： 

```
msame --model /root/yolov5/yolov5.om --input /root/yolov5/bin --output /root/yolov5/output/ --outfmt TXT
```

```
...
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
Inference average time : 89.70 ms
Inference average time without first time: 89.70 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success
[INFO] end to destroy stream
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl
...
```

平均推理性能为 89.70ms

## 精度测试

执行精度对比文件：

```
python3 compare.py
```

最终精度：(暂未达标)

```
Ascend310推理结果：
    gpu结果:       
    npu结果:       
```





