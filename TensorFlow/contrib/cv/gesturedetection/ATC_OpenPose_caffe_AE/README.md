## 模型功能

人体骨骼关键点检测

## 原始模型

参考实现 ：

https://github.com/CMU-Perceptual-Computing-Lab/openpose


原始模型权重下载地址 :

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/gesturedetection/pose_iter_440000.caffemodel

原始模型网络下载地址 ：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/gesturedetection/pose_deploy.prototxt

aipp配置文件下载地址 ：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/gesturedetection/insert_op.cfg

## om模型

om模型下载地址：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/gesturedetection/pose_deploy.om

使用ATC模型转换工具进行模型转换时可以参考如下指令:

```
atc --input_shape="data:1,3,128,128" --weight="./pose_iter_440000.caffemodel" --input_format=NCHW --output="./pose_deploy" --soc_version=Ascend310 --insert_op_conf=./insert_op.cfg --framework=0 --model="./pose_deploy.prototxt" 
```

## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，将待检测om文件放在model文件夹，然后进行性能测试。

## 性能测试

使用msame推理工具，参考如下命令，发起推理性能测试： 

```
./msame --model ../model/pose_deploy.om --output output/ --loop 100
```

```
[INFO] output data success
Inference average time: 9.369970 ms
Inference average time without first time: 9.364525 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
Test Finish!
```

Batch: 1, shape: 3 * 128 * 128，带AIPP，平均推理性能9.37ms

## 精度测试

待完善

推理效果

![输入图片说明](https://images.gitee.com/uploads/images/2021/0322/162551_fb65d2a9_8070502.png "捕获.PNG")