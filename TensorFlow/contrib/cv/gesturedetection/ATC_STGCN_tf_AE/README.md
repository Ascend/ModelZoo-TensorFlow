## 模型功能

人体动作识别

## 原始模型

参考实现 ：

https://github.com/VeritasYin/STGCN_IJCAI-18

原始模型网络下载地址 ：

https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/gesturedetection/stgcn_fps30_sta_ho_ki4.pb

## om模型

om模型下载地址：

https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/gesturedetection/stgcn_fps30_sta_ho_ki4.om

使用ATC模型转换工具进行模型转换时可以参考如下指令:

```
atc --input_shape="input_features:1,2,30,14" --input_format=NCHW --output="./stgcn_fps30_sta_ho_ki4" --soc_version=Ascend310 --framework=3 --model="./stgcn_fps30_sta_ho_ki4.pb" 
```

## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，将待检测om文件放在model文件夹，然后进行性能测试。

## 性能测试

使用msame推理工具，参考如下命令，发起推理性能测试： 

```
./msame --model ../model/stgcn_fps30_sta_ho_ki4.om --output output/ --loop 100
```

```
[INFO] output data success
Inference average time: 19.236920 ms
Inference average time without first time: 19.233576 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
Test Finish!
```

Batch: 1, shape: 2 * 30 * 14，不带AIPP，平均推理性能19.24ms

## 精度测试

待完善

推理效果

![输入图片说明](https://images.gitee.com/uploads/images/2021/0322/163311_ef09527a_8070502.png "捕获1.PNG")

![输入图片说明](https://images.gitee.com/uploads/images/2021/0322/163321_bbf6cffe_8070502.png "捕获.PNG")
