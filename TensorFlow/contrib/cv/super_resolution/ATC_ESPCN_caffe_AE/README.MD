## 模型功能

图像超分辨率

## 原始模型

参考实现 ：

https://github.com/wangxuewen99/Super-Resolution/tree/master/ESPCN

原始模型权重下载地址 :

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/super_resolution/ESPCN/ESPCN.caffemodel

原始模型网络下载地址 ：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/super_resolution/ESPCN/ESPCN.prototxt


## om模型

om模型下载地址：

https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/super_resolution/ESPCN/ESPCN_280_280.om

使用ATC模型转换工具进行模型转换时可以参考如下指令:

```
atc --model=./ESPCN.prototxt --weight=./ESPCN.caffemodel --framework=0 --input_format=NCHW --input_shape="data: 1, 1, 280, 280" --output=./ESPCN_280_280 --soc_version=Ascend310 --output_type=FP32
```

## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，将待检测om文件放在model文件夹，然后进行性能测试。

## 性能测试

使用msame推理工具，参考如下命令，发起推理性能测试： 

```
./msame --model model/ESPCN_280_280.om --output output/ --loop 100
```

```
[INFO] output data success
Inference average time: 141.238460 ms
Inference average time without first time: 141.244515 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
Test Finish!
```

Batch: 1, shape: 1 * 280 * 280，不带AIPP，平均推理性能141.23ms

## 精度测试

待完善

推理效果

![输入图片说明](https://images.gitee.com/uploads/images/2021/0204/135517_350973c4_8070502.png "head_GT_bicubic.png")
![输入图片说明](https://images.gitee.com/uploads/images/2021/0204/135526_edd3f8e8_8070502.png "head_GT_ESPCN.png")
