## 模型功能

 对输入的图像进行风格的迁移。

## 原始模型

原始模型下载地址 :
（1）星空风格
```
https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/models/style_transfer_picture/xingkong1.pb
```
（2）糖果风格
```
https://c7xcode.obs.myhuaweicloud.com/models/style_transfer_picture/tangguo.pb
```
（3）毕加索风格
```
https://c7xcode.obs.myhuaweicloud.com/models/style_transfer_picture/bijiasuo.pb
```
（4）工农兵风格
```
https://c7xcode.obs.myhuaweicloud.com/models/style_transfer_picture/work_soldiers.pb
```

## om模型

使用ATC模型转换工具进行模型转换时可以参考如下指令:

```
atc --model=./xingkong1.pb --framework=3 --output=xingkong1_fp32_nchw_no_aipp --soc_version=Ascend310
```

```
atc --model=./bijiasuo.pb --framework=3 --output=bijiasuo_fp32_nchw_no_aipp --soc_version=Ascend310
```

```
atc --model=./tangguo.pb --framework=3 --output=tangguo_fp32_nchw_no_aipp --soc_version=Ascend310
```

```
atc --model=./work_soldiers.pb --framework=3 --output=work_soldiers_fp32_nchw_no_aipp --soc_version=Ascend310
```

## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，将待检测om文件放在model文件夹，然后进行性能测试。

## 性能测试

使用msame推理工具，参考如下命令，发起推理性能测试： 

```
./msame --model ./model/xingkong1_fp32_nchw_no_aipp.om  --output ./msame_infer --outfmt TXT --loop 100
```

```
Inference average time: 25.844950 ms
Inference average time without first time: 25.843960 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
[INFO] end to destroy stream
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl

```

Batch: 1, shape: 1080* 720 *3 ，不带AIPP，平均推理性能25.84ms

## 精度测试

待完善

推理效果(星空风格)

![输入图片说明](https://c7xcode.obs.myhuaweicloud.com/models/style_transfer_picture/verify/verify_test.jpg "style_transfer.jpg")

