## 模型功能

图像卡通画

## 原始模型

参考实现 ：

https://github.com/TachibanaYoshino/AnimeGAN

三种不同shape的pb模型及其aipp配置文件下载地址 ：
```
https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/models/animeGAN/AnimeGAN_256_256.pb
https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/models/animeGAN/aipp_256_256.cfg
```
```
https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/models/animeGAN/AnimeGAN_512_512.pb
wget https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/models/animeGAN/aipp_512_512.cfg
```
```
https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/models/animeGAN/AnimeGAN_1024_1024.pb
wget https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/models/animeGAN/aipp_1024_1024.cfg
```

## om模型

om模型下载地址：

`https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/models/animeGAN/AnimeGANv2_256.om`

`https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/models/animeGAN/AnimeGANv2_512.om`

`https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/models/animeGAN/AnimeGANv2_1024.om`

使用ATC模型转换工具进行模型转换时可以参考如下指令:

```shell
atc --model="./AnimeGAN_256_256.pb" --output_type=FP32 --input_shape="test:1,256,256,3" --input_format=NHWC --output="AnimeGANv2_256" --soc_version=Ascend310 --framework=3 --precision_mode=allow_fp32_to_fp16 --insert_op_conf=aipp_256_256.cfg
atc --model="./AnimeGAN_512_512.pb" --output_type=FP32 --input_shape="test:1,512,512,3" --input_format=NHWC --output="AnimeGANv2_512" --soc_version=Ascend310 --framework=3 --precision_mode=allow_fp32_to_fp16 --insert_op_conf=aipp_512_512.cfg
atc --model="./AnimeGAN_1024_1024.pb" --output_type=FP32 --input_shape="test:1,1024,1024,3" --input_format=NHWC --output="AnimeGANv2_1024" --soc_version=Ascend310 --framework=3 --precision_mode=allow_fp32_to_fp16 --insert_op_conf=aipp_1024_1024.cfg
```

## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，将待检测om文件放在model文件夹，然后进行性能测试。

## 性能测试

使用msame推理工具，参考如下命令，发起推理性能测试： 

```shell
./msame --model model/AnimeGANv2_256.om --output output/ --loop 100
./msame --model model/AnimeGANv2_512.om --output output/ --loop 100
./msame --model model/AnimeGANv2_1024.om --output output/ --loop 100
```

```shell
[INFO] output data success
Inference average time: 12731.3 ms
Inference average time without first time: 12724.6 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
Test Finish!
```

Batch: 1, shape: 1 * 512* 512，平均推理性能141.23ms

## 精度测试

待完善

推理效果

![输入图片说明](https://images.gitee.com/uploads/images/2021/0810/164154_41da2b25_8070502.png "屏幕截图.png")

![输入图片说明](https://images.gitee.com/uploads/images/2021/0810/164617_834d8f45_8070502.png "微信图片_20210810164604.png")