## 模型功能

超高分辨率图像修复

## 原始模型

参考实现 ：

https://github.com/Atlas200dk/sample-imageinpainting-HiFill

原始模型网络下载地址 ：

```
https://c7xcode.obs.myhuaweicloud.com/models/imageinpainting_hifill/hifill.pb
```

## om模型

使用ATC模型转换工具进行模型转换时可以参考如下指令:

```
atc --output_type=FP32 --input_shape="img:1,512,512,3;mask:1,512,512,1" --input_format=NHWC --output="./hifill" --soc_version=Ascend310 --framework=3 --save_original_model=false --model="./hifill.pb"
```

## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，将待检测om文件放在model文件夹，然后进行性能测试。

## 性能测试

使用msame推理工具，参考如下命令，发起推理性能测试： 

```
./msame --model ./hifill.om --output ./out/ --outfmt TXT --loop 100
```

```
Inference time: 116.014ms
[INFO] model execute success
Inference time: 116.508ms
[INFO] get max dynamic batch size success
[INFO] output data success
Inference average time: 115.859300 ms
Inference average time without first time: 115.882909 ms
[INFO] destroy model input success
[INFO] unload model success, model Id is 1
[INFO] Execute sample success
```
Batch: 1, shape: 512 * 512 * 3，不带AIPP，平均推理性能115.86ms
## 精度测试

待完善

