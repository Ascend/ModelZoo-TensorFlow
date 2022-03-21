## 模型功能

 提出了一种新的领域自适应表示学习方法,在MNIST图像分类问题中证明了该方法的成功
## 原始模型
参考实现 ：https://github.com/pumpikano/tf-dann

## pb模型
由自己训练的ckpt生成pb模型：

pb模型获取链接 ：
- 华为obs链接：https://byq.obs.cn-north-4.myhuaweicloud.com:443/dann.pb?AccessKeyId=CHRWGFJ4FCZCTEMBJYKY&Expires=1670743698&Signature=ZXea/ZAh7v6LjAypgOL2FULfcZc%3D

## om模型

om模型
- 华为obs下载地址：https://byq.obs.cn-north-4.myhuaweicloud.com:443/dann.om?AccessKeyId=CHRWGFJ4FCZCTEMBJYKY&Expires=1670744092&Signature=2OuHTVYnxMc8idOD3a5t0wUBbs8%3D


使用ATC模型转换工具进行模型转换时可以参考如下指令，具体操作详情和参数设置可以参考  
- [ATC工具使用指导 - Atlas 200dk](https://support.huaweicloud.com/ti-atc-A200dk_3000/altasatc_16_002.html) 
- [ATC工具使用环境搭建_昇腾CANN社区版(5.0.3.alpha002)(推理)_ATC模型转换_华为云](https://support.huaweicloud.com/atctool-cann503alpha2infer/atlasatc_16_0004.html)

```
atc --model=/home/HwHiAiUser/tf-dann/dann.pb --framework=3 --output=/home/HwHiAiUser/tf-dann/om/dann --soc_version=Ascend310   --input_shape="input:32,28,28,3"
```

## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，将待检测om文件放在model文件夹，然后进行性能测试。

## 性能测试

使用msame推理工具，参考如下命令，发起推理性能测试： 

```
~/msame/msame --model ~/tf-dann/om/dann.om --output ~/tf-dann/bin/  --outfmt TXT --input ~/tf-dann/out/demo.bin

```

```
[INFO] acl init success
[INFO] open device 0 success
[INFO] create context success
[INFO] create stream success
[INFO] get run mode success
[INFO] load model /home/HwHiAiUser/tf-dann/om/dann.om success
[INFO] create model description success
[INFO] get input dynamic gear count success
[INFO] create model output success
/home/HwHiAiUser/tf-dann/bin//2021111_13_38_24_954420
[INFO] start to process file:/home/HwHiAiUser/tf-dann/out/out1/0.bin
[INFO] model execute success
Inference time: 1.234ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
......
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
Inference average time : 1.01 ms
Inference average time without first time: 1.01 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success
[INFO] end to destroy stream
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl

```

Batch: 32, shape: 28，28，3，平均推理性能 1.01 ms

## 精度测试
- 生成数据
```
python to_bin.py
```
- om模型推理
```
~/msame/msame --model ~/tf-dann/om/dann.om --output ~/tf-dann/bin/  --outfmt TXT --input ~/tf-dann/out/out1
```
## 推理精度

|      | 推理  | 论文  |
| ---- | ----- | ----- |
| ACC  | 98.12 | 76.00 |