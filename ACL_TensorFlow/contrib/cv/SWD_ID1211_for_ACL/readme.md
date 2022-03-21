## 模型功能

 对图像中的物体进行识别分类。

## 原始模型

参考实现 ：[ml-cvpr2019-swd/swd.py at master · apple/ml-cvpr2019-swd](https://github.com/apple/ml-cvpr2019-swd/blob/master/swd.py)

原始pb模型网络下载地址 ：https://gather.obs.cn-north-4.myhuaweicloud.com:443/swd/swd.pb?AccessKeyId=6CVKNYMLTP9K0JNCCT6A&Expires=1666795192&Signature=sNd6KNk7u8am82iDHSQ2Dt%2BlTaM%3D



## om模型

om模型
- 华为obs下载地址：https://gather.obs.cn-north-4.myhuaweicloud.com:443/swd/swd_310_v1.om?AccessKeyId=6CVKNYMLTP9K0JNCCT6A&Expires=1666795218&Signature=TXNKSN%2B%2BVo95HHUCsXRxP/7u9tc%3D
- 百度网盘链接：https://pan.baidu.com/s/1U1Kga_qY4VhsyQ4XM4yh5w 
提取码：u8sk

使用ATC模型转换工具进行模型转换时可以参考如下指令，具体操作详情和参数设置可以参考  
- [ATC工具使用指导 - Atlas 200dk](https://support.huaweicloud.com/ti-atc-A200dk_3000/altasatc_16_002.html) 
- [ATC工具使用环境搭建_昇腾CANN社区版(5.0.3.alpha002)(推理)_ATC模型转换_华为云](https://support.huaweicloud.com/atctool-cann503alpha2infer/atlasatc_16_0004.html)

```
atc --model=$HOME/infer/swd.pb --framework=3 --output=./swd_v1  --soc_version=Ascend310   --input_shape="input:1,2"
```

## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，将待检测om文件放在model文件夹，然后进行性能测试。

## 性能测试

使用msame推理工具，参考如下命令，发起推理性能测试： 

```
~/msame/msame --model ~/infer/swd_310_v1.om --output ~/msame/out-10-31  --outfmt TXT --loop 1

```

```
[INFO] acl init success
[INFO] open device 0 success
[INFO] create context success
[INFO] create stream success
[INFO] get run mode success
[INFO] load model /home/HwHiAiUser/infer/swd_310_v1.om success
[INFO] create model description success
[INFO] get input dynamic gear count success
[INFO] create model output success
/home/HwHiAiUser/msame/out-10-31/20211031_9_3_35_65633
[INFO] model execute success
Inference time: 0.988ms
[INFO] get max dynamic batch size success
[INFO] output data success
Inference average time: 0.988000 ms
[INFO] destroy model input success
[INFO] unload model success, model Id is 1
[INFO] Execute sample success
[INFO] end to destroy stream
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl

```

Batch: 1, shape: 1 * 2，平均推理性能 1.174000 ms

## 精度测试
- 生成数据
```
python gen_bin_data.py
```
- om模型推理
```
~/msame/msame --model ~/infer/swd_310_v1.om  --input "/home/HwHiAiUser/infer/point/" --output ~/msame/out-10-31-v2  --outfmt TXT
```
- 数据可视化
```
python visual_data.py
```


推理效果：经过域适应训练，不光能够分类出红蓝的两类数据，而且对于没见过label的绿点也能够成功分类。
![域适应-图片说明](https://images.gitee.com/uploads/images/2021/1116/161432_c5672263_2069446.png "屏幕截图.png")
