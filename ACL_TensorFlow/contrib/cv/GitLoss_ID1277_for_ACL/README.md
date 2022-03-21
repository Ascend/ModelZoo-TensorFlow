## 模型功能

 对gitloss的tensorflow模型的输出进行解析, 使用固化后gitloss模型进行推理，对mnist数据集进行图像分类。

## 原始模型

参考实现 ：
```
https://github.com/kjanjua26/Git-Loss-For-Deep-Face-Recognition
```

原始pb模型网络下载地址 :
```
https://yfiles.obs.cn-north-4.myhuaweicloud.com:443/gitloss_infer/gitloss.pb?AccessKeyId=TRUQTC6GBQWLPC2LQEAW&Expires=1668151265&Signature=so3Vx41Ayqi4OqvcMl49pQl3rRg%3D
```


## om模型

原始om模型网络下载地址 :
```
https://yfiles.obs.cn-north-4.myhuaweicloud.com:443/gitloss_infer/gitloss_.om?AccessKeyId=TRUQTC6GBQWLPC2LQEAW&Expires=1668151398&Signature=AvdMONnHS9OEllqX7v%2BbhtTk1hA%3D
```

使用ATC模型转换工具进行模型转换时可以参考如下指令:

```
atc --model=./gitloss.pb --framework=3 --output=./gitloss_  --soc_version=Ascend310   --input_shape="input_images:1,28,28,1"
```

## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，将待检测om文件放在model文件夹，然后进行性能测试。

## 性能测试

使用msame推理工具，参考如下命令，发起推理性能测试： 

```
./msame --model ./gitloss_.om --output ./gitloss/out-11-15  --outfmt TXT --loop 1  --input "./test_img/"
```

```
Inference average time: 0.73 ms
Inference average time without first time: 0.73 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
[INFO] end to destroy stream
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl
```

平均推理性能 0.73 ms

## 精度测试

推理数据集(.bin)和数据label下载地址:

数据集
```
URL:
https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=IhRhNmLCO+ThRv/EeEc9K/+b2p3GjqHeFkayrRNOyT5tSAOxjIdgBOSLFZoZQDKkXdBFo5gDAgm6ixbIT4jwyU2eSI5Wesqf6nBb/ZUgdZVRMBusYIsXEK5DXn5+P42/pZe3hZZGLazXNggkA9IGzQYsjJ/46i7FB2NfZXvNpRhCk1IUjwYmCbZfyHrULHkT8dzv2MJpE6ueFHp2YzZqsqZEJbrnKYRmUMe6vxxjgoCGVgec6tjUNSGOiBnsojf0hRiBaQqVXgAl6+i8lqNr1pzwm9WtkhW2hayGlTSf5P+LfGTUP0LRdJfWZe4Lzsg3bS7kHIkqF4QqT69BP6Q/MwbT7ca44LAmj1oa3txIC1C0hNvWhAlojLSLQDiwpp7T00e5jxemPneS9zQFAPwTiY1FjVMnnoSim+nNRBWezVPJgUNz1NlUEtsB5hm7IkUkRiKfCj7r6JMZhB0UoW6WWHDPWWro2MsS8KThHLb6R6s4p7JRZJR8EZA85c0Uq/MVLNCp74+LrEk4c6ghBW/7bgjUHjLJXcJ7clvYZuQqU7Oz9JdoWeKlPsu1YTZ/+jacIonpCtM1MDij8bzScQ3ctidjsBOe8BKFs/5Oyzh4dO0=

提取码:
123456

*有效期至: 2022/11/10 23:31:50 GMT-08:00
```

label
```
URL:
https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=IhRhNmLCO+ThRv/EeEc9K/+b2p3GjqHeFkayrRNOyT5tSAOxjIdgBOSLFZoZQDKkXdBFo5gDAgm6ixbIT4jwyU2eSI5Wesqf6nBb/ZUgdZVRMBusYIsXEK5DXn5+P42/pZe3hZZGLazXNggkA9IGzQYsjJ/46i7FB2NfZXvNpRhCk1IUjwYmCbZfyHrULHkT8dzv2MJpE6ueFHp2YzZqsqZEJbrnKYRmUMe6vxxjgoBj1Lb0JvIM7k81pq7zjOLEhccRROCjpZN453cbCBu2u7MpVV8lnZWaKVNIKbscbVPfmFA3H7K8sbx90UdrbpBWY0ml140UfeBD8SAa7tWMRNOfmiuC9YNMqwB4tx568A6PynyCW/G3FPAEroadpQY5oPbHjXdHunAaHvUZPZACMM3v7k49w9gdWdy2XwTDI7KvROxaciaKv+bhLhKkXbiqeNrhicOVvwEwmcbVRPrMLnT++7oGqF4dX7yalycigxiqKjXjVb3mpOtvxDDnhRaP3y40/tMK9t/crT8NCE+yG0YAY9VgsgLnGaaHz9blrUjHusYn66obp5Kars/gCnTYPURlen0BLwpcCCN7gUSlcw==

提取码:
123456

*有效期至: 2022/11/10 23:31:11 GMT-08:00
```

推理结果下载地址：
```
URL:
https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=IhRhNmLCO+ThRv/EeEc9K/+b2p3GjqHeFkayrRNOyT5tSAOxjIdgBOSLFZoZQDKkXdBFo5gDAgm6ixbIT4jwyU2eSI5Wesqf6nBb/ZUgdZVRMBusYIsXEK5DXn5+P42/pZe3hZZGLazXNggkA9IGzQYsjJ/46i7FB2NfZXvNpRhCk1IUjwYmCbZfyHrULHkT8dzv2MJpE6ueFHp2YzZqsqZEJbrnKYRmUMe6vxxjgoAMmgBQlXld7fR+J7yZXBW8r9CLM1BGiC0nxt220R+C/SQ7VgjbcrDTPgwqKuE2uaL0tDtQyiJk79okVMQuoooZUOcync/CX8dCV9Z1NsL7yWROlqgKrlkUL+08WECODtL8k0ZhszgthQjQecI2TcfQKuZWGNP7dXA0lZEBJpiKzUBz8h2nUE7pgssmnUoLU7eqRDr2LRjO59Op+oTg/CR8bkbJeCJa2kc4P5GRqb8somBhNJvgZf3GoXy0zzNmv5Q/wItr2Rz0gb9aFWf72EzboWfVFc95bNTxiIvbZFHsFZWj2CAGdpTglkVHT0Wz2J47ya+O15F9S0g3iDzGJHM7yTII2dPuei6+sJSojrWnQzx8eFxtQvHUOKOBEW1zimA=

提取码:
123456

*有效期至: 2022/11/10 23:32:39 GMT-08:00
```
推理精度：
```
5000张测试图像：
原始精度: 0.9902
推理精度: 0.9884
```

