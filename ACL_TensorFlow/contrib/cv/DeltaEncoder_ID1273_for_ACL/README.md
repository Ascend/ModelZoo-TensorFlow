## 模型功能

 对delta-encoder的tensorflow模型的输出进行解析, 使用固化后delta-encoder模型分类验证部分进行推理，使用VGG编码后的miniImageNet数据集（每张图像被编码成2048维的特征向量）, 进行 1-shot  5-way （训练集最初只有五类，每类一张，通过delta-encoder模型生成后训练集每类合成1000个，共5000个新图像向量，验证集由miniImageNet中属于训练集五类中的图像向量构成，共3000个）分类。

## 原始模型

参考实现 ：
```
https://github.com/EliSchwartz/DeltaEncoder
```
原始ckpt模型网络下载地址 :
```
URL:
https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=IhRhNmLCO+ThRv/EeEc9K/+b2p3GjqHeFkayrRNOyT5tSAOxjIdgBOSLFZoZQDKkXdBFo5gDAgm6ixbIT4jwyU2eSI5Wesqf6nBb/ZUgdZVRMBusYIsXEK5DXn5+P42/pZe3hZZGLazXNggkA9IGzQYsjJ/46i7FB2NfZXvNpRhCk1IUjwYmCbZfyHrULHkT8dzv2MJpE6ueFHp2YzZqsn7CbhGt0OtxEkgvuCvqKXyAxIQpK3WGk39Ew+qXEHF5fbJaAJuYmGDVST24iqb89+UrJSXj+CmpTIrRqDPZ51jkhb+4vK5twzO9AKJ9DWqf62nTYTi0W6feAKf0OfVsuR/hKbO4YJnLSpestfCHEDqEDRhpJjvBpSr/aAzeZwTXpbWWgevyqqWE2MBjGjPeHVC3xUtWcWZpa6Z/gf0ZH3vWIlfaMS7Iu1+OOHs6UmMZKnnDb6HXgfG74/HiP7FW1xby4qQ3pddmf8D+OrPAv70uj8xmcMO7fb23MD2vfbWSLT8HBpDpr4B1Nf8lUJ968aKB9dZVtfBjvm0MLRZZJG5MQCWuxjnE8CZbflF1A93a0ydtDVETIN+FnEu57ZWR85juRgpKBN+IjMrjJV+z+oNJmL68zabQ+dz9a74f3yHJWwn+BmmOStrEghK4pGYEvQ==

提取码:
123456

*有效期至: 2022/12/12 00:16:53 GMT-08:00
```

原始pb模型网络下载地址 :
```
https://yfiles.obs.cn-north-4.myhuaweicloud.com:443/delta_encoder_infer/model/delta_encoder_net.pb?AccessKeyId=TRUQTC6GBQWLPC2LQEAW&Expires=1670832914&Signature=6XRaDIe5qe1qf9aoxijlotZ2mXM%3D
```


## om模型

原始om模型网络下载地址 :
```
https://yfiles.obs.cn-north-4.myhuaweicloud.com:443/delta_encoder_infer/model/delta_encoder_.om?AccessKeyId=TRUQTC6GBQWLPC2LQEAW&Expires=1670832938&Signature=a29uYCwFS1M5JOeBcHr6SdXxku8%3D
```

使用ATC模型转换工具进行模型转换时可以参考如下指令:

```
atc --model=./delta_encoder_net.pb --framework=3 --output=./delta_encoder_  --soc_version=Ascend310  --input_shape="input:1,2048"
```

## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，将待检测om文件放在model文件夹，然后进行性能测试。

## 性能测试

使用msame推理工具，参考如下命令，发起推理性能测试： 

```
./msame --model "./delta_encoder_.om" --input "./bin_data/" --output "./out/" --outfmt TXT --loop 1
```
以上bin_data为bin格式的待推理数据，out为推理结果输出路径

```
Inference average time: 0.71 ms
Inference average time without first time: 0.71 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success.
[INFO] end to destroy stream
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl
```

平均推理性能 0.71 ms

## 精度测试

推理数据集(.bin)和数据label下载地址:

数据集
```
URL:
https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=IhRhNmLCO+ThRv/EeEc9K/+b2p3GjqHeFkayrRNOyT5tSAOxjIdgBOSLFZoZQDKkXdBFo5gDAgm6ixbIT4jwyU2eSI5Wesqf6nBb/ZUgdZVRMBusYIsXEK5DXn5+P42/pZe3hZZGLazXNggkA9IGzQYsjJ/46i7FB2NfZXvNpRhCk1IUjwYmCbZfyHrULHkT8dzv2MJpE6ueFHp2YzZqsi3P6zdcNlchuaiCGxKhhAxnulX23pz2062tPAthvCO37LrOEQzFkKBEo/E2z/PY91UXQq/CzEYGf5s/8WGI+Py2OPpFjzB1ZpeWYDal95fH572HGkdVR5UIrpvzAF57Ild+7UAZEnY99DlqWsTkfaDVVgLaRqMvUPi3UHZ/PSHLy0tKvhucdnpOkjGJnbgx+rW8r0ET9DWNI27rA4L7hMJpJ5FhQaDjIFYYPI/DYy2vaLEx98hY3j3C0FHss6RdaVNrm73CbWo/pxsN9luXyKex4IF5UltK3T8Fbul7PUjukb6X1bUvdPne7s1IJ4Jgt+H74WHixn39FzZbASYkyqSewsTPF+R+Tv481CsFDySB4CJl7MVAf6nVpc7RA+flaJN5sU0OdvS9tUkpMx/8wim3ID2evHS1JVC0YnSuGPOz

提取码:
123456

*有效期至: 2022/12/12 00:21:15 GMT-08:00
```

label
```
https://yfiles.obs.cn-north-4.myhuaweicloud.com:443/delta_encoder_infer/dataset/test_labels.npy?AccessKeyId=TRUQTC6GBQWLPC2LQEAW&Expires=1670833305&Signature=d2bPzMwEEQAOxDvH2YkSsRJwn1w%3D
```

推理结果下载地址：
```
URL:
https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=IhRhNmLCO+ThRv/EeEc9K/+b2p3GjqHeFkayrRNOyT5tSAOxjIdgBOSLFZoZQDKkXdBFo5gDAgm6ixbIT4jwyU2eSI5Wesqf6nBb/ZUgdZVRMBusYIsXEK5DXn5+P42/pZe3hZZGLazXNggkA9IGzQYsjJ/46i7FB2NfZXvNpRhCk1IUjwYmCbZfyHrULHkT8dzv2MJpE6ueFHp2YzZqsi3P6zdcNlchuaiCGxKhhAwM4tQlyQcMPn+Q/q1wBCx0Og2dQLtahOXLy8ezR3SoJdIQlR2XY0gmP641zfGTl+OFAuPRiGz/4B424qTHf4MfygpEectoHANfzMry24+HrbFofdMNWMyEh9TLEa6d7MqsSMNijTcMktKzg/jUAnNOCj50Ps/xVH2WvOQaP3xN4ibrImDrPGEDz56I4J9MKrx17zQ8+9E8i1IjRlowqIuVMhHzK8iBr2xkXKtLKqk3hsFg4SUkcsrcL4hkQVnDr6QILtHc2PmUDAYpmIj1EaNoD2vLBoWgaTkgGEWukfu273jfe96jcAuY1m365bueMcFwOMB8nJsaiACTQI2vgN7GCc8VUyZjJy4YIzCpexHn6sMcOF1MTbWKVb1Ic9voPLgzzZPcG1MfsH4i8gDVEjeg

提取码:
123456

*有效期至: 2022/12/12 00:20:30 GMT-08:00
```
推理精度：
```
3000张测试图像向量：
论文精度: 0.599
推理精度: 0.731
```

