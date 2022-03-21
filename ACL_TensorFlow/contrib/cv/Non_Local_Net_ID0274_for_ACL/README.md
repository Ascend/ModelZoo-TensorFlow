## 模型功能

 对Non_Local_Net的tensorflow模型的输出进行解析, 使用固化后Non_Local_Net模型进行推理，对mnist数据集进行图像分类。

## 原始模型

参考实现 ：
```
https://github.com/huyz1117/Non_Local_Net_TensorFlow
```

原始pb模型网络下载地址 :
```
https://no-local-net.obs.cn-north-4.myhuaweicloud.com:443/no-local-nets/output/V0129/Model/model.pb?AccessKeyId=YITM7NCJQWMO1NQIPCGJ&Expires=1670727865&Signature=k9iC43f7C1srh5YJvUOuy%2Bovb4w%3D
```


## om模型

om模型网络下载地址 :
```
https://no-local-net.obs.cn-north-4.myhuaweicloud.com:443/no-local-nets/output/V0129/Model/modelo_32.om?AccessKeyId=YITM7NCJQWMO1NQIPCGJ&Expires=1670727897&Signature=qdUmgii0xWtxFa1qObtB%2BNcQ1Yc%3D
```

使用ATC模型转换工具进行模型转换时可以参考如下指令:

```
atc --model=/home/HwHiAiUser/AscendProjects/Non_loca_net/input/model.pb  --framework=3 --output=/home/HwHiAiUser/AscendProjects/Non_loca_net/output/modelo_32 --soc_version=Ascend310  --input_shape="inputs:32,784" --input_format=NHWC --log=error --out_nodes="output:0"
```

## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

## 性能测试

使用msame推理工具，参考如下命令，发起推理性能测试： 

```
./msame --model "/home/HwHiAiUser/AscendProjects/Non_loca_net/output/modelo_32.om" --input  "/home/HwHiAiUser/AscendProjects/Non_loca_net/test/input/bin/pc_batchx.bin " --output "/home/HwHiAiUser/AscendProjects/Non_loca_net/test/output/"  --outfmt TXT --loop 5
```

```
Inference time: 679.432ms
[INFO] get max dynamic batch size success
[INFO] output data success
Inference average time: 679.892800 ms
Inference average time without first time: 679.650250 ms
[INFO] destroy model input success
[INFO] unload model success, model Id is 1
[INFO] Execute sample success
[INFO] end to destroy stream
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl
```

平均推理性能 679.432ms
## 精度测试

推理数据集(.bin)和数据label下载地址:

数据集
```
https://no-local-net.obs.cn-north-4.myhuaweicloud.com:443/no-local-nets/output/V0129/bin/pc_batchx.bin?AccessKeyId=YITM7NCJQWMO1NQIPCGJ&Expires=1670728004&Signature=iIcmIl0%2Bvey8hwsGIYWLo0tnKuM%3D

```

label
```
https://no-local-net.obs.cn-north-4.myhuaweicloud.com:443/no-local-nets/output/V0129/bin/pc_batchx_label.bin?AccessKeyId=YITM7NCJQWMO1NQIPCGJ&Expires=1670728046&Signature=xy%2BRLZfW4htZh/3JRIFAoFkvUtY%3D

单个数值
https://no-local-net.obs.cn-north-4.myhuaweicloud.com:443/no-local-nets/output/V0129/bin/label.npy?AccessKeyId=YITM7NCJQWMO1NQIPCGJ&Expires=1670728493&Signature=3YcA%2BeNkkToLzu3PRdSl4Fccnyw%3D

```

推理结果下载地址：
```
https://no-local-net.obs.cn-north-4.myhuaweicloud.com:443/no-local-nets/output/V0129/bin/modelo_32_output_0.txt?AccessKeyId=YITM7NCJQWMO1NQIPCGJ&Expires=1670728072&Signature=3v98WRFRbWQuD9RcRx8V0BOFAG0%3D
```
推理精度：
```
32张图片无误差
说明：因为训练批次问题，模型没有经过充分的训练(大概需要50000/batchsize个批次)，故选用训练过数据集进行测试，测试的结果和训练时候的精度相比没有任何差异。后续会将充足训练的过程补上。 
```

