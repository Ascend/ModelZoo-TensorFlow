## 模型功能

 对Rebar的tensorflow模型的输出进行解析, 使用固化后outom1021.om模型进行推理。

## 原始模型

参考实现 ：
```
https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/Rebar_ID2016_for_TensorFlow
```


原始ckpt模型网络下载地址 :
```
链接：https://pan.baidu.com/s/1sWukTQUzzdbqWbjqwkjhHg 
提取码：1111

```
原始pb模型网络下载地址 :
```
链接：https://pan.baidu.com/s/1cK8-e22Oi4zIl7Sc_U1qmg 
提取码：1111
```


## om模型

om模型网络下载地址 :
```
链接：https://pan.baidu.com/s/14wHs1QrGirvdmoICWnlniQ 
提取码：1111
```

使用ATC模型转换工具进行模型转换时可以参考如下指令:

```
atc --model=./frozen_model1025.pb  --framework=3  --input_shape="Placeholder_1:12,784" --output=./outom1021  --soc_version=Ascend910 
```

## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

## 性能测试

使用msame推理工具，参考如下命令，发起推理性能测试： 

```
./msame --model "./outom1021.om" --input  "/home/TestUser01/Pycode/rebar_npu/img2bin/bin" --output "./output"  --outfmt TXT --loop 5
```


