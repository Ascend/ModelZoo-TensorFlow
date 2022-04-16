## 模型功能

目标跟踪

## 原始模型

参考：


原实现模型：

https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/DeepSort_ID0505_for_TensorFlow

pb文件下载地址 :

链接：https://pan.baidu.com/s/1v7Fe_YYT-hZUGCf2u7TI5g 
提取码：v41g

## om模型

om模型下载地址：

链接：https://pan.baidu.com/s/1GCuGdEUiniYZlTdFn2sknw 
提取码：keil

使用ATC模型转换工具进行模型转换时可以参考如下指令:

```
atc --model=/root/deepsort/deep_sort.pb --framework=3 --output=/root/deepsort/deep_sort --soc_version=Ascend310 --input_shape="input:1,128,64,3" 
```

## 数据集准备

market1501测试集中的图像数据转换为bin数据集 地址：


链接：https://pan.baidu.com/s/1h9bWDVEW7-voFHFi7TaTuw 提取码：nwyj


## 使用msame工具推理


参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。


获取到msame可执行文件之后，进行性能测试。

./msame --model "/root/deepsort/deep_sort.om" --input "/root/osnet/query" --output "/root/deepsort/out/" --outfmt TXT


## 性能测试

使用msame推理工具，参考如下命令，发起推理性能测试： 

```

```

```
...
Inference average time : 10.66 ms
Inference average time without first time: 10.66 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success
[INFO] end to destroy stream
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl

...
```

平均推理性能为 10.66ms

## 精度测试


```

```

最终精度：(暂无)

```
Ascend310推理结果：
    gpu结果:       
    npu结果:       
```





