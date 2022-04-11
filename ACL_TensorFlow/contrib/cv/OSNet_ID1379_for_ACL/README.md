## 模型功能

行人重识别（REID)

## 原始模型

参考论文：

[Omni-Scale Feature Learning for Person Re-Identification](https://arxiv.org/pdf/1905.00953.pdf)

原实现模型：

https://gitee.com/dw8023/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/OSNet_ID1379_for_TensorFlow

pb文件下载地址 :

链接：https://pan.baidu.com/s/1bD3MtUVUR8DLmJ3Xkk6dwg 
提取码：345r

## om模型

om模型下载地址：

链接：https://pan.baidu.com/s/1t2GLeCTI6URBW6x8AW-Ijw 
提取码：c90c

使用ATC模型转换工具进行模型转换时可以参考如下指令:

```
atc --model=/root/osnet/model/osnet.pb --framework=3 --output=/root/osnet/osnet --soc_version=Ascend310 --input_shape="input:1,128,64,3" 
```

## 数据集准备

Market1501原始验证集中的图像数据转换为bin文件参见img2bin.py文件：


bin格式数据集下载地址：

链接：https://pan.baidu.com/s/1h9bWDVEW7-voFHFi7TaTuw 
提取码：nwyj



## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，进行性能测试。



## 性能测试

使用msame推理工具，参考如下命令，发起推理性能测试： 

```
msame --model /root/osnet/osnet.om --input /root/osnet/query --output /root/osnet/output_query/ --outfmt TXT
msame --model /root/osnet/osnet.om --input /root/osnet/gallery --output /root/osnet/output_gallery/ --outfmt TXT
```

```
...
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
Inference average time : 3.70 ms
Inference average time without first time: 3.70 ms
[INFO] unload model success, model Id is 1
[INFO] Execute sample success
[INFO] end to destroy stream
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl
...
```

平均推理性能为 3.70ms

## 精度测试

执行精度对比文件：

```
python3 compare.py
```

最终精度：(暂未达标)

```
Ascend310推理结果：
    gpu结果:       
    npu结果:       
```





