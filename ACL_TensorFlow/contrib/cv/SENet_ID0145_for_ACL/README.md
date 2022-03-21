## 模型功能

图像分类

## 原始模型

参考论文：

[Squeeze Excitation Networks](https://arxiv.org/abs/1709.01507)

原实现模型：

https://gitee.com/dw8023/modelzoo/tree/master/contrib/TensorFlow/Research/cv/senet/SE-ResNet110_tf_hw_dw8023

pb文件下载地址 :

链接：https://pan.baidu.com/s/1oNgePPR3VdIY2ukbm_NCHg 
提取码：wrti

## om模型

om模型下载地址：

链接：https://pan.baidu.com/s/1V9lec1wCkIA0ZhcWNI5aaQ 
提取码：by3k

使用ATC模型转换工具进行模型转换时可以参考如下指令:

```
atc --model=/home/HwHiAiUser/dingwei/senet.pb --framework=3 --output=/home/HwHiAiUser/dingwei/senet --soc_version=Ascend310 --input_shape="input:10000,32,32,3" 
```

## 数据集准备

cifar10原始验证集中的图像数据转换为bin文件以及获取label可参考如下代码：

```
train_x, train_y, test_x, test_y = prepare_data()
train_x, test_x = color_preprocessing(train_x, test_x)
test_batch_x = test_x[0: 10000]
test_batch_x.tofile(dst_path + "/" + "test.bin")
label = tf.argmax(test_y, 1)
with tf.Session() as sess:
    data_numpy = label.eval()
```

bin格式数据集下载地址：

链接：https://pan.baidu.com/s/14NhHC7qR4QP16k3kyPl1pA 
提取码：lcu2



## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，进行性能测试。



## 性能测试

使用msame推理工具，参考如下命令，发起推理性能测试： 

```
msame --model /home/HwHiAiUser/dingwei/senet.om --input /home/HwHiAiUser/dingwei/test.bin --output /home/HwHiAiUser/dingwei/output_bin/ --outfmt TXT
```

```
Inference time: 11604ms
[INFO] get max dynamic batch size success
[INFO] output data success
Inference average time: 11604.038000 ms
[INFO] destroy model input success
[INFO] unload model success, model Id is 1
[INFO] Execute sample success
[INFO] end to destroy stream
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl
```

平均推理性能为 1.16ms

## 精度测试

执行精度对比文件：

```
python3 compare.py
```

最终精度：

```
Ascend310推理结果：Totol pic num: 10000, Top1 error（%）: 5.48
    gpu结果:       Totol pic num: 10000, Top1 error（%）: 5.21
    npu结果:       Totol pic num: 10000, Top1 error（%）: 5.45
```





