## 模型功能

人脸识别


## pb模型
```

python3.7 ckpt2pb.py
```
pb模型获取链接：

obs://csgm-submit/离线推理/

## om模型

使用ATC模型转换工具进行模型转换时可以参考如下指令:

```
atc --model=/usr/model_test/frozen_model.pb 
--framework=3 
--output=/usr/model_test/frozen_model
--soc_version=Ascend310 
--out_nodes="gen_1/Sigmoid:0" 
--input_shape "x_ph:100,784;x_ph_1:100,20"
```

## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，进行推理测试。

## 数据集转换bin

```
python3.7 np2bin.py
```

## 推理测试

使用msame推理工具，参考如下命令，发起推理测试： 

```
./msame --model "/usr/model_test/frozen_model.om" 
--input "/usr/model_test/test_img.bin,/usr/model_test/test_z.bin" 
--output "/usr/model_test/output/" 
--outfmt TXT  
--loop 1

```

对验证集进行推理输出,计算精度

```
python3.7 recom_err.py
```


## 推理精度

|gpu|npu|推理|
|:----:|:----:|:----:|
|0.009|0.011|0.011|


## 推理性能
batch_size：100
![img_1.png](img_1.png)