# TCN-KERAS
## 模型简介
https://gitee.com/ascend/ModelZoo-TensorFlow/blob/master/TensorFlow/contrib/cv/Keras-TCN_ID2665_for_TensorFlow/README.md

## pb模型
`bash freeze_graph.sh`
## om模型
使用ATC模型转换工具进行模型转换时可以参考如下指令:  
```
/home/HwHiAiUser/Ascend/ascend-toolkit/5.0.3.alpha001/x86_64-linux/atc/bin/atc 
--model=/root/msame/model/tcn.pb --framework=3 --output=/root/msame/model/tcn --input_shape="input_1:1,600,2" --soc_version=Ascend310 --precision_mode=allow_fp32_to_fp16
```
## 链接        
[GPU上得到的h5、pb及om 提取码：1234](https://pan.baidu.com/s/1Ix3yPq0C7fVm4f0pC-jUmA)     
[NPU上得到的h5、pb及om 提取码：1234](https://pan.baidu.com/s/16ZqMKL_jUNK3a-yGfKaqWQ)
## 使用msame推理   
参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，将待检测om文件放在model文件夹，然后进行推理测试。
## 性能测试  
数据转换bin文件：使用numpy.array的tofile函数   
[bin文件及其生成代码 提取码：1234](https://pan.baidu.com/s/1KTNYgG8VZzK9nUui3qJ_8A)  
使用msame推理工具，参考如下命令，发起推理性能测试：   
```
./msame --model "/root/msame/model/tcn.om" --output "/root/msame/out/" --input "/root/msame/mydata"  --outfmt TXT
```   
## 精度测试      
分别使用利用GPU和NPU生成的pb文件转om文件进行推理，结果对比如下：    

均方误差：
|GPU                      | NPU                        | 
|:-----------------------:| :----------------------:   | 
|0.1777398184542767       | 0.27024116107347707        | 

部分结果：
|序号    | 真实值              | GPU        |  NPU      |
|:------:| --------:          |  -----:    |  ----:    |
|1       | 1.4733921519255126 | 1.49268    | 1.46504   |
|2       | 0.8314698301349642 | 0.790464   | 0.755314  |
|3       | 0.8891396232000033 | 0.909286   | 0.928128  |
|4       | 1.0107378695468388 | 0.760409   | 0.703838  |
|5       | 0.9851944977337316 | 1.08851    | 1.46067   |
|6       | 1.689383561485223  | 0.940242   | 1.52666   |
|7       | 0.637807723655282  | 0.918035   | 1.43259   |
|8       | 0.5436813711189764 | 1.0083     | 0.960144  |
|9       | 1.7295318130559836 | 1.03426    | 1.18166   |
|10      | 1.1142951232471812 | 0.898364   | 1.38794   |

[完整结果 提取码：1234](https://pan.baidu.com/s/1iQfMOF2KUj6zDozJz3X6Ew)    

GPU与NPU结果余弦相似度：0.9550463  

## 性能测试 
* NPU
Inference average time : 3.31 ms
Inference average time without first time: 3.30 ms
* GPU
Inference average time : 5.27 ms
Inference average time without first time: 5.27 ms
