# TCN-KERAS
## 模型简介
TCN = 1D fully-convolutional network (FCN) + casual convolution  
### task
The adding problem  
输入一个深度为2，长度为n的序列，其中一维是取值在[0,1]内的实数，另一维除了两个元素为1，其余元素均为0，将这两维的数字做内积，得到最终的结果是一个实数，取值范围是[0,2]。使用TCN网络预测最终的结果，与真实值比较，误差使用MSE衡量。
#### Input shape
3D tensor with shape `(batch_size, timesteps, input_dim)`.
#### Output shape
2D tensor with shape `(batch_size, nb_filters)`.
## 数据集
由`utils.py`生成  
Length of the adding problem data = 600  
\# of data in the train set = 200000  
\# of data in the validation set = 40000
## 原始模型
参考论文：  
http://export.arxiv.org/pdf/1803.01271   
参考实现:  
https://github.com/philipperemy/keras-tcn  
## 代码及路径解释  
```
├── train.py　　　　　　　　　　　　　　　//训练代码
├── utils.py　　　　　　　　　　　　　　　//数据生成代码
├── tcn.py　　　　　　　　　　　         //模型代码
├── freeze_graph.py　　　　　　　　　　　//固化代码
├── scripts
│    ├──train_1p_ci.sh                 //训练脚本
│    ├──freeze_graph.sh                //固化脚本
```
## 训练命令
`bash train_1p_ci.sh`
## GPU和NPU训练效果
* GPU   
[GPU训练精度性能完整数据及日志 提取码：1234](https://pan.baidu.com/s/1sHniPqIwLn7VC2lWdQK0wQ)
``` 
Epoch 1/50  391/391  - 345s 883ms/step - loss: 0.5514 - val_loss: 0.04493
Epoch 2/50  391/391  - 133s 340ms/step - loss: 0.0141 - val_loss: 0.0121
[...]
Epoch 50/50  391/391 - 136s 347ms/step - loss: 2.2935e-04 - val_loss: 2.1177e-04
```
loss：   
<div align=center>
<img src=https://s3.bmp.ovh/imgs/2021/09/92fc0134a77c3760.png />
</div>
val_loss:
<div align=center>
<img src=https://s3.bmp.ovh/imgs/2021/09/3367b1149e398fe8.png />
</div>     

* NPU  
[NPU训练精度性能完整数据及日志 提取码：1234](https://pan.baidu.com/s/18nDU6eFti0vAeTSapk_iVw)
```
Epoch 1/50  391/391  - 235s 601ms/step - loss: 0.3362 - val_loss: 0.1427
Epoch 2/50  391/391  - 58s 148ms/step - loss: 0.0232 - val_loss: 0.0094
[...]
Epoch 50/50  391/391 - 56s 143ms/step - loss: 4.7023e-04 - val_loss: 5.4636e-04
```
loss：
<div align=center>
<img src=https://s3.bmp.ovh/imgs/2021/09/3b969c09def2440a.png />
</div>
val_loss:
<div align=center>
<img src=https://s3.bmp.ovh/imgs/2021/09/79d457fbd8e533c6.png />
</div>

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

获取到msame可执行文件之后，将待检测om文件放在model文件夹，然后进行性能测试。
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