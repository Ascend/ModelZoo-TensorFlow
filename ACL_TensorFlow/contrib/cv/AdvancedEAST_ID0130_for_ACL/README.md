# AdvancedEAST
AdvancedEAST推理部分实现，模型概述详情请看AdvancedEAST_ID0130_for_TensorFlow README.md

## 训练环境

* TensorFlow 1.15.0
* Python 3.7.0

## 代码及路径解释

```
AdvancedEAST_ID0130_for_ACL
├── h5_to_pb.py             h5模型固化为pb
├── atc.sh  				act工具 pb==》om 转换命令
├── bin2predict.py 			推理数据后处理
├── msame.sh				msame工具：om离线推理命令
├── image2bin.py			推理数据预处理：将image_test中的image文件转换为bin并进行其他图片预处理
├── preprocess.py           图片预处理
├── nms.py                  预测用到的一个函数
├── cfg.py			        参数配置
├── image_test1				数据集位置		
│   └── ..			        评估数据集
├── image_test_bin		    图片转为bin存放位置		
│   └── ..			            
├── image_test_output		msame推理结果bin文件存放位置		
    └── ..			            
```


## 数据集
* tianchi ICPR MTWI 2018

测试集下载地址：链接：https://pan.baidu.com/s/1pU4TXFWfOoZxAmIeAx98dQ 提取码：1234

精度验证链接: https://tianchi.aliyun.com/competition/entrance/231685/rankingList

## 模型文件
包括初始h5文件，固化pb文件，以及推理om文件
链接：https://pan.baidu.com/s/13yqpnLBy6uoVjzhgge_Y6g 
提取码：1234

## pb模型

模型固化
```shell
python3.7.5 h5_to_pb.py
```
## 生成om模型

使用ATC模型转换工具进行模型转换时可参考如下指令 atc.sh:
```shell
atc --model=model.pb --input_shape="input_img:1,736,736,3" --framework=3 --output=./om/modelom --soc_version=Ascend310 --input_format=NHWC \
```
具体参数使用方法请查看官方文档。

## 将测试集图片转为bin文件

```shell
python3.7.5 image2bin.py
```
## 使用msame工具推理

使用msame工具进行推理时可参考如下指令 msame.sh
```shell
./msame --model $MODEL --input $INPUT --output $OUTPUT --outfmt BIN
```
参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

## 使用推理得到的bin文件进行推理
```shell
python3.7.5 bin2predict.py
```

## 精度
* Ascend910模型predict精度：

| Score    | Precision  | Recall    | 
| :--------: | ---------- | ------ |
|   0.582  | 0.762     | 0.471     | 

* Ascend310推理精度：

| Score    | Precision  | Recall    | 
| :--------: | ---------- | ------ |
|   0.582  | 0.762     | 0.471     | 

## 推理图片

* Input

![Image](pics/img_2.png)

* Output

![Image](pics/img_1.png)

## 推理数据对比

* cpu/gpu

![Image](pics/img_3.png)

* Ascend 310

![Image](pics/img_4.png)

## 推理性能：

![Image](pics/img.png)
