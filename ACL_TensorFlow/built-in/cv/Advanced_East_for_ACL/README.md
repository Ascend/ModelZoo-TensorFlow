# Advanced_East_for_ACL
AdvancedEAST 是一种用于场景图像文本检测的算法，主要基于 EAST:An Efficient and Accurate Scene Text Detector，并进行了重大改进，使长文本预测更加准确。
## 训练环境

* TensorFlow 1.15.0+
* Python 3.7.0+

## 代码及路径解释

```
Advanced_EAST_for_ACL
├── nms.py                  预测用到的一个函数
├── cfg.py			        参数配置
├── cfg_bank.py			    参数配置
├── advanced_east.py	    参数配置
├── image_util.py		    参数配置
├── label.py			    参数配置
├── network_add_bn.py	    参数配置
├── icpr				    数据集位置		
│   └── image_10000	        图像文件
│   └── txt_10000	        标签文件
├── demo				    样例图片		
│   └── 001.png			        			        
├── image_test_bin		    图片转为bin存放位置					            
├── image_test_output		msame推理结果bin文件存放位置				
├── preprocess.py           图片预处理
├── image2bin.py			推理数据预处理：将image_test中的image文件转换为bin并进行其他图片预处理
├── h5_to_pb.py             h5模型固化为pb
├── atc.sh  				act工具 pb==》om 转换命令
├── msame.sh				msame工具：om离线推理命令         
├── postprocess.py 			后处理
├── predict.py 			    精度预测
```


## 数据集 tianchi ICPR MTWI 2018

测试集下载地址：链接： 提取码：1234

精度验证链接: https://tianchi.aliyun.com/competition/entrance/231685/rankingList

## 图片预处理
```shell
python3.7.5 preprocess.py
```

## 将测试集图片转为bin文件

```shell
python3.7.5 image2bin.py
```

## 模型文件
包括初始h5文件，固化pb文件，以及推理om文件
h5模型下载地址：链接： 提取码：
pb模型下载地址：链接： 提取码：

## pb模型

模型固化
```shell
python3.7.5 h5_to_pb.py
```
## 生成om模型

使用ATC模型转换工具进行模型转换时可参考如下指令 atc.sh:
```shell
atc --model=model.pb --input_shape="input_img:1,736,736,3" --framework=3 --output=model --soc_version=Ascend310 --input_format=NHWC 
```
具体参数使用方法请查看官方文档。

## 使用msame工具推理

使用msame工具进行推理时可参考如下指令 msame.sh
```shell
./msame --model $MODEL --input $INPUT --output $OUTPUT --outfmt BIN
```
参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

## 执行predict
```shell
python3.7.5 predict.py
```

## 精度
* 论文精度：

| Score    | Precision  | Recall    | 
| :--------: | ---------- | ------ |
|   0.611  | 0.809     | 0.492     | 

* GPU目标精度：

| Score    | Precision  | Recall    | 
| :--------: | ---------- | ------ |
|   0.554  | 0.760     | 0.436     | 

* Ascend推理精度：

| Score    | Precision  | Recall    | 
| :--------: | ---------- | ------ |
|   0.632  | 0.849     | 0.513     | 


