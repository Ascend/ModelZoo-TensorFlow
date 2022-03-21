# AdvancedEAST
AdvancedEAST 是一种用于场景图像文本检测的算法，主要基于 EAST:An Efficient and Accurate Scene Text Detector，并进行了重大改进，使长文本预测更加准确。参考项目：https://github.com/huoyijie/AdvancedEAST

## 训练环境

* python 3.7.5+
* tensorflow-gpu 1.15.0+
* numpy 1.14.1+
* tqdm 4.19.7+

## 代码及路径解释

```
AdvancedEAST_ID0130_for_TensorFlow
├── advanced_east.py		GPU训练入口
├── cfg.py			        参数配置
├── icpr			        数据集
│   ├── image_10000		    图像文件
│   └── txt_10000		    标签文件
├── demo			        样例图片
│   ├── 001.png					
│   └── 004.png
│	└── ...
├── data			        数据集
│   └── image_test		    测试图像文件
├── model                   checkpoints
├── saved_model             保存的模型	
├── data_generator.py       数据生成
├── image_util.py           keras中的一些工具
├── keras_npu.py            npu训练入口
├── label.py                给图像打标签
├── losses.py		        损失函数
├── network_tensorflow_changeVGG_npu.py    模型结构
├── nms.py                  预测用到的一个函数
├── predict.py              预测函数
├── preprocess.py           图片预处理
├── LICENSE
```

## 数据集
```
选择使用 tianchi ICPR dataset
```

## training

* tianchi ICPR dataset download
链接: https://pan.baidu.com/s/1NSyc-cHKV3IwDo6qojIrKA 密码: ye9y    

* 数据预处理：
```bash
    $ python3 preprocess.py   
    $ python3 label.py
   ```
* 执行GPU训练：
```bash
    $ python3 advanced_east.py
```
* 执行NPU训练：
```bash
    $ python3 keras_npu.py
```
* 执行predict:
```bash
    $ python3 predict.py 
```
## 验证精度

* tianchi ICPR MTWI 2018

测试集下载地址：链接：https://pan.baidu.com/s/1pU4TXFWfOoZxAmIeAx98dQ 提取码：1234

精度验证链接: https://tianchi.aliyun.com/competition/entrance/231685/rankingList

## 模型文件

链接：https://pan.baidu.com/s/1csf-VEwEIF-P0pArvf9lnw 
提取码：7kru

## 精度

* 论文精度：

| Score    | Precision  | Recall    | 
| :--------: | ---------- | ------ |
|   0.611  | 0.809     | 0.492     | 

* GPU目标精度：

| Score    | Precision  | Recall    | 
| :--------: | ---------- | ------ |
|   0.554  | 0.760     | 0.436     | 

* Ascend精度：

| Score    | Precision  | Recall    | 
| :--------: | ---------- | ------ |
|   0.582  | 0.762     | 0.471     | 


## 性能对比：

| GPU V100  | Ascend 910 | 
| :--------: | --------| 
|   1057s/epoch  | 956s/epoch   | 


