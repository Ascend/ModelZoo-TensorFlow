# STNet_ID2360_for_ACL

STNet推理部分实现，模型概述详情请看STNet_ID0130_for_TensorFlow README.md

## 训练环境

* TensorFlow 1.15.0
* Python 3.7.5

## 代码及路径解释

```
STNet_ID2360_for_ACL
├── LICENSE
├── README.md   
├── atc.sh     //pb=>om
├── freeze_graph.py //固化模型
├── img2bin.py      //图片转二进制
├── preprocess.py     //图片预处理
├── modelzoo_level.txt 
├── stnet_offline_inference.py  //离线推理
└── stnet_online_inference.py   //在线推理           
```

## 数据集

* MNIST 手写数据集

数据集下载地址：obs://stnet-id2360/dataset/

## 模型文件

包括ckpt文件，固化pb文件，以及推理om文件:
obs://stnet-id2360/npu/MA-new-STNet_ID2360_for_TensorFlow-01-20-14-07/output/


## pb模型

模型固化需要ckpt文件参照上述方式获取，重新构建网络需要引用stnet.py和tf_utils.py文件。ckpt_path参数指定ckpt文件夹路径。output_path指定固化后的pb文件输出路径。

```shell
python3 freeze_graph.py --ckpt_path ./ckpt/ --output_path ./output/
```

## 生成om模型

使用ATC模型转换工具进行模型转换时可参考如下指令 atc.sh:

```shell
atc --model=stnet.pb --framework=3 --output=stnet_om --soc_version=Ascend310 --input_shape="Placeholder:1,1600"
```

具体参数使用方法请查看官方文档。

## 图片预处理

从数据集中读取图片进行预处理，保存为灰度图

```shell 
python3 preprocess.py --total_num 200 --data_path ./dataset/data.npz
```

## 将测试集图片转为bin文件

```shell
python img2bin.py -i ./images -t float32 -o ./out -w 40 -h 40 -f GRAY -c [0.00392]
```

## 使用msame工具推理

使用msame工具进行推理时可参考如下指令 

```shell
./msame --model $MODEL --input $INPUT --output $OUTPUT --outfmt BIN
```

参考 <https://gitee.com/ascend/tools/tree/master/msame>, 获取msame推理工具及使用方法。

## 使用转换得到的bin文件进行在线推理

```shell
python3 stnet_online_inference.py --model_path ./output/stnet.pb --input_tensor_name Placeholder:0 --output_tensor_name output_logit:0 --data_url ./dataset/mnist_sequence1_sample_5distortions5x5.npz --inference_url ./out/
```

## 使用转换得到的bin文件进行离线推理

```shell
python3 stnet_offline_inference.py --model_path ./stnet_om.om --output_path ./out --data_path ./dataset/mnist_sequence1_sample_5distortions5x5.npz --inference_path ./inference_out/
```

## 精度

* Ascend910模型predict精度：

| Accuracy   |
| :--------: |
|   0.937    |

* Ascend310推理精度：

| Accuracy   |
| :--------: |
|   0.951    |



## 推理性能

![Image](images/inference.png)
