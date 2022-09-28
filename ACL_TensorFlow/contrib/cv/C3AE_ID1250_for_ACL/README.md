# C3AE_ID1250_for_ACL

模型详情及训练部分：https://gitee.com/ascend/modelzoo/tree/master/built-in/TensorFlow/Official/nlp/BertTnews_for_TensorFlow

## 项目路径说明

```
C3AE_ID1250_for_ACL
├── LICENSE
├── README.md   
├── dataset    //数据集存放位置
├── detect     //数据预处理相关代码
├── atc.sh     //pb转om
├── h52pb.py 	 //h5转pb
├── img2bin.py      //图片转bin文件
├── modelzoo_level.txt 
├── c3ae_offline_inference.py  //离线推理
└── c3ae_online_inference.py   //在线推理
```

## 数据集

数据集使用wiki_crop，请将数据集解压到dataset下。

obs链接：obs://cann-id1250/dataset/wiki_crop.tar

## 模型文件

模型的h5文件、pb文件和om文件的obs链接：obs://cann-id1250/inference/

可通过h52pb.py完成h5到pb的格式转换。

## 生成om模型

使用ATC模型转换工具进行模型转换时可参考如下指令 atc.sh:

```
atc --model=./c3ae_npu_train_v2.pb --framework=3 --output=om_C3AE --soc_version=Ascend310 --input_shape="input_2:1,64,64,3;input_3:1,64,64,3;input_4:1,64,64,3"
```

## 数据集图片转bin文件

```
python img2bin.py
```

由于数据集图片需要经过人脸识别、图片剪裁等预处理过程，该步骤需要较长时间完成（取决于数据集大小）。本模型为多输入模型，图片将输出至同级目录下的"output1"、"output2"、"output3"三个文件夹下（请提前创建这三个目录）。

## 使用msame工具推理

使用msame工具进行推理时可参考如下指令，推理结果将默认以bin格式保存。

```
./msame --model ./om_C3AE.om --input ./output1,./output2,./output3 --output ./output/ 
```

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

## 离线推理

备注：不添加参数也可以，已设置默认参数

```
python3 ./c3ae_offline_inference.py --model_path ./om_C3AE.om --output_path ./output1/,./output2/,./output3/ --data_path ./dataset/wiki_crop --inference_path ./output/
```

## 在线推理

```
python ./c3ae_online_inference.py
```

## 预测精度

| Ascend910训练MAE（AGE） | Ascend310推理MAE（AGE） |
| ----------------------- | ----------------------- |
| 7.44                    | 7.47                    |

