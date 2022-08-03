推理情况表   
| 模型 |数据集| 输入shape | 输出shape | 推理时长(单张) | msame精度 | 目标精度 |
|--|--|--|---| -- | --| -- |
| efficientnet|Imagenet val  | `50*224*224*3` | `50*1000`  |98.18ms~ | 0.756 | 0.762| 

## 1、生成pb模型
找到对应文件夹,下载对应的`h5文件`，使用该文件夹下面的`h5_pb.py`脚本转成pb模型。(链接：https://pan.baidu.com/s/19yr692PX0ZTPDGr0guG8Kg 
提取码：h1is)

## 2、转om模型

atc转换命令参考：

```sh
atc --model=./faceboxes.pb --framework=3 --output=./faceboxes_base310 --soc_version=Ascend310         --input_shape="image_tensor:1,1024,1024,3"         --log=info          --out_nodes="nms/map/TensorArrayStack/TensorArrayGatherV3:0;nms/map/TensorArrayStack_1/TensorArrayGatherV3:0;nms/map/TensorArrayStack_2/TensorArrayGatherV3:0"
```

## 3、编译msame推理工具
参考https://gitee.com/ascend/tools/tree/master/msame, 编译出msame推理工具


## 4、全量数据集精度测试：

测试集总共5000张图片，每一张图片一个bin
使用data_make.py脚本,设定好路径后,
执行 python3 data_make.py \

数据集下载地址
URL:
obs://public-dataset/imagenet/orignal/valid_tf/

注：image将保存在xdata文件夹中 label将保存在ydata文件夹中
（已制作完成数据集下载链接：obs://efficientnet-v1-imagenet/inference_data/data/）

### 4.1 执行推理和精度计算

  
执行命令 `./msame --model "/home/test_user04/model_base310.om" --input "/home/test_user04/inference_data" --output "/home/test_user04/" --outfmt TXT  --outputSize "10000,10000,10000"
`

最后执行python3  inference.py 
得到最后推理精度\

## 5、精度

![输入图片说明](https://images.gitee.com/uploads/images/2021/0918/115655_f1c59afb_8376014.png "acc.PNG")

## 6、性能

![输入图片说明](https://images.gitee.com/uploads/images/2021/0918/115706_10dd82b7_8376014.png "时间.PNG")