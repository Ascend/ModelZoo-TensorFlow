推理情况表   
| 模型 |数据集| 输入shape | 输出shape | 推理时长(单张) | msame精度 | 目标精度 |
|--|--|--|---| -- | --| -- |
| efficientnet|Imagenet val  | `50*224*224*3` | `50*1000`  |98.18ms~ | 0.761 | 0.766| 

## 1、原始模型
找到对应文件夹,下载对应的`ckpt文件`，使用该文件夹下面的`converty.py`脚本转成pb模型。(链接：链接：https://pan.baidu.com/s/1rejQzGUunIfSbPpImzctsg 
提取码：n3xm)

## 2、转om模型

atc转换命令参考：

```sh
atc --model=efficientv2.pb  --framework=3 --input_shape="inputx:50,224,224,3" --output=./om_model/efficient2 --out_nodes="logits:0" --soc_version=Ascend310

```

## 3、编译msame推理工具
参考https://gitee.com/ascend/tools/tree/master/msame, 编译出msame推理工具


## 4、全量数据集精度测试：

测试集总共50000张图片，每50张图片一个bin
使用data_make.py脚本,设定好路径后,
执行 python3 data_make.py \

数据集下载地址
URL:obs://public-dataset/imagenet/orignal/valid_tf/


注：image将保存在xdata文件夹中 label将保存在ydata文件夹中
(也可下载已制作完成数据集 obs://efficientnet-v2/data/)

### 4.1 执行推理和精度计算

  
执行命令 `./msame --model /root/efficientv2/om_model/efficient2.om  --input  /root/efficientv2/data/xdata/ --output /root/efficientv2/data/ypre
`

最后执行python3  inference.py 
得到最后推理精度\

## 5、精度
![输入图片说明](https://images.gitee.com/uploads/images/2021/0918/120447_04e94974_8376014.png "efficientv2.PNG")

## 6、性能

![输入图片说明](https://images.gitee.com/uploads/images/2021/0918/120453_51fb259d_8376014.png "efficient_inference.PNG")