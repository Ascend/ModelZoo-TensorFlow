# 模型概述

详情请看**DeepAlignmentNetwork_ID0874_for_TensorFlow** README.md *概述*

#  数据集

- DAN采用的数据集是300W,
  - 训练集：afw + helen/trainset + lfpw/trainset
  - 测试集：CommonSet(helen/test + lfpw/testset) 和ChallengeSet(ibug)两种情况

1、用户自行准备好数据集。

2、数据集的处理可以参考 **DeepAlignmentNetwork_ID0874_for_TensorFlow** README.md *概述*  "模型来源"

# pb模型

模型文件

将需要固化的ckpt文件移动到ckpt文件夹下

```
python3.7.5 ckpt2pb.py --ckptdir=model/ckpt/model.ckpt --pbdir=model/pb/ --input_node_name=input --output_node_name=output --pb_name=test.pb --width=112 --height=112
```

参数解释：

```
--ckptdir 			ckpt文件路径 
--pbdir     		转换后的pb模型的保存路径
--pb_name			pb模型名字 默认：test
--input_node_name 	输入节点 默认：input
--output_node_names	输出节点 默认：output
--width
--heigh
```

链接：https://pan.baidu.com/s/1lxEQkvYODUMctJXy7b0RVw?pwd=4521 
提取码：4521

# om模型

使用ATC模型转换工具进行模型转换时可参考如下指令:

```
atc --model=model/pd/test.pb --input_shape="input:1,112,112,3" --framework=3 --output=model/om/test --soc_version=Ascend310 --input_format=NHWC
```

# 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

## 数据集转换bin

```
python3.7.5 img2bin.py --input=datasets/testset --output=bin/testset --width=112 --height=112
```

参数解释：

```
--input 	图片位置，文件夹或是单张图片
--output	bin文件位置，文件夹
--width		图片reshape：width
--height	图片reshape：height
```

## 推理

可参考如下命令 msame.sh：

```
msame --model=model/om/test.om --input=bin/testset --output=output/bin_infer/testset --outfmt=BIN
```

![image-20211202000624817](https://gitee.com/DatalyOne/picGo-image/raw/master/202112020006107.png)

## 推理结果后处理

## 测试精度

```
python3.7.5 offline_infer_acc.py --metric=3 --bin_dir=output/bin_infer/image --ptv_dir=datasets/testset
```

测试推理精度

参数解释：

```
--bin_dir 使用msame推理后生成的bin文件位置 默认：bin/testset
--ptv_dir 推理图片的Landmark文件(.ptv)位置  默认：datasets/testset
--metric  精度评价标准 指定均方误差(MSE)归一化因子 0:WITHOUT_NORM 1:OCULAR 2:PUPIL 3:DIAGONAL 默认：3
```

## 推理样例展示

```
python3.7.5 bin2image.py --input_bin=test/test.bin --input_img=test/test.png --output=test/test_out.png
```

展示推理所得的bin文件

参数解释：

```
--input 	推理所得的bin文件或文件夹 默认：test/test.bin
--input_img bin文件对应的图片或文件夹 默认：test/test.png
--output 	生成的展示图片的保存位置 默认：test/test_out.png
```

<img src="https://gitee.com/DatalyOne/picGo-image/raw/master/202112020013627.png" alt="134212_2@b15bdf12-4ade-11ec-8769-b46e082ef454_output_0" style="zoom:200%;" />
