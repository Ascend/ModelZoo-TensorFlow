推理情况表   
| 模型 |数据集| 输入shape | 输出shape | 推理时长(单张) | msame LPIPS | NPU LPIPS |GPU LPIPS |
|--|--|--|---| -- | --| -- |-- |
| BicycleGAN | Google Maps Val  | `1*256*256*3`&`1*8` | `1*256*256*3`  |12.85ms~ | 0.526 | 0.393|0.412| 

## 1、原始模型
训练后保存ckpt文件，使用[模型训练代码](https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/BicycleGAN_ID1287_for_TensorFlow)下的`bicyclegan_pb_frozen.py`脚本转成pb模型。

## 2、转om模型

atc转换命令参考：

```sh
atc --model=bicyclegan.pb  --framework=3 --input_shape="input_images:1,256,256,3;latent_vector:1,8" --output=./om_model/bicyclegan --out_nodes="Generator/output:0" --soc_version=Ascend310
```

## 3、编译msame推理工具
参考https://gitee.com/ascend/tools/tree/master/msame, 编译出msame推理工具。


## 4、数据集预处理：

测试集总共1098张图片，每一张图片一个bin，BicycleGAN生成图片还需要Latent Vector(即z)，每张图片bin需要n(默认为20)个噪声bin。

使用`data_preprocess.py`脚本,设定好路径后,
执行`python3 data_preprocess.py`，
会在input文件中生成图片的bin，并在z文件夹下生成n个子文件夹，每个子文件夹中含有和input对应的噪声bin。


## 5、执行推理和精度计算
根据msame编译输出的位置以及数据，模型所在的位置，修改`./inference.sh`中相应位置，并执行`./inference.sh`
该命令主要功能就是加载om执行推理同时计算精度

本模型计算LPIPS精度需要联网下载权重，若下载失败，可到
[模型训练代码](https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/BicycleGAN_ID1287_for_TensorFlow)下weights文件夹中找到`net-lin_alex_v0.1.pb`和`net-lin_alex_v0.1_27.pb`，放在根目录下的weights文件夹中，最后执行`python3 ./eval.py --output_path ……`
重新得到推理精度
