## 模型功能  

- 基于TensorFlow框架的FusionGAN红外和可见光图像融合网络训练代码。FusionGAN提出了一种生成式对抗网络融合这可见光和红外信息，通过建立一个生成器和鉴别器之间的对抗博弈，其中生成器的目标是生成具有主要红外强度和附加可见光梯度的融合图像，而鉴别器的目标是迫使融合图像具有更多可见图像中的细节。
- 论文地址：Ma J ,  Wei Y ,  Liang P , et al. FusionGAN: A generative adversarial network for infrared and visible image fusion[J]. Information Fusion, 2019, 48:11-26.（https://www.researchgate.net/publication/

- 参考实现 ：https://github.com/yugezi0128/FusionGAN


## ckpt模型
1. 找到对应文件夹,下载对应的ckpt文件,下载到本地。
链接:https://pan.baidu.com/s/18XyRpdWTUhD3P6qTQb1B2Q?pwd=r6c5 
提取码:r6c5

## pb模型
```
python3.7  ckpt2pb.py
```
1. 找到对应文件夹,下载对应的`ckpt文件`，使用该文件夹下面的`ckpt2pb.py`脚本转成pb模型。
2. 为了保证之后顺利执行om，需要将一些图节点命名进行修正。将第一步产生的pb文件执行check_pb.py得到能够正确使用的pb模型。
pb文件在如下连接
链接:https://pan.baidu.com/s/18XyRpdWTUhD3P6qTQb1B2Q?pwd=r6c5 
提取码:r6c5

## om模型
atc转换命令参考：
```sh
atc --model=./good_frozen11.pb --framework=3 --input_shape="images_ir:1,282,372,1;images_vi:1,282,372,1" --output=./toom11 --soc_version=Ascend310
```

om模型链接:https://pan.baidu.com/s/18XyRpdWTUhD3P6qTQb1B2Q?pwd=r6c5 
提取码:r6c5
##  编译msame推理工具
-参考https://gitee.com/ascend/tools/tree/master/msame, 编译出msame推理工具

## 全量数据集精度测试：
测试集总共32张图片，每1张图片一个bin
使用bmp2bin.py脚本,设定好路径后,
执行 python3 bmp2bin.py \
数据集下载地址
URL:https://figshare.com/articles/TNO_Image_Fusion_Dataset/1008029.

(也可下载已制作完成数据集 obs://deserant-fusiongan/dataset/Test_img/Nato_camp/)
## 执行推理和精度计算
  
1. 执行命令 `./msame --model "./toom8.om" --input "./Nato_camp/vi,./Nato_camp/ir" --output "./out/" --outfmt TXT --debug true
`
2. 执行python3 txt2bmp.py 将推理产生的txt数据进行处理并转成bmp图片格式。

3. 修改转好的bmp路径，执行compute_metrics.py可以得到最后推理精度。

## 离线推理精度

|     |  论文精度	| GPU精度  |  NPU精度  | 推理精度|
|  ----     |  ----     |  ----    |   ----   |  ----  |
| EN(主)   | 7.0618 | 6.9749|6.9439 |7.228|
| SSIM(主) | 0.6349 |0.6402 |0.6647 |0.5942|
| CC |0.6425 |0.7215|0.7664 |0.7459|
| SF |7.3528|0.7215 |0.7664 |0.7459|
| VIF |0.3512|0.434 |0.4273 |0.3857|
## 离线推理性能
NPU平均1.5s输出一个测试图，离线推理平均60ms输出一个测试图。
```sh
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:./Nato_camp/vi/29.bin
[INFO] start to process file:./Nato_camp/ir/29.bin
[INFO] model execute success
Inference time: 60.107ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:./Nato_camp/vi/30.bin
[INFO] start to process file:./Nato_camp/ir/30.bin
[INFO] model execute success
Inference time: 60.492ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:./Nato_camp/vi/31.bin
[INFO] start to process file:./Nato_camp/ir/31.bin
[INFO] model execute success
Inference time: 59.43ms
[INFO] get max dynamic batch size success
[INFO] output data success
[INFO] destroy model input success
[INFO] start to process file:./Nato_camp/vi/32.bin
[INFO] start to process file:./Nato_camp/ir/32.bin
```
