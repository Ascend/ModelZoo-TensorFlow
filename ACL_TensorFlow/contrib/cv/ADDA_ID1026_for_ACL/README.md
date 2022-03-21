# ADDA

## 模型功能

ADDA模型是在《Adversarial Discriminative Domain Adaptation》（2017 CVPR）中提出来的，其中作者结合了discriminative模型，untie weight sharing以及GAN 损失。利用源域的标签学习具有判别性的representation，之后通过域对抗损失和不对称映射将目标域的数据映射到相同的空间中，最终使得目标域的数据得到了良好的分类效果。

## 原始模型

参考实现 ：[https://github.com/erictzeng/adda](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Ferictzeng%2Fadda)

由自己训练的权重生成pb模型：

mnist：数据集

train：数据集训练集

lenet：model的name

snapshot/adda_lenet_svhn_mnist：模型保存训练权重的路径

```bash
python pbmake.py mnist train lenet snapshot/adda_lenet_svhn_mnist
python pbmake.py mnist2000 train lenet snapshot/adda_lenet_usps_mnist
python pbmake.py usps1800 train lenet snapshot/adda_lenet_mnist_usps
```

pb模型获取链接：

百度网盘链接：https://pan.baidu.com/s/1yyK4hs7r3qkK8GFjS-19gQ 
提取码：9lvu

## om模型

om模型

- 百度网盘链接：https://pan.baidu.com/s/1qSteCi-Ubexk2nip__uSWA 
提取码：0xuh

使用ATC模型转换工具进行模型转换时可以参考如下指令，具体操作详情和参数设置可以参考  
- [ATC工具使用指导 - Atlas 200dk](https://support.huaweicloud.com/ti-atc-A200dk_3000/altasatc_16_002.html) 
- [ATC工具使用环境搭建_昇腾CANN社区版(5.0.3.alpha002)(推理)_ATC模型转换_华为云](https://support.huaweicloud.com/atctool-cann503alpha2infer/atlasatc_16_0004.html)

命令行示例：

```ba
atc --model=/home/HwHiAiUser/pb_model/svhn_mnist.pb --framework=3 --output=/home/HwHiAiUser/om_model/svhn_mnist --soc_version=Ascend310 --out_nodes="output:0" --input_shape="input_image:1,28,28,1"
```

## 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，将待检测om文件放在model文件夹，然后进行性能测试。

## 性能测试

使用msame推理工具，参考如下命令，发起推理性能： 

```
./msame --model "/home/HwHiAiUser/om_model/svhn_mnist.om" --output "/home/out/" --outfmt TXT --loop 1
```


Batch: 1, shape: 28,28,1， 推理性能 4.547ms

## 精度测试
- 数据集准备。

  a.请用户自行准备好数据集，包含训练集和验证集两部分，数据集包括Mnist、usps、svhn等，包含train和 	val两部分。以Mnist数据集为例。

  b.上传数据压缩包到训练环境上,无需解压

  ├── tools/data/mnist
  │   ├──t10k-images-idx3-ubyte.gz
  │   ├──t10k-labels-idx1-ubyte.gz
  │   ├──train-images-idx3-ubyte.gz
  │   ├──train-labels-idx1-ubyte.gz
  
- 生成数据

  mnist：需要转bin格式的数据集

  train：数据集的训练集

  lenet：模型name
```bash
python data_bin.py mnist train lenet
```
生成的bin文件存储在当前目录的image文件夹下。

- om模型推理

  model：om模型路径

  output：推理结果输出路径

```bash
./msame --model "/home/HwHiAiUser/om_model/svhn_mnist.om" --output "/home/out/" --outfmt TXT --loop 1
./msame --model "/home/HwHiAiUser/om_model/usps_mnist.om" --output "/home/out/" --outfmt TXT --loop 1
./msame --model "/home/HwHiAiUser/om_model/mnist_usps.om" --output "/home/out/" --outfmt TXT --loop 1
```
Batch: 1, shape: 28,28,1， 推理性能 1.18ms

## 推理精度

|                | 推理    | 论文        |
| -------------- | ------- | ----------- |
| ACC_svhn_mnist | 0.76655 | 0.760±0.018 |
| ACC_usps_mnist | 0.922   | 0.901±0.008 |
| ACC_mnist_usps | 0.89333 | 0.894±0.002 |

