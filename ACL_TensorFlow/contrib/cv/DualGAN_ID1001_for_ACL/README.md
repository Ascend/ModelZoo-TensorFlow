# 模型功能
现有的图像之间转换的方法，大部分都是需要图像对的方法，但是实际上有的场景下，很难得到这样的图像对。DualGAN使用对偶学习模式的GAN网络结构来进行image to image translation，将 domain A 到 domain B 之间的转换，构成一个闭环（loop）。通过 minimize 该图和重构图像之间的 loss 来优化学习的目标：
给定一个 domain image A，用一个产生器 P 来生成对应的 domain image B，由于没有和A匹配的图像对，这里是没有ground truth的。如果该图伪造的很好，那么反过来，用另一个产生器 Q，应该可以很好的恢复出该图，即Q(P(A, z), z') 应该和 A 是类似的。对于 domain image B 也是如此。

- 参考论文：

    https://arxiv.org/abs/1704.02510

- 参考实现：

    https://github.com/duxingren14/DualGAN 

# pb模型

运行freeze_graph.py会得到A2B.pb和B2A.pb两个模型文件
```
python freeze_graph.py --ckpt_path=./ckpt/DualNet.model-99002

```

# om模型

在Ascend310推理服务器下，使用ATC模型转换工具进行模型转换:

A2B模型转换
```
atc --model=/root/dualgan/pb_model/A2B.pb --framework=3 --output=/root/dualgan/A2B --soc_version=Ascend310 --input_shape="img:1,256,256,3" --log=info --out_nodes="output:0" --op_select_implmode=high_precision
```

B2A模型转换
```
atc --model=/root/dualgan/pb_model/B2A.pb --framework=3 --output=/root/dualgan/B2A --soc_version=Ascend310 --input_shape="img:1,256,256,3" --log=info --out_nodes="output:0" --op_select_implmode=high_precision
```

推理使用的pb及om模型  [在此获取](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=kgEgUsFhjaxRtkatN8fRvFV+TH374pu2rOvPEqrVTMRgxZIZPaP/OO/LcJC/breBuJlVBpbluvxL9zIZvNovzuukk+juP5i+2xTxKhOtHcT+Xt4y+MzbBgqactNNZAkAFvv7b2zcldOJ7qYzJYin/qAEOMLpR50fwk4aPXCO6HwZXvLtgrUhilWNluGpW2QxcS7gj9uiBXF8nUcl+jhrO+v2k0zNbhvI7ISO/0IORuHbyNFKtmagrqIrQ5CsWiuOXc18LwvcO3CPUFk4kHG7/fgr6UyOaN3fMY8wtva+0pvBueGzCkr8+kPBbn5h7/lFOCHaL5P5XYV9qj237hA3LXVOdgkT4+gqsovjYAASlcra9YatkRMj3wimMEpOEZ/gIxpY00UNtxDaz0uF5QCjhbvxUELTo+iUlTmJ3fNresKRkR+3IffNYPoQbjqVYpxCO8qAIeRKD55PbKv8iP4I5LGzS1Smf00LZroDN5gdwe7e3QMgAZFKZEMVghh2Hqh87pLv00FXN8IxmFDVeVNfnjlst0/I0vyDyJDUYUejogQ=)  （提取码：123456）

# 使用msame工具推理

参考 https://gitee.com/ascend/tools/tree/master/msame, 获取msame推理工具及使用方法。

获取到msame可执行文件之后，进行推理测试。

## 1.数据集转换bin

data_dir=测试集路径

data_bin_dir=转化后的bin数据集路径
```
python3 jpg2bin.py --datasets_dir=data_dir --output_dir=data_bin_dir
```

## 2.推理

使用A2B.om模型推理（input为输入数据路径，output为输出数据路径）
```
./msame --model /root/dualgan/A2B.om --input "/root/dualgan/data_bin" --output "/root/dualgan/outbin" --outfmt BIN
```

使用B2A.om模型推理
```
./msame --model /root/dualgan/B2A.om --input "/root/dualgan/data_bin" --output "/root/dualgan/outbin" --outfmt BIN
```

## 3.推理结果后处理

推理结果为二进制.bin格式数据，需要将其转换为可视的.jpg格式图片。

input_dir=输入.bin数据路径

output_dir=输出路径
```
python3 bin2jpg.py --data_dir=input_dir   --dst_dir=output_dir
```