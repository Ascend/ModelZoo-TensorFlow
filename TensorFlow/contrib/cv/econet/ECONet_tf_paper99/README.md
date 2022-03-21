# ECONet

该仓库为*ECO: Efficient Convolutional Network for Online Video Understanding*的NPU版(tf1.15)代码实现。经测试，该代码在Ascend 910平台ascend-toolkit (version: 3.3.0.alpha003) 下可以正常运行。

## 模型简介

![econet_arch](figs/econet_arch.png)

网络结构示意图如上，具体模型原理和细节可参考论文：[Zolfaghari *etal*, ECO: Efficient Convolutional Network for Online Video Understanding, ECCV 2018](https://arxiv.org/pdf/1804.09066.pdf)

作者论文的官方代码实现：[[Caffe版]](https://github.com/mzolfaghari/ECO-efficient-video-understanding)(原始版本) [[Pytorch版]](https://github.com/mzolfaghari/ECO-pytorch)

该仓库中ECONet网络实现在`model/econet.py`中。

## 准备

### 环境准备

- Python 3.7
- Tensorflow 1.15.0
- PIL
- tqdm
- numpy
- Ascend 910
- ascend-toolkit >= 3.3.0.alpha003



### 数据准备

在准备开始训练或测试前，请**下载并解压好**抽帧完成的ucf101和hmdb51数据集，抽帧及切分数据集方式参考了[mmaction2](https://github.com/open-mmlab/mmaction2)，数据集链接如下：

**注意**:将数据解压至`./data`文件夹中。



### 预训练模型准备

如论文所述，由于需要ECONet在Kinetics数据集预训练好的模型进行参数初始化来训练ucf101和hmdb51两个数据集，因此需要**下载并解压**该模型文件。下载地址：[[OBS link]](https://econet.obs.myhuaweicloud.com:443/ECONet_public_data/pretrained_models/ECOfull.zip?AccessKeyId=PIURBFT7UIAIRJTMUSDN&Expires=1650094399&Signature=ZGNh5l6nwNNVcDHDKnarO4oyup4%3D)  [[Baiduyun link]](https://pan.baidu.com/s/1Hmgy_e9iVIm3hAWSKYbNJA) (passwd: klgc)

**注意**:将预训练模型解压至`./experiments`文件夹中。



## 代码结构及内容

执行完上述准备步骤后，代码目录结构展示如下（只展示到2级目录）

```shell
.                                      
    ├── dataset.py                                              # dataset信息提取
    ├── figs
    │   └── econet_arch.png
    ├── model                                                   # 包括模型相关文件
    │   ├── econet.py                                              # ECONet网络实现
    │   ├── __init__.py   
    │   └── model_infos                                            # 存放ECONet网络相关信息
    ├── data                                                    # 存放数据集
    │   ├── hmdb51_extracted                                       # hmdb51解压后文件夹
    │   └── ucf101_extracted                                       # ucf101解压后文件夹
    ├── experiments                                             # 用来存放模型及实验相关文件
    │   └── ECOfull                                                # 预训练模型解压后的文件夹
    ├── opts.py                                                 # 训练/测试参数设置
    ├── README.md 
    ├── run.sh                                                  # 训练及测试启动脚本
    ├── scripts                                                 # 包括离线推理需要使用的多个脚本
    │   ├── convert_ckpt2pb.py                                     # CKPT模型文件转换为PB模型文件
    │   ├── convert_data2bin.py                                    # 图像格式数据转换为二进制文件
    │   ├── offline_inference.sh                                   # 使用om模型进行离线推理脚本
    │   └── get_offline_acc.py                                     # 获取离线推理准确率
    ├── splits_txt                                              # 该目录存放ucf101和hmdb51的训练验证集切分信息
    │   ├── hmdb51
    │   └── ucf101                                                         
    ├── test.py                                                 # 测试脚本
    ├── train.py                                                # 训练脚本
    └── transforms.py                                           # 包含数据预处理相关操作
```



## 代码运行步骤

> **注意**: 该复现过程涉及两个数据集（ucf101/hmdb51），需要分别针对两个数据集执行下述流程

### 1. 在NPU上启动训练

在训练前确认`run.sh`中数据集的对应路径，执行下述两行中其一以启动针对相关数据集的训练

```shell
# 训练ucf101
sh run.sh train ucf101
# 或者训练hmdb51
sh run.sh train hmdb51
```

训练好的模型及log信息保存在`experiments/`的对应文件夹中，文件夹名称为训练脚本启动时间。

在npu上训练好的**ckpt模型**可根据下述链接下载：[[OBS link]](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=SSvMJBoCzDovvHhIh2gPGmIi3bKDm4FAmrNJyRXADHX4QqS41AiGMsoTK/hWaqLqALugAsgITdpMGELCPAYWii0ZyhyrNiIbc6Oxlqz0T81/RuzvB32/NqxyzjZ2tdqjdZpbYyoyhBXsqAzGRE0ftwAZItqpVB1/jjctdKOuuDuLlqycDrZDwgL7fWbHhR1h49FNvbeGwdUOii/5gKHk7OdfACNAdvLdpXeuMqujDIp70/sIiafrprjOF3nf6PpRMswtMIjpx6OIjCCVdcW+lycdqL7KN3rQ/fU3yIWFqtGjDv+oJwzI2muCo9QyWd07X6gHVXApIrf/j5BmF7AppOW7gVhQJ2vVVkqpONdY6aK8ME89d1EYzn284F8koomRFXpbDhRjicoawy1LoNDuLxbwsx2PMuFU21yWK1VXzOL19ogfmBFH8q4I+1ZL+tsBekF4OazLAlGZYsZO6zATfsQXOFSBUFbCrE7UF4ht/zqiJo/+UgabfDWTTF+W+1Hx3o07ju9eQRrKz8qQQmKZ/X48zpILjwwbYi/xMkiUSw92dMNOx6AAe+mk5ZTsVZFxh+A/OkokwxXuixt7QTOar8Bf/qu1Wfk1Yk/Z+raISSTo1RKgPCP9lVeG6kG4EI4A) (passwd: nankai) [[Baiduyun link]](https://pan.baidu.com/s/1D983t6EXHWqQj477W22_Ig) (passwd: 8m27)



### 2. 准备离线推理环节

#### PB模型转换

运行`scripts/convert_ckpt2pb.py`脚本，来将上一步中训练好的ckpt模型转换为pb模型

**参数说明：**

```shell
--dataset         # ucf101 或者 hmdb51
--ckpt_path       # 指定欲转换ckpt模型文件的路径
--output_name     # 指定输出pb模型文件的名称
```

**示例（ucf101）：**

```shell
python3.7 scripts/convert_ckpt2pb.py --dataset ucf101 --ckpt_path experiments/ucf101_best/ckpt/best.ckpt --output_name ucf101_best.pb
```

在执行完后，转换成功的模型将会出现在  `pb_model/ucf101_best.pb`。

转换好的**pb模型**可根据下述链接下载：[[OBS link]](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=SSvMJBoCzDovvHhIh2gPGmIi3bKDm4FAmrNJyRXADHX4QqS41AiGMsoTK/hWaqLqALugAsgITdpMGELCPAYWii0ZyhyrNiIbc6Oxlqz0T81/RuzvB32/NqxyzjZ2tdqjdZpbYyoyhBXsqAzGRE0ftwAZItqpVB1/jjctdKOuuDuLlqycDrZDwgL7fWbHhR1h49FNvbeGwdUOii/5gKHk7JYLlSv/Oi1RIyxPJoFPokjp8nBv3yLzNBtXCTujvboO/BTHyA5BIuKeFZ8tK+th8Gf9UGnRw8hSRBewZXWW5M5HBwo7h4AqPM966noRc7XdAJbfZhQKBYYUY+D0KMuXMN4gI+3dOPDBPFF8iXnwdzv4fu/Y58CZWPIBZy0JQM1q/MVu+ig3q5gdG3JlpBa7DJGFPAgndGDJCWiKiO9t9AYMHSEVBeraYHpQ6SMj/IV/oZqEVthy6HHUVCselxfFyfB9DMJ47Yz5QsmbWIov3D2EL6BOXJfR1wPqa3Yxyaf2M/Ul7TsvJB8qKLLmLTef5O5mk75wOyRvQwhkR62xQr0bl1KGhyJEMk7yQsBKYJb+MxA1/OCE+4oGC0GduDkyktewdB7ePshNNyLVbXDhIjklLVwmaSzlYJKGfb5St1BT) (passwd: nankai)  [[Baiduyun link]](https://pan.baidu.com/s/1yLszUZ71z0L_N4CZp679gQ) (passwd: 9t5b)



#### 数据格式转换

运行`scripts/convert_data2bin.py`脚本，来将数据集的验证集部分由图像格式文件转换为二进制文件

**参数说明：**

```shell
--dataset         # ucf101 或者 hmdb51
--data_path       # 指定欲转换数据集的存放位置
```

**示例（ucf101）：**

```shell
python3.7 scripts/convert_data2bin.py --dataset ucf101 --data_path ./data/ucf101_extracted
```

在执行完后，转换为bin的数据将会出现在  `bin/ucf101` 文件夹中，同时还会生成出包含label 信息的pkl文件（路径为`bin/ucf101_label.pkl`），该label信息可以方便之后的精度对比过程。



### 3. 执行离线推理获取精度

#### OM模型转换

参考[该Wiki文档](https://gitee.com/ascend/modelzoo/wikis/MobileNetV3_Large_MindStudio%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B?sort_id=3338402)进行om模型转换。**注意**:在进行om模型转换时，在MindStudio中键入input的shape不是常规的(1,224,224,3)而是(4,224,224,3)

转换好的**om模型**可根据下述链接下载：[[OBS link]](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=SSvMJBoCzDovvHhIh2gPGmIi3bKDm4FAmrNJyRXADHX4QqS41AiGMsoTK/hWaqLqALugAsgITdpMGELCPAYWii0ZyhyrNiIbc6Oxlqz0T81/RuzvB32/NqxyzjZ2tdqjdZpbYyoyhBXsqAzGRE0ftwAZItqpVB1/jjctdKOuuDuLlqycDrZDwgL7fWbHhR1h49FNvbeGwdUOii/5gKHk7GZMQxGzfz+vJw+4wyT7u7nO9Sf31V7rY8sDGk8wIgR+Awf1ZK5VGWWOVMFcN+lz0OQiFNA/JDiHjk8JrXvSsd+EVHnjPscUWVh0w2X5LDCzXXAATZuwhobfjtk2ykpkRs5TkZWWYPyuzuQNhthNHtfTtoEMq0d95Uac8yns5BTgBVu8jxj4cMgiw0c3yuq65ACnu36n5l35oPN0p3ZuED6IEKQ8Bsw9TdZjOP6kHws4Bias9mmbdTavOmbaAycMJJ9Rzo/Sg0D3pDtaL8k5Bd8IDklrJZbDJpVY9tt35xsL4sfo1V7aKVQNabgNkhA6bmwJ7cBlc1cPt5/sI1IRfuUO8wWPhf4cg00M996+FoZlGZ/VYKf9h9FxcLcmc9lHKNwWk0+C1Niq7q5qDtKOcR9N2DacRL/0MiQQKoO+IKQL) (passwd: nankai)  [[Baiduyun link]](https://pan.baidu.com/s/1Eq_OxYIrplTcnheDN2NODw) (passwd: 8u6b)



#### 使用OM模型进行离线推理
根据[msame主分支](https://gitee.com/ascend/tools/tree/master/msame)指导，配置msame工具
根据om文件及数据集bin文件路径，更改`scripts/offline_inference.sh`中的`MODEL`、`INPUT`和`OUTPUT`路径。

**然后运行：**

```shell
sh scripts/offline_inference.sh
```



#### 获取OM模型推理精度

运行`scripts/get_offline_acc.py`脚本，来获取OM模型推理精度。

**参数说明：**

```shell
--label_path       # 指定存储包含label信息的pkl文件路径
--output_path      # 指定经过om模型输出后的结果数据文件夹路径
```

**示例（ucf101）：**

```shell
python3.7 scripts/get_offline_acc.py --label_path econet/bin/ucf101_label.pkl --output_path econet/ucf101_results_bin/
```

运行ucf101/hmdb51推理模型后得到如下屏幕打印信息：

```
# ucf101 dataset
...
Totol image num: 3783, Top1 accuarcy: 0.8837
# hmdb51 dataset
...
Totol image num: 1530, Top1 accuarcy: 0.5980
```



## 模型性能

### 精度
|  数据集   |   ucf101验证集      |    hmdb51验证集  |
| :----------: | :-------------: | :------------: |
| 所要求精度  | 87.4% |   58.1%       |
| GPU训练精度  | 88.1% |   58.6%  |
| NPU训练精度 |  88.4% |   59.8%  |

### 训练速度
| Batchsize        |  1xTesla V100 | 1xAscend 910  |
| :--------------: | :--------: |:--------: |
| 4    |    49 videos/s     | 70 videos/s|
| 8 |    61 videos/s   | 90 videos/s |
| 16     |   69 videos/s     | 98 videos/s |


