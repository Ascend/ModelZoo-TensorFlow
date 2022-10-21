# <font face="微软雅黑">
中文|[English](README_EN.md)

# VAEGAN TensorFlow离线推理

***
此链接提供Vgg16 TensorFlow模型在NPU上离线推理的脚本和方法

VAEGAN 推理，基于 [VAE/GAN](https://github.com/zhangqianhui/vae-gan-tensorflow)

***

## 注意
**此案例仅为您学习Ascend软件栈提供参考，不用于商业目的。**

在开始之前，请注意以下适配条件。如果不匹配，可能导致运行失败。

| Conditions | Need |
| --- | --- |
| CANN版本 | >=5.0.3 |
| 芯片平台| Ascend310/Ascend310P3 |
| 第三方依赖| 请参考 'requirements.txt' |

## 快速指南

### 1. 拷贝代码
```shell
git clone https://gitee.com/ascend/ModelZoo-TensorFlow.git
cd Modelzoo-TensorFlow/ACL_TensorFlow/contrib/cv/VAEGAN_for_ACL
```

### 2. 下载数据集和预处理

请自行下载数据集, 更多详细信息请参见: [CelebA](./Data/img_align_celeba/README.md)



### 3. 获取ckpt模型

获取ckpt模型, 更多详细信息请参见: [ckpt](./model/README.md)


### 4. 编译程序
构建推理应用程序, 更多详细信息请参见: [xacl_fmk](./xacl_fmk/README.md)

### 5. 离线推理



**环境变量设置**
  请参考[说明](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719)，设置环境变量

**预处理**
```Bash
python3 main.py --act dump
```


**Pb模型转换为om模型**

[**pb模型下载链接**](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/modelzoo/Research/cv/VAE_GAN_for_ACL.zip)

```Bash
atc --model=./model/VAE_GAN_gpu.pb --framework=3 --output=./model/VAE_GAN_gpu --soc_version=Ascend310 --input_shape="Placeholder:64,64,64,3" --log=info
```



**运行推理**
```Bash
python3 inference/xacl_inference.py
```

**后期处理**

```Bash
python3 main.py --act compare
```


### 6. 演示

### 结果
 
我们通过PSNR来衡量性能。


|                 | ascend310 |
|----------------|--------|
| PSNR |  11.922089  |

