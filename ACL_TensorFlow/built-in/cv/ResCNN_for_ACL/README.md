中文|[English](README_EN.md)

# ResCNN TensorFlow离线推理

此链接提供ResCNN TensorFlow模型在NPU上离线推理的脚本和方法

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
cd Modelzoo-TensorFlow/ACL_TensorFlow/built-in/cv/ResCNN_for_ACL/
```

### 2. 下载数据集和预处理

1. 请自行下载DIV2K测试数据集

2. 将100张LR图片放到'./DIV2K_test_100/'目录下作为测试数据.

3. 创建推理输入和输出目录:
```
cd scripts
mkdir input_bins
mkdir results
```
   将保存临时bin文件


### 3. 离线推理

**Pb模型转换为om模型**

  [pb模型下载链接](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/modelzoo/Official/cv/ResCNN_for_ACL.zip)

- 环境变量设置

  请参考[说明](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719)，设置环境变量

- Pb模型转换为om模型
  由于整个测试图片将被分割为一些不同的大小，包括64 x 64、32 x 64、32x 44，这里我们转换了三个om文件：

  ```
  atc --model=ResCNN_tf.pb --framework=3 --output=ResCNN_64_64_1batch --output_type=FP32 --soc_version=Ascend310 --input_shape="input:1,64,64,3" --log=info
  atc --model=ResCNN_tf.pb --framework=3 --output=ResCNN_32_64_1batch --output_type=FP32 --soc_version=Ascend310 --input_shape="input:1,32,64,3" --log=info
  atc --model=ResCNN_tf.pb --framework=3 --output=ResCNN_32_44_1batch --output_type=FP32 --soc_version=Ascend310 --input_shape="input:1,32,44,3" --log=info
  ```

- 编译程序

  ```
  bash build.sh
  ```

- 开始运行:

  ```
  bash benchmark_tf.sh
  ```

## NPU 性能
### 结果

本结果是通过运行上面适配的推理脚本获得的。要获得相同的结果，请按照《快速指南》中的步骤操作。

#### 推理精度结果
|       model       | data  |    mean-PSNR    | mean-SSIM|
| :---------------: | :-------: | :-------------: |:-------------:|
| offline Inference | 100 images | 23.748 |0.747|


## 参考
[1] https://github.com/payne911/SR-ResCNN-Keras

