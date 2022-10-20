中文|[English](README_EN.md)

# DeepLabv3+ Tensorflow离线推理

此链接提供DeepLabv3+ TensorFlow模型在NPU上离线推理的脚本和方法

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
cd Modelzoo-TensorFlow/ACL_TensorFlow/built-in/cv/DeepLabv3_plus_for_ACL
```

### 2. 下载数据集和预处理

1. 请自行下载PascalVoc2012数据集

2. 将数据集文件放入 **scripts/PascalVoc2012** 中:
```
--PascalVoc2012
|----Annotations
|----ImageSets
|----JPEGImages
|----SegmentationClass
|----SegmentationObject
```

3. 图片预处理:
```
cd scripts
mkdir input_bins
python3 preprocess/preprocessing.py ./PascalVoc2012/ ./input_bins/
```
jpegs图片将被预处理为bin文件。

### 3. 离线推理

**离线模型转换**

  [pb模型下载链接](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/modelzoo/Official/cv/deepLabv3_plus_for_ACL.zip)

- 环境变量设置

  请参考[说明](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719)，设置环境变量


- Pb模型转换为om模型

  ```
  atc --model=deeplabv3_plus_tf.pb --framework=3 --output=deeplabv3_plus_tf_1batch --output_type=FP32 --soc_version=Ascend310 --input_shape="ImageTensor:1,513,513,3" --out_nodes=SemanticPredictions:0 --log=info
  ```

- 编译程序

  ```
  bash build.sh
  ```

- 开始运行:

  ```
  cd scripts
  bash benchmark_tf.sh
  ```

## 性能

### 结果

本结果是通过运行上面适配的推理脚本获得的。要获得相同的结果，请按照《快速指南》中的步骤操作。

#### 验证集推理精度结果

|       model       | **data**  |    MeanIOU    |
| :---------------: | :-------: | :-------------: |
| offline Inference | 1449 images | 93.6% |

## 参考
[1] https://github.com/tensorflow/models/tree/master/research/deeplab
