中文|[English](README_EN.md)

# ShuffleNetv2 TensorFlow离线推理

此链接提供ShuffleNetv2 TensorFlow模型在NPU上离线推理的脚本和方法

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
cd Modelzoo-TensorFlow/ACL_TensorFlow/built-in/cv/ShuffleNetv2_for_ACL
```

### 2. 下载数据集和预处理

1. 请自行下载ImageNet2012测试数据集，其中包括50000 JPEGS。

2. 移动 **ILSVRC2012val** 到 **'scripts/'**
```
———scripts
     |————ILSVRC2012val
           |————ILSVRC2012_val_00000001.JPEG
           |————ILSVRC2012_val_00000002.JPEG
           |————ILSVRC2012_val_00000003.JPEG
           |————ILSVRC2012_val_00000004.JPEG
           |————...
```

3. 图像预处理:
```
cd scripts
mkdir input_bins
python3 imagenet_preprocessing.py --src_path ./ILSVRC2012val/
```
jpeg图片将被预处理为bin fils.

### 3. 离线推理

**离线模型转换**

- 环境变量设置

  请参考[说明](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719)，设置环境变量

- Pb模型转换为om模型

  下载本报告训练的shuffenetv2的pb模型: [Repo of train](https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/shufflenetv2/shufflenetv2_ID0185_for_TensorFlow/Offline_shufflenetv2_tf_uestclzx)
  [pb模型下载链接](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/modelzoo/Official/cv/ShuffleNetv2_for_ACL.zip)

  ```
  atc --model=shufflenetv2.pb --framework=3 --output=shufflenetv2_tf_1batch --output_type=FP32 --soc_version=Ascend310 --input_shape="input:1,224,224,3" --out_nodes="classifier/BiasAdd:0" --log=info
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

## 推理结果

### 结果

本结果是通过运行上面适配的推理脚本获得的。要获得相同的结果，请按照《快速指南》中的步骤操作。

#### 推理精度结果


|       model       | **data**  |    Top1/Top5    |
| :---------------: | :-------: | :-------------: |
| offline Inference | 50000 images | 61.4 %/ 82.9% |


## 参考
https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/shufflenetv2/shufflenetv2_ID0185_for_TensorFlow

