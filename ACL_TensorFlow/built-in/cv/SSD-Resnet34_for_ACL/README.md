中文|[English](README_EN.md)

# SSD Resnet34 TensorFlow离线推理

此链接提供SSD Resnet34 TensorFlow模型在NPU上离线推理的脚本和方法

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
cd Modelzoo-TensorFlow/ACL_TensorFlow/built-in/cv/SSD-Resnet34_for_ACL
```

### 2. 下载数据集和预处理

1. 请自行下载COCO2017测试数据集

 

### 3. 离线推理

**离线模型转换**

- 环境变量设置

  请参考[说明](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719)，设置环境变量

- Pb模型转换为om模型

  [pb模型下载链接](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/modelzoo/Official/cv/SSD_Resnet34_for_ACL.zip)

  For Ascend310:
  ```
  atc --model=ssdresnet34_1batch_tf.pb --framework=3 --output=ssdresnet34_1batch_tf_aipp --output_type=FP32 --soc_version=Ascend310 --input_shape="input:1,300,300,3" --log=info --insert_op_conf=ssdresnet34_tf_aipp.cfg
  ```
  For Ascend310P3:
  ```
  atc --model=ssdresnet34_1batch_tf.pb --framework=3 --output=ssdresnet34_1batch_tf_aipp --output_type=FP32 --soc_version=Ascend310P3 --input_shape="input:1,300,300,3" --log=info --insert_op_conf=ssdresnet34_tf_aipp.cfg
  ```

- 编译程序

  For Ascend310:
  ```
  unset ASCEND310P3_DVPP
  bash build.sh
  ```
  For Ascend310P3:
  ```
  export ASCEND310P3_DVPP=1
  bash build.sh
  ```

- 开始运行:

  ```
  cd scripts
  bash benchmark_tf.sh --batchSize=1 --modelType=ssdresnet34 --imgType=raw --precision=fp16 --outputType=fp32 --useDvpp=1 --deviceId=0 --modelPath=ssdresnet34_1batch_tf_aipp.om --dataPath=COCO2017/val2017 --trueValuePath=instances_val2017.json
  ```



## 推理结果

### 结果

本结果是通过运行上面适配的推理脚本获得的。要获得相同的结果，请按照《快速指南》中的步骤操作。

#### 推理精度结果

|       model     |  SOC  | **data**  |    mAP    |
| :---------------:|:-------:|:-------: | :-------------: |
| offline Inference| Ascend310     | 5K images | 24.6% |
| offline Inference| Ascend310P3     | 5K images | 24.9% |

