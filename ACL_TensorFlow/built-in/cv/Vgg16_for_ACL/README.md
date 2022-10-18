中文|[English](README_EN.md)

# Vgg16 TensorFlow离线推理

此链接提供Vgg16 TensorFlow模型在NPU上离线推理的脚本和方法

## 注意
**此案例仅为您学习Ascend软件堆栈提供参考，不用于商业目的。**

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
cd Modelzoo-TensorFlow/ACL_TensorFlow/built-in/cv/Vgg16_for_ACL
```

### 2. 下载数据集和预处理

1. 请自行下载ImageNet2012测试数据集

   

### 3. 离线推理

**离线模型转换**

- 环境变量设置

  ```
  #Please modify the environment settings as needed
  export install_path=/usr/local/Ascend
  export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
  export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg:$PYTHONPATH
  export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
  export ASCEND_OPP_PATH=${install_path}/opp
  ```

- Pb模型转换为om模型

  [pb模型下载链接](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/modelzoo/Official/cv/Vgg16_for_ACL.zip)

  For Ascend310:
  ```
  atc --model=vgg16_tf.pb --framework=3 --output=vgg16_tf_aipp --output_type=FP32 --soc_version=Ascend310 --input_shape="input:1,224,224,3" --log=info --insert_op_conf=vgg16_tf_aipp.cfg --enable_small_channel=1 --enable_compress_weight=true
  ```
  For Ascend310P3:
  ```
  atc --model=vgg16_tf.pb --framework=3 --output=vgg16_tf_aipp --output_type=FP32 --soc_version=Ascend310P3 --input_shape="input:1,224,224,3" --log=info --insert_op_conf=vgg16_tf_aipp.cfg --enable_small_channel=1 --enable_compress_weight=true
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
  bash benchmark_tf.sh --batchSize=1 --modelType=vgg16 --imgType=raw --precision=fp16 --outputType=fp32 --useDvpp=1 --deviceId=0 --modelPath=vgg16_tf_aipp.om --dataPath=image-1024 --trueValuePath=val_lable.txt
  ```



## 推理结果

### 结果

本结果是通过运行上面适配的推理脚本获得的。要获得相同的结果，请按照《快速入门指南》中的步骤操作。

#### 推理精度结果

|       model     |  SOC  | **data**  |    Top1/Top5    |
| :---------------:|:-------:|:-------: | :-------------: |
| offline Inference| Ascend310     | 50K images | 72.82 %/ 91.24% |
| offline Inference| Ascend310P3     | 50K images | 73.4 %/ 91.7% |

## 参考
[1] https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/built-in/cv/image_classification/VGG16_ID0068_for_TensorFlow
