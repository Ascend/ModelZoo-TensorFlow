

# ResNet101 Inference for Tensorflow 

This repository provides a script and recipe to Inference the ResNet101 model.

## Notice
**This sample only provides reference for you to learn the Ascend software stack and is not for commercial purposes.**

Before starting, please pay attention to the following adaptation conditions. If they do not match, may leading in failure.

| Conditions | Need |
| --- | --- |
| CANN Version | >=5.0.3 |
| Chip Platform| Ascend310/Ascend310P3 |
| 3rd Party Requirements| Please follow the 'requirements.txt' |

## Quick Start Guide

### 1. Clone the respository

```shell
git clone https://gitee.com/ascend/ModelZoo-TensorFlow.git
cd Modelzoo-TensorFlow/ACL_TensorFlow/built-in/cv/Resnet101_for_ACL
```

### 2. Download and preprocess the dataset

1. Download the ImageNet2012 dataset by yourself



### 3. Offline Inference

**Convert pb to om.**

  [pb download link](https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/modelzoo/Official/cv/Resnet101_for_ACL.zip)

- configure the env

  ```
  #Please modify the environment settings as needed
  export install_path=/usr/local/Ascend
  export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
  export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg:$PYTHONPATH
  export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
  export ASCEND_OPP_PATH=${install_path}/opp
  ```

- convert pb to om

  For Ascend310:
  ```
  atc --model=resnet101_tf.pb --framework=3 --output=resnet101_tf_aipp --output_type=FP32 --soc_version=Ascend310 --input_shape="input:1,224,224,3" --log=info --insert_op_conf=resnet101_tf_aipp.cfg
  ```
  For Ascend310P3:
  ```
  atc --model=resnet101_tf.pb --framework=3 --output=resnet101_tf_aipp --output_type=FP32 --soc_version=Ascend310P3 --input_shape="input:1,224,224,3" --log=info --insert_op_conf=resnet101_tf_aipp.cfg
  ```

- Build the program

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

- Run the program:

  ```
  cd scripts
  bash benchmark_tf.sh --batchSize=1 --modelType=resnet101 --imgType=raw --precision=fp16 --outputType=fp32 --useDvpp=1 --deviceId=0 --modelPath=resnet101_tf_aipp.om --dataPath=image-1024 --trueValuePath=val_lable.txt
  ```



## Performance

### Result

Our result were obtained by running the applicable inference script. To achieve the same results, follow the steps in the Quick Start Guide.

#### Inference accuracy results

|       model     |  SOC  | **data**  |    Top1/Top5    |
| :---------------:|:-------:|:-------: | :-------------: |
| offline Inference| Ascend310     | 50K images | 78.51 %/ 94.28% |
| offline Inference| Ascend310P3     | 50K images | 78.7 %/ 94.4% |
