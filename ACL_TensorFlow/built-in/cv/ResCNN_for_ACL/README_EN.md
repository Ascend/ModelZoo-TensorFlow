English|[中文](README.md)

# ResCNN Inference for Tensorflow 

This repository provides a script and recipe to Inference of the ResCNN model.

## Notice
**This sample only provides reference for you to learn the Ascend software stack and is not for commercial purposes.**

Before starting, please pay attention to the following adaptation conditions. If they do not match, may leading in failure.

| Conditions | Need |
| --- | --- |
| CANN Version | >=5.0.4 |
| Chip Platform| Ascend310/Ascend310P3 |
| 3rd Party Requirements| Please follow the 'requirements.txt' |

## Quick Start Guide

### 1. Clone the respository

```shell
git clone https://gitee.com/ascend/ModelZoo-TensorFlow.git
cd Modelzoo-TensorFlow/ACL_TensorFlow/built-in/cv/ResCNN_for_ACL/
```

### 2. Download and preprocess the dataset

1. Download the DIV2K dataset by yourself. 

2. Put 100 LR pictures to './DIV2K_test_100/' as test data.

3. Make directories for inference input and output:
```
cd scripts
mkdir input_bins
mkdir results
```
   Temporary bin files will be saved.


### 3. Offline Inference

**Convert pb to om.**

  [pb download link](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/modelzoo/Official/cv/ResCNN_for_ACL.zip)

- configure the env

  Please follow the [guide](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719) to set the envs

- convert pb to om
  Because of the whole test picture will be split to some different sizes,including 64 x 64, 32 x 64, 32 x 44, etc, here,we convert three om files:

  ```
  atc --model=ResCNN_tf.pb --framework=3 --output=ResCNN_64_64_1batch --output_type=FP32 --soc_version=Ascend310 --input_shape="input:1,64,64,3" --log=info
  atc --model=ResCNN_tf.pb --framework=3 --output=ResCNN_32_64_1batch --output_type=FP32 --soc_version=Ascend310 --input_shape="input:1,32,64,3" --log=info
  atc --model=ResCNN_tf.pb --framework=3 --output=ResCNN_32_44_1batch --output_type=FP32 --soc_version=Ascend310 --input_shape="input:1,32,44,3" --log=info
  ```

- Build the program

  ```
  bash build.sh
  ```

- Run the program:

  ```
  bash benchmark_tf.sh
  ```

## NPU Performance
### Result

Our result was obtained by running the applicable inference script. To achieve the same results, follow the steps in the Quick Start Guide.

#### Inference accuracy results

|       model       | data  |    mean-PSNR    | mean-SSIM|
| :---------------: | :-------: | :-------------: |:-------------:|
| offline Inference | 100 images | 23.748 |0.747|


## Reference
[1] https://github.com/payne911/SR-ResCNN-Keras

