English|[中文](README.md)

# 3DResNet Inference for Tensorflow 

This repository provides a script and recipe to Inference of the 3DResNet model.

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
cd Modelzoo-TensorFlow/ACL_TensorFlow/contrib/cv/3DResNet_for_ACL
```

### 2. Generate random test dataset

1. Because of this is not a well trained model we test the model with random test dataset

2. Generate random test dataset:
```
cd scripts
python3 generate_random_data.py
```
There will random testdata bin fils under *input_bins/*.

### 3. Offline Inference

**Convert pb to om.**

- configure the env

  Please follow the [guide](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719) to set the envs


- convert pb to om

  [**pb download link**](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/modelzoo/Research/cv/3DResNet_for_ACL.zip)

  ```
  atc --model=3DResNet_tf_gpu.pb --framework=3 --output=3DResNet_tf_gpu --soc_version=Ascend310 --input_shape="input_1:1,64,64,32,1" --log=info

  ```

- Build the program

  ```
  bash build.sh
  ```

- Run the program:

  ```
  cd scripts
  bash benchmark_tf.sh
  ```
  
- Run the post process:

  ```
  cd scripts
  python3 post_processing.py
  ```
  

## Reference
[1] https://github.com/JihongJu/keras-resnet3d
