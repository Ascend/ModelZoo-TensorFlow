

# 3DResNet Inference for Tensorflow 

This repository provides a script and recipe to Inference of the 3DResNet model.

## Notice
**This sample only provides reference for you to learn the Ascend software stack and is not for commercial purposes.**

Before starting, please pay attention to the following adaptation conditions. If they do not match, may leading in failure.

| Conditions | Need |
| --- | --- |
| CANN Version | >=5.0.3 |
| Chip Platform| Ascend310/Ascend710 |
| 3rd Party Requirements| Please follow the 'requirements.txt' |

## Quick Start Guide

### 1. Clone the respository

```shell
git clone https://gitee.com/ascend/ModelZoo-TensorFlow.git
cd modelzoo/built-in/ACL_TensorFlow/Reseach/cv/3DResNet_for_ACL
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

  ```
  export install_path=/usr/local/Ascend
  export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
  export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg:$PYTHONPATH
  export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
  export ASCEND_OPP_PATH=${install_path}/opp
  ```

- convert pb to om

  [**pb download link**](https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/modelzoo/Research/cv/3DResNet_for_ACL.zip)

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
