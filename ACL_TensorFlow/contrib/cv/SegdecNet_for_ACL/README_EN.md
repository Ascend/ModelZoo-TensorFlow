English|[中文](README.md)

# SegdecNet Inference for Tensorflow 

This repository provides a script and recipe to Inference of the SegdecNet model.

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
cd Modelzoo-TensorFlow/ACL_TensorFlow/contrib/cv/SegdecNet_for_ACL
```

### 2. Download and preprocess the dataset

1. Download the KolektorSDD Validation dataset by yourself. 

2. You should split the dataset into three folds to perform 3-fold cross validation.[split](./scripts/db/README.md)

3. Images Preprocess:
```
cd scripts
bash run_preprocess.sh
```
The images bin files is stored in output/images/
The labels bin files is stored in output/labels/
 

### 3. Offline Inference

**Convert pb to om.**

- configure the env

  Please follow the [guide](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719) to set the envs

- convert pb to om

  [**pb download link**](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/modelzoo/Research/cv/SEGDEC-NET_for_ACL.zip)

  ```
  atc --model=./output/SEGDEC-NET_tf.pb --framework=3 --output=./output/SEGDEC-NET_tf --output_type=FP32 --soc_version=Ascend310 --input_shape="images:1,1408,512,1" --log=info
  ```

- Build the program

  ```
  cd ../
  bash build.sh
  ```

- Run the program:

  ```
  cd scripts
  bash benchmark_tf.sh
  ```

- Postprocess:

  ```
  bash run_postprocess.sh
  ```

## Performance

### Result

Our result was obtained by running the applicable inference script. To achieve the same results, follow the steps in the Quick Start Guide.

#### Inference accuracy results

|       model       | AP of CPU | AP of NPU |
| :---------------: | :-------: | :-------: |
|     SegdecNet     |   0.9536  |   0.9528  |


## Reference
[1] https://github.com/skokec/segdec-net-jim2019
