English|[中文](README.md)

# Resnet34 Inference for Tensorflow 

This repository provides a script and recipe to Inference of the Resnet34 model.

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
cd Modelzoo-TensorFlow/ACL_TensorFlow/built-in/cv/Resnet34_for_ACL
```

### 2. Download and preprocess the dataset

1. Download the Tiny-ImageNet-200 dataset by yourself.

2. Move **tiny-imagenet-200** to **'scripts/'**
```
———scripts
     |————tiny-imagenet-200
           |————test
           |————train
           |————val
           |————wnids.txt
           |————words.txt
```

3. Images Preprocess:
```
cd scripts
mkdir input_bins
python3 imagenet_tiny_preprocessing.py
```
The jpegs pictures will be preprocessed to bin fils.

### 3. Offline Inference

**Convert pb to om.**

- configure the env

  Please follow the [guide](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719) to set the envs

- convert pb to om

  [pb download link](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/modelzoo/resnet34_tf.pb)

  ```
  atc --model=resnet34_tf.pb --framework=3 --output=resnet34_tf_1batch --output_type=FP32 --soc_version=Ascend310 --input_shape="test_inputs:1,64,64,3" --log=info --insert_op_conf=resnet34_tf_aipp.cfg --enable_small_channel=1
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

## Performance

### Result

Our result was obtained by running the applicable inference script. To achieve the same results, follow the steps in the Quick Start Guide.

#### Inference accuracy results

|       model       | **data**  |    Top1/Top5    |
| :---------------: | :-------: | :-------------: |
| offline Inference | 10000 images | 51.9 %/ 76.6% |


## Reference
[1] https://github.com/taki0112/ResNet-Tensorflow
