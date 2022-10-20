English|[中文](README.md)

# PixelLink Inference for Tensorflow 

This repository provides a script and recipe to Inference of the PixelLink model.

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
cd Modelzoo-TensorFlow/ACL_TensorFlow/built-in/cv/PixelLink_for_ACL
```

### 2. Download and preprocess the dataset

1. Download the Icdar2015 test dataset by yourself. You can get the test pictures(500 JPEGS)

2. Put JPEGS to **'scripts/ch4_test_images'**

3. Images Preprocess:
```
cd scripts
mkdir input_bins
python3 pixellink_preprocessing.py ./ch4_test_images/ ./input_bins/
```
The jpegs pictures will be preprocessed to bin fils.

### 3. Offline Inference

**Convert pb to om.**

  [pb download link](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/modelzoo/Official/cv/PixelLink_for_ACL.zip)

- configure the env

  Please follow the [guide](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719) to set the envs

- convert pb to om

  ```
  atc --model=pixellink_tf.pb --framework=3 --output=pixellink_tf_1batch --output_type=FP32 --soc_version=Ascend310 --input_shape="input:1,768,1280,3" --insert_op_conf=pixellink_tf_aipp.json --log=info
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

|       model       | **data**  |    Hmean    |
| :---------------: | :-------: | :-------------: |
| offline Inference | 500 images | 82.4% |

