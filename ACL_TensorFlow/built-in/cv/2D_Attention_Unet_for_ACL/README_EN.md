English|[中文](README.md)

# 2D_Attention_Unet inference for Tensorflow

This repository provides a script and recipe to Inference the 2D_Attention_Unet model.

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
cd Modelzoo-TensorFlow/ACL_TensorFlow/built-in/cv/2D_Attention_Unet_for_ACL
```

### 2. Download and preprocess the dataset

1. Download the Massachusetts Roads dataset by yourself


2. Put the dataset files to **'2D_Attention_Unet_for_ACL/image_ori/'** like this:
```
--image_ori
  |----lashan
    |----test(xxx.tiff,total:49images)
    |----test_labels
    |----val
    |----val_labels
  |----Val(xxx_gt/xxx_img/xxx_pred .png)
```


3. Executing the Preprocessing Script
   ```
   cd scripts
   python3 preprocessdata_test.py --dataset=../image_ori/lashan --crop_height=224 --crop_width=224
   ```
 
### 3. Offline Inference

**Convert pb to om.**

- configure the env

  Please follow the [guide](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719) to set the envs

- convert pb to om

  [pb download link](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/modelzoo/Official/cv/2D_Attention_Unet_for_ACL.zip)

  For Ascend310:
  ```
  atc --model=2D_Attention_Unet_tf.pb --framework=3 --output=model/2DAttention_fp16_1batch --soc_version=Ascend310 --input_shape=inputs:1,224,224,3 --enable_small_channel=1 --insert_op_conf=2DAttention_aipp.cfg
  ```
  For Ascend310P3:
  ```
  atc --model=2D_Attention_Unet_tf.pb --framework=3 --output=model/2DAttention_fp16_1batch --soc_version=Ascend310P3 --input_shape=inputs:1,224,224,3 --enable_small_channel=1 --insert_op_conf=2DAttention_aipp.cfg
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
  bash benchmark_tf.sh 
  ```
  
## Performance

### Result

Our result were obtained by running the applicable inference script. To achieve the same results, follow the steps in the Quick Start Guide.

#### Inference accuracy results

|       model       | **data**   |       accuracy      |    Road      |    Others    |    precision    |    F1_score    |    Iou    |
| :---------------: | :-------:  | :-----------------: | :----------: | :----------: | :-------------: | :------------: | :-------: |
| offline Inference |  49 images |     97.19%          |    60.25%    |    99.36%    |     97.88%      |      97.44%    |    76.02% |
