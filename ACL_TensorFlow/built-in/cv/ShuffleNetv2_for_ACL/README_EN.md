English|[中文](README.md)

# ShuffleNetv2 Inference for Tensorflow 

This repository provides a script and recipe to Inference of the ShuffleNetv2 model.

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
cd Modelzoo-TensorFlow/ACL_TensorFlow/built-in/cv/ShuffleNetv2_for_ACL
```

### 2. Download and preprocess the dataset

1. Download the ImageNet2012 Validation dataset by yourself which includes 50000 JPEGS.

2. Move **ILSVRC2012val** to **'scripts/'**
```
———scripts
     |————ILSVRC2012val
           |————ILSVRC2012_val_00000001.JPEG
           |————ILSVRC2012_val_00000002.JPEG
           |————ILSVRC2012_val_00000003.JPEG
           |————ILSVRC2012_val_00000004.JPEG
           |————...
```

3. Images Preprocess:
```
cd scripts
mkdir input_bins
python3 imagenet_preprocessing.py --src_path ./ILSVRC2012val/
```
The jpeg pictures will be preprocessed to bin fils.

### 3. Offline Inference

**Convert pb to om.**

- configure the env

  Please follow the [guide](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719) to set the envs

- convert pb to om

  Download pb model of shufflenetv2 which was trained by this repo: [Repo of train](https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/shufflenetv2/shufflenetv2_ID0185_for_TensorFlow/Offline_shufflenetv2_tf_uestclzx)
  [Pb Download Link](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/modelzoo/Official/cv/ShuffleNetv2_for_ACL.zip)

  ```
  atc --model=shufflenetv2.pb --framework=3 --output=shufflenetv2_tf_1batch --output_type=FP32 --soc_version=Ascend310 --input_shape="input:1,224,224,3" --out_nodes="classifier/BiasAdd:0" --log=info
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
| offline Inference | 50000 images | 61.4 %/ 82.9% |


## Reference
https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/shufflenetv2/shufflenetv2_ID0185_for_TensorFlow

