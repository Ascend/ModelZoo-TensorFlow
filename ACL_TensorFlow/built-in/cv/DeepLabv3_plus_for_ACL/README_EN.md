English|[中文](README.md)

# DeepLabv3+ Inference for Tensorflow 

This repository provides a script and recipe to Inference of the Deeplabv3+ model.

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
cd Modelzoo-TensorFlow/ACL_TensorFlow/built-in/cv/DeepLabv3_plus_for_ACL
```

### 2. Download and preprocess the dataset

1. Download the PascalVoc2012 dataset by yourself. 

2. Put the dataset files to **'scripts/PascalVoc2012'** like this:
```
--PascalVoc2012
|----Annotations
|----ImageSets
|----JPEGImages
|----SegmentationClass
|----SegmentationObject
```

3. Images Preprocess:
```
cd scripts
mkdir input_bins
python3 preprocess/preprocessing.py ./PascalVoc2012/ ./input_bins/
```
The jpegs pictures will be preprocessed to bin fils.

### 3. Offline Inference

**Convert pb to om.**

  [pb download link](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/modelzoo/Official/cv/deepLabv3_plus_for_ACL.zip)

- configure the env

  Please follow the [guide](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719) to set the envs

- convert pb to om

  ```
  atc --model=deeplabv3_plus_tf.pb --framework=3 --output=deeplabv3_plus_tf_1batch --output_type=FP32 --soc_version=Ascend310 --input_shape="ImageTensor:1,513,513,3" --out_nodes=SemanticPredictions:0 --log=info
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

#### Inference accuracy results of Validation testset

|       model       | **data**  |    MeanIOU    |
| :---------------: | :-------: | :-------------: |
| offline Inference | 1449 images | 93.6% |

## Reference
[1] https://github.com/tensorflow/models/tree/master/research/deeplab
