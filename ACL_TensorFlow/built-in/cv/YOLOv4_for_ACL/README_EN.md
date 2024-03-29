English|[中文](README.md)

# YOLOv4 Inference for Tensorflow 

This repository provides a script and recipe to Inference of the YOLOv4 model.

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
cd Modelzoo-TensorFlow/ACL_TensorFlow/built-in/cv/YOLOv4_for_ACL
```

### 2. Download and preprocess the dataset

1. Download the COCO-2017 validation dataset by yourself. 

2. Put pictures to **'scripts/val2017'**

3. Images Preprocess:
```
cd scripts
mkdir input_bins
python3 preprocess.py ./val2017/ ./input_bins/
```
   The pictures will be preprocessed to bin files.

4. Split ground-truth labels from **instances_val2017.json**
```
python3 load_coco_json.py
```

### 3. Offline Inference

**Convert pb to om.**

- configure the env

  Please follow the [guide](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719) to set the envs

- convert pb to om

  [pb download link](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/modelzoo/yolov4_tf.pb)

  ```
  atc --model=yolov4_tf.pb --framework=3 --output=yolov4_tf_1batch --output_type=FP32 --soc_version=Ascend310 --input_shape="x:1,416,416,3" --log=info
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

|       model       | **data**  |    mAP    |
| :---------------: | :-------: | :-------------: |
| offline Inference | 5000 images | 60.7% |


## Reference
[1] https://github.com/hunglc007/tensorflow-yolov4-tflite

[2] https://github.com/rafaelpadilla/Object-Detection-Metrics
