English|[中文](README.md)

# Yolov5 Inference for Tensorflow 

This repository provides a script and recipe to Inference of the Yolov5 model.

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
cd Modelzoo-TensorFlow/ACL_TensorFlow/contrib/cv/Yolov5_for_ACL
```

### 2. Download and preprocess the dataset

1. Refer to this [url](https://github.com/hunglc007/tensorflow-yolov4-tflite/README.md) to download and preprocess the dataset
The operation is as follows:
```
# run script in /script/get_coco_dataset_2017.sh to download COCO 2017 Dataset
# preprocess coco dataset
cd data
mkdir dataset
cd ..
cd scripts
python coco_convert.py --input ./coco/annotations/instances_val2017.json --output val2017.pkl
python coco_annotation.py --coco_path ./coco 
python img2bin.py --img-dir ./coco/images --bin-dir ./coco/input_bins
mv coco ..
```
There will generate coco2017 test data set under *data/dataset/*.

### 3. Offline Inference

**Convert pb to om.**

- configure the env

  Please follow the [guide](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719) to set the envs

- convert pb to om

  ```
  atc --model=yolov5_tf2_gpu.pb --framework=3 --output=yolov5_tf2_gpu --soc_version=Ascend310 --input_shape="Input:1,640,640,3" --out_nodes="Identity:0;Identity_1:0;Identity_2:0;Identity_3:0;Identity_4:0;Identity_5:0" --log=info
  ```

- Build the program

  ```
  bash build.sh
  ```

- Run the program:

  ```
  cd offline_inference
  bash benchmark_tf.sh
  ```
  
- Run the post process:

  ```
  cd ..
  python3 offline_inference/postprocess.py
  ```
  
## Performance

### Result

Our result were obtained by running the applicable inference script. To achieve the same results, follow the steps in the Quick Start Guide.

#### Inference accuracy results

|       model       | **data**  |   AP/AR   |
| :---------------: | :-------: | :-----------: |
| offline Inference | 4952 images | 0.221/0.214 |
  

## Reference
[1] https://github.com/hunglc007/tensorflow-yolov4-tflite

[2] https://github.com/ultralytics/yolov5

[3]https://github.com/khoadinh44/YOLOv5_customized_data
