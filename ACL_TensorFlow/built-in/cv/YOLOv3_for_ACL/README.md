

# Yolov3 Inference for Tensorflow 

This repository provides a script and recipe to Inference the Yolov3 model.

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
cd Modelzoo-TensorFlow/ACL_TensorFlow/built-in/cv/YOLOv3_for_ACL
```

### 2. Requirements

opencv-python==4.2.0.34


### 3. Download and preprocess the dataset

1. dataset
  To compare with official implement, for example, we use [get_coco_dataset.sh](https://github.com/pjreddie/darknet/blob/master/scripts/get_coco_dataset.sh) to prepare our dataset.

2. annotation file

   cd scripts

   Using script generate `coco2014_minival.txt` file. Modify the path in `coco_minival_anns.py` and `5k.txt`, then execute:

   ```
   python3 coco_minival_anns.py
   ```

   One line for one image, in the format like `image_index image_absolute_path img_width img_height box_1 box_2 ... box_n`.    
   Box_x format: 

   - `label_index x_min y_min x_max y_max`. (The origin of coordinates is at the left top corner, left top => (xmin, ymin), right bottom => (xmax, ymax).)    
   - `image_index` is the line index which starts from zero. `label_index` is in range [0, class_num - 1].

   For example:

   ```
   0 xxx/xxx/a.jpg 1920 1080 0 453 369 473 391 1 588 245 608 268
   1 xxx/xxx/b.jpg 1920 1080 1 466 403 485 422 2 793 300 809 320
   ...
   ```


### 3. Offline Inference

**Convert pb to om.**

- Configure the env according to your installation path 

  ```
  #Please modify the environment settings as needed
  export install_path=/usr/local/Ascend
  export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
  export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg:$PYTHONPATH
  export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
  export ASCEND_OPP_PATH=${install_path}/opp
  ```

- convert pb to om

  [pb download link](https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/modelzoo/yolov3_tf.pb)

  For Ascend310:
  ```
  atc --model=yolov3_tf.pb --framework=3 --output=yolov3_tf_aipp --output_type=FP32 --soc_version=Ascend310 --input_shape="input:1,416,416,3" --log=info --insert_op_conf=yolov3_tf_aipp.cfg
  ```
  For Ascend310P3:
  ```
  atc --model=yolov3_tf.pb --framework=3 --output=yolov3_tf_aipp --output_type=FP32 --soc_version=Ascend310P3 --input_shape="input:1,416,416,3" --log=info --insert_op_conf=yolov3_tf_aipp.cfg
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
  bash benchmark_tf.sh --batchSize=1 --modelType=yolov3 --imgType=raw --precision=fp16 --outputType=fp32 --useDvpp=1 --deviceId=0 --modelPath=yolov3_tf_aipp.om --trueValuePath=instance_val2014.json --imgInfoFile=coco2014_minival.txt --classNamePath=../../coco.names
  ```



## Performance

### Result

Our result were obtained by running the applicable inference script. To achieve the same results, follow the steps in the Quick Start Guide.

#### Inference accuracy results

IoU=0.5
| model  | Npu_nums | **mAP** | 
| :----: | :------: | :-----: | 
| Yolov3 |    1     |  55.3%   | 
