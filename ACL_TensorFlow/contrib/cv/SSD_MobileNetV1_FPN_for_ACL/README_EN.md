English|[中文](README.md)
# SSD_MobileNetV1_FPN Inference for Tensorflow

This repository provides a script and recipe to Inference of SSD_MobileNetV1_FPN model.

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
cd Modelzoo-TensorFlow/ACL_TensorFlow/contrib/cv/SSD_MobileNetV1_FPN_for_ACL
```

### 2. Download and preprocess the dataset

1. Download the coco2014 test dataset by yourself

2. Executing the Preprocessing Script
   ```
   cd scripts
   mkdir input_bins
   python3 scripts/ssd_dataPrepare.py --input_file_path=Path of the image --output_file_path=./input_bins --crop_width=640 --crop_height=640 --save_conf_path=./img_info
   
   ```
3. Download gt labels
   [instances_minival2014.json](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com:443/010_Offline_Inference/Official/cv/ID1654_ssd_resnet50fpn/scripts/instances_minival2014.json?AccessKeyId=APWPYQJZOXDROK0SPPNG&Expires=1656057065&Signature=ydPmdux71bGzs38Q/xV7USQIdCg%3D)

   put json file to **'scripts'**
 
### 3. Offline Inference

**Convert pb to om.**

- configure the env

  Please follow the [guide](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719) to set the envs

- convert pb to om
  
  [pb download link](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/modelzoo/Research/cv/SSD_for_ACL/ssd_mobilenetv1_fpn_tf.pb)

  ```
  atc --model=model/ssd_mobilenetv1_fpn_tf.pb --framework=3 --output=model/ssd_mobilenetv1_fpn --output_type=FP16 --soc_version=Ascend310P3 --input_shape="image_tensor:1,640,640,3" --out_nodes="detection_boxes:0;detection_scores:0;num_detections:0;detection_classes:0"
  ```

- Build the program

  ```
  bash build.sh
  ```

- Run the program:

  ```
  bash benchmark.sh
  ```
  
## Performance

### Result

Our result were obtained by running the applicable inference script. To achieve the same results, follow the steps in the Quick Start Guide.

#### Inference accuracy results

|       model       | ***data***  |    map      |
| :---------------: | :---------: | :---------: |
| offline Inference | 4952 images |   34.8%     |

## Reference

[1] https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md

