# SSD_InceptionV2 Inference for Tensorflow

This repository provides a script and recipe to Inference of SSD_InceptionV2 model.

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
cd Modelzoo-TensorFlow/ACL_TensorFlow/contrib/cv/SSD_InceptionV2_for_ACL
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

  ```
  export install_path=/usr/local/Ascend
  export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
  export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg:$PYTHONPATH
  export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
  export ASCEND_OPP_PATH=${install_path}/opp
  ```

- convert pb to om
  
  [pb download link](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/modelzoo/Research/cv/SSD_for_ACL/ssd_inceptionv2_tf.pb)

  ```
  atc --model=model/ssd_inceptionv2_tf.pb --framework=3 --output=model/ssd_inceptionv2 --output_type=FP16 --soc_version=Ascend310P3 --input_shape="image_tensor:1,640,640,3" --out_nodes="detection_boxes:0;detection_scores:0;num_detections:0;detection_classes:0"
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
| offline Inference | 4952 images |   27.8%     |

## Reference

[1] https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md

