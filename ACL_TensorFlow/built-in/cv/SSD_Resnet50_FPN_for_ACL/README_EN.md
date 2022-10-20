English|[中文](README.md)
# SSD-RESNET50FPN inference for Tensorflow

This repository provides a script and recipe to Inference the

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
cd Modelzoo-TensorFlow/ACL_TensorFlow/built-in/cv/SSD_Resnet50_FPN_for_ACL
```

### 2. Download and preprocess the dataset

1. Download the coco2014 dataset by yourself

2. Executing the Preprocessing Script
   ```
   python3 scripts/ssd_dataPrepare.py --input_file_path=Path of the image --output_file_path=Binary path for inference --crop_width=Width of the image cropping --crop_height=height of the image cropping --save_conf_path=Image configuration file path
   
   ```
3. Download gt labels
   [instances_minival2014.json](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com:443/010_Offline_Inference/Official/cv/ID1654_ssd_resnet50fpn/scripts/instances_minival2014.json?AccessKeyId=APWPYQJZOXDROK0SPPNG&Expires=1656057065&Signature=ydPmdux71bGzs38Q/xV7USQIdCg%3D)

   put json file to **'scripts'**
 
### 3. Offline Inference

**Convert pb to om.**

- configure the env

  Please follow the [guide](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719) to set the envs

- convert pb to om
  
  [pb download link](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/modelzoo/Official/cv/SSD_Resnet50_FPN_for_ACL.zip)

  ```
  atc --model=model/ssd-resnet50fpn_tf.pb --framework=3 --output=model/ssd_resnet50_fpn --output_type=FP16 --soc_version=Ascend310P3 --input_shape="image_tensor:1,640,640,3" "input_name1:image_tensor" --enable_scope_fusion_passes=ScopeBatchMultiClassNMSPass,ScopeDecodeBboxV2Pass,ScopeNormalizeBBoxPass,ScopeToAbsoluteBBoxPass
  ```

- Build the program

  ```
  bash build.sh
  ```

- Run the program:

  ```
  bash benchmark_tf.sh --batchSize=1 --modelPath=../../model/ssd_resnet50_fpn.om --dataPath=../../datasets/ --modelType=ssd_resnet50_fpn --imgType=rgb --trueValuePath=../../scripts/instances_minival2014.json
  ```
  
## Performance

### Result

Our result were obtained by running the applicable inference script. To achieve the same results, follow the steps in the Quick Start Guide.

#### Inference accuracy results

|       model       | ***data***  |    map      |
| :---------------: | :---------: | :---------: |
| offline Inference | 4952 images |   37.8%     |

