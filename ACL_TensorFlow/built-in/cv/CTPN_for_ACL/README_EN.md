English|[中文](README.md)

# CTPN inference for Tensorflow

This repository provides a script and recipe to Inference the CTPN

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
cd Modelzoo-TensorFlow/ACL_TensorFlow/built-in/cv/CTPN_for_ACL
```

### 2. Download and preprocess the dataset

1. Download the dataset by yourself
   ```
   https://github.com/eragonruan/text-detection-ctpn
   ```

2. Executing the Preprocessing Script
   ```
   python3 scripts/ctpn_dataPrepare.py --image_path=../image --output_path=$datasets --crop_width=1072 --crop_height=608 --img_conf=./img_info
   
   ```
3. Installing the BBOX
   ```
   cd scripts/utils/bbox
   bash make.sh
   ```
4. Download icdar2013 test dataset groundtruth files by yourself and put it in the test directory
   
### 3. Offline Inference

**Convert pb to om.**

  [pb download link](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/modelzoo/Official/cv/CTPN_for_ACL.zip)

- configure the env

  Please follow the [guide](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719) to set the envs


- convert pb to om

  ```
  atc --model=model/ctpn_tf.pb --framework=3 --output=model/ctpn_model --input_shape=input_image:1,608,1072,3 --soc_version=Ascend310P3 --enable_scope_fusion_passes=ScopeDynamicRNNPass --enable_small_channel=1
  ```

- Build the program 

  ```
  bash build.sh
  ```

- Run the program:

  ```
  bash benchmark_tf.sh --batchSize=1 --modelPath=../../model/ctpn_model.om --dataPath=../../datasets/ --modelType=ctpn  --imgType=rgb 
  ```
  
## Performance

### Result

Our result were obtained by running the applicable inference script. To achieve the same results, follow the steps in the Quick Start Guide.

#### Inference accuracy results

|       model       | **data**   |    precision    |    recall       |    heamn        |
| :---------------: | :-------:  | :-------------: | :-------------: | :-------------: |
| offline Inference | 233 images |    74.73%       |    70.26%       |    72.43%       |

