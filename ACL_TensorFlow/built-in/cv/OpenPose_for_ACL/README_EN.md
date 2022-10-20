English|[中文](README.md)
# <font face="微软雅黑">

# OpenPose Inference for TensorFlow
This repository provides a script and recipe to Inference the OpenPose model.

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
cd Modelzoo-TensorFlow/ACL_TensorFlow/built-in/cv/OpenPose_for_ACL
```

### 2. Download and preprocess the dataset

Download the COCO2014 dataset by yourself, more details see: [dataset](./dataset/coco/README.md)


### 3. Obtain the pb model

Obtain the OpenPose pb model, more details see: [models](./models/README.md)

### 4. Obtain process scripts

Obtain pafprocess and slidingwindow packages from: [tf_openpose](https://github.com/BoomFan/openpose-tf/tree/master/tf_pose) and put them into libs


### 5. Offline Inference
**Preprocess the dataset**
```Bash
python3 preprocess.py \
    --resize 656x368 \
    --model cmu \
    --coco-year 2014 \
    --coco-dir ../dataset/coco/ \
    --output-dir ../input/

```

**Convert pb to om.**
- configure the env

  Please follow the [guide](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719) to set the envs


- convert pb to om

  [pb download link](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/modelzoo/Official/cv/OpenPose_for_ACL.zip)

  ```
  atc --framework=3 \
      --model=./models/OpenPose_for_TensorFlow_BatchSize_1.pb \
      --output=./models/OpenPose_for_TensorFlow_BatchSize_1 \
      --soc_version=Ascend310 \
      --input_shape="image:1,368,656,3"
  ```

**Build the program**
Build the inference application, more details see: [xacl_fmk](./xacl_fmk/README.md)

**Run the inference**
```
/xacl_fmk -m ./models/OpenPose_for_TensorFlow_BatchSize_1.om \
    -o ./output/openpose \
    -i ./input \
    -b 1
```

**PostProcess**
```
python3 postprocess.py \
    --resize 656x368 \
    --resize-out-ratio 8.0 \
    --model cmu \
    --coco-year 2014 \
    --coco-dir ../dataset/coco/ \
    --data-idx 100 \
    --output-dir ../output/openpose 
```

**Sample scripts**
We also supoort the predict_openpose.sh to run the steps all above except **build the program**

### 6.Result
***
OpenPose Inference ：

| Type | IoU | Area | MaxDets | Result |
| :------- | :------- | :------- | :------- | :------- |
| Average Precision  (AP) | 0.50:0.95 | all | 20 | 0.399 |
| Average Precision  (AP) | 0.50 | all | 20 | 0.648 |
| Average Precision  (AP) | 0.75| all | 20 | 0.400 |
| Average Precision  (AP) | 0.50:0.95 | medium | 20 | 0.364 |
| Average Precision  (AP) | 0.50:0.95 | large | 20 | 0.443 |
| Average Recall     (AR) | 0.50:0.95 | all | 20 | 0.456 |
| Average Recall     (AR) | 0.50 | all | 20 | 0.683 |
| Average Recall     (AR) | 0.75 | all | 20 | 0.465 |
| Average Recall     (AR) | 0.50:0.95 | medium | 20 | 0.371 |
| Average Recall     (AR) | 0.50:0.95 | large | 20 | 0.547 |

***

## Reference

[1] https://github.com/ildoonet/tf-pose-estimation/tree/master/tf_pose/pafprocess/


# </font>
