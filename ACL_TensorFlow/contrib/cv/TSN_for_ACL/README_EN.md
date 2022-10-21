# <font face="微软雅黑">
English|[中文](README.md)

# TSN Inference for TensorFlow

***
This repository provides a script and recipe to Inference the TSN Inference

TSN Inference, based on [M-PACT](https://github.com/MichiganCOG/M-PACT)

***

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
cd modelzoo/tree/master/built-in/ACL_TensorFlow/Research/cv/TSN_for_ACL/inference
```

### 2. Download and preprocess the dataset

Download the dataset by yourself, more details see: [UCF101](./dataset/tfrecords_UCF101/readme)



### 3. Obtain the weights

Obtain the ckpt model, more details see: [weights](./inference/models/weights/readme) 


### 4. Build the program
Build the inference application, more details see: [xacl_fmk](./xacl_fmk/README.md)


### 5. Offline Inference

**Configure the env**
  Please follow the [guide](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719) to set the envs


**PreProcess**
```
bash run_dump.sh
```


**Convert pb to om**

[**pb download link**](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/modelzoo/Research/cv/TSN_for_ACL.zip)

```Bash
atc --model=./models/TSN_gpu.pb --framework=3 --output=./models/TSN_gpu --soc_version=Ascend310 --input_shape="Placeholder:1,250,224,224,3" --log=info
```



**Run the inference**
```Bash
python3 xacl_inference.py
```

**PostProcess**

```
bash run_infernece.sh
```


### 6. Performance

### Result
 
We measure performance by Accuracy.


|          |  Ascend310|
|----------|-----------|
| Accuracy |  0.9266   |

