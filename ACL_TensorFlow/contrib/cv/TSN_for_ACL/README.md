# <font face="微软雅黑">

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
```
ASCEND_HOME=/usr/local/Ascend
PYTHON_HOME=/usr/local/python3.7
export PATH=$PATH:$PYTHON_HOME/bin:$ASCEND_HOME/atc/ccec_compiler/bin:$ASCEND_HOME/atc/bin:$ASCEND_HOME/toolkit/bin/
export LD_LIBRARY_PATH=$ASCEND_HOME/acllib/lib64:$ASCEND_HOME/toolkit/lib64:$ASCEND_HOME/add-ons:$ASCEND_HOME/opp/op_proto/built-in/:$ASCEND_HOME/opp/framework/built-in/tensorflow/:$ASCEND_HOME/opp/op_impl/built-in/ai_core/tbe/op_tiling
export PYTHONPATH=$ASCEND_HOME/atc/python/site-packages/auto_tune.egg:$ASCEND_HOME/atc/python/site-packages/schedule_search.egg:/caffe/python/:$ASCEND_HOME/ops/op_impl/built-in/ai_core/tbe/
export ASCEND_OPP_PATH=$ASCEND_HOME/opp
export SOC_VERSION=Ascend310
# HOST_TYPE in Ascend310 support Atlas300 and MiniRC
export HOST_TYPE=Atlas300
```

**PreProcess**
```
bash run_dump.sh
```


**Convert pb to om**

[**pb download link**](https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/modelzoo/Research/cv/TSN_for_ACL.zip)

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

