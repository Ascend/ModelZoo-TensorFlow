# <font face="微软雅黑">

# VAEGAN Inference for TensorFlow

***
This repository provides a script and recipe to Inference the VAEGAN Inference

VAEGAN Inference, based on [VAE/GAN](https://github.com/zhangqianhui/vae-gan-tensorflow)

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
cd Modelzoo-TensorFlow/ACL_TensorFlow/contrib/cv/VAEGAN_for_ACL
```

### 2. Download and preprocess the dataset

Download the dataset by yourself, more details see: [CelebA](./Data/img_align_celeba/README.md)



### 3. Obtain the ckpt model

Obtain the ckpt model, more details see: [ckpt](./model/README.md)


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
```Bash
python3 main.py --act dump
```


**Convert pb to om**

[**pb download link**](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/modelzoo/Research/cv/VAE_GAN_for_ACL.zip)

```Bash
atc --model=./model/VAE_GAN_gpu.pb --framework=3 --output=./model/VAE_GAN_gpu --soc_version=Ascend310 --input_shape="Placeholder:64,64,64,3" --log=info
```



**Run the inference**
```Bash
python3 inference/xacl_inference.py
```

**PostProcess**

```Bash
python3 main.py --act compare
```


### 6. Performance

### Result
 
We measure performance by  PSNR.


|                 | ascend310 |
|----------------|--------|
| PSNR |  11.922089  |

