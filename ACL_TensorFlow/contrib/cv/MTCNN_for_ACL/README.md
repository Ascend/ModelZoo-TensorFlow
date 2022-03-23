

# MTCNN Inference for Tensorflow 

This repository provides a script and recipe to Inference the MTCNN model.

## Notice
**This sample only provides reference for you to learn the Ascend software stack and is not for commercial purposes.**

Before starting, please pay attention to the following adaptation conditions. If they do not match, may leading in failure.

| Conditions | Need |
| --- | --- |
| CANN Version | >=5.0.3 |
| Chip Platform| Ascend310/Ascend710 |
| 3rd Party Requirements| Please follow the 'requirements.txt' |

## Quick Start Guide

### 1. Clone the respository

```shell
git clone https://gitee.com/ascend/ModelZoo-TensorFlow.git
cd Modelzoo-TensorFlow/ACL/Official/cv/MTCNN_for_ACL
```


### 2. Offline Inference

**Convert pb to om and inference.**

  [pb download link](https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/modelzoo/Research/cv/MTCNN_for_ACL.zip)

- configure the env

  ```
  export install_path=/usr/local/Ascend
  export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
  export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg:$PYTHONPATH
  export LD_LIBRARY_PATH=${install_path}/atc/lib64:$LD_LIBRARY_PATH
  export ASCEND_OPP_PATH=${install_path}/opp
  ```

- Build the program

  ```
  cd xacl_fmk-master
  bash xacl_fmk.sh
  ```

- Run the program:

  ```
  cd ..
  python3 acltest.py ompath data_in_om data_out_om Ascend710 ./mtc_pnet.pb ./mtc_rnet.pb ./mtc_onet.pb
  ```
  Notes: 
  By default， image in the "picture" directory are inferred. If you want to replace image, replace the image in the "picture" directory. 

  The ompath，data_in_om and data_out_om directorys don't need to be created. They will be automatically created.

