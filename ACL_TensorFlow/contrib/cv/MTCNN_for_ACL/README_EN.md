English|[中文](README.md)

# MTCNN Inference for Tensorflow 

This repository provides a script and recipe to Inference the MTCNN model.

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
cd Modelzoo-TensorFlow/ACL/Official/cv/MTCNN_for_ACL
```


### 2. Offline Inference

**Convert pb to om and inference.**

  [pb download link](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/modelzoo/Research/cv/MTCNN_for_ACL.zip)

- configure the env

  Please follow the [guide](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719) to set the envs

- Build the program

  ```
  cd xacl_fmk-master
  bash xacl_fmk.sh
  ```

- Run the program:

  ```
  cd ..
  python3 acltest.py ompath data_in_om data_out_om Ascend310P3 ./mtc_pnet.pb ./mtc_rnet.pb ./mtc_onet.pb
  ```
  Notes: 
  By default， image in the "picture" directory are inferred. If you want to replace image, replace the image in the "picture" directory. 

  The ompath，data_in_om and data_out_om directorys don't need to be created. They will be automatically created.

