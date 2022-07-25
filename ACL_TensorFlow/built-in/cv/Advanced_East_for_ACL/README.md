# Advanced_East_for_ACL
This repository provides a script and recipe to Inference the Advanced_East model.

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
cd Modelzoo-TensorFlow/ACL_TensorFlow/built-in/cv/Advanced_East_for_ACL
```

### 2. Download and preprocess the dataset tianchi ICPR MTWI 2018

1. Download the dataset by yourself
   ```
    tianchi ICPR MTWI 2018
   
   ```

2. Executing the Preprocessing Script
  
   ```
   python3. preprocess.py

   ```
   ```
   python3.7.5 image2bin.py

   ```


### 3. Offline Inference
**Convert h5 to pb.**
  ```
  python3.7.5 h5_to_pb.py

  ```
**Convert pb to om.**

  [pb download link]()


- configure the env

  ```
  export install_path=/usr/local/Ascend
  export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
  export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg:$PYTHONPATH
  export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
  export ASCEND_OPP_PATH=${install_path}/opp
  ```

- convert pb to om

  ```
  atc --model=model.pb --input_shape="input_img:1,736,736,3" --framework=3 --output=Advanced_East --soc_version=Ascend310 --input_format=NHWC 
  
  ```

- Build the program 

  ```
  bash build.sh
  ```

- Run the program:

```
bash benchmark_tf.sh --batchSize=1 --modelPath=../../model/Advanced_East.om --dataPath=../../datasets/ --modelType=Advanced_East  --imgType=rgb 

```
## Performance

### Result

Our result were obtained by running the applicable inference script. To achieve the same results, follow the steps in the Quick Start Guide.
  ```
   python3.7.5 predict.py

  ```

#### Inference accuracy results

|       model       | **data**   |    precision    |    recall       |    heamn        |
| :---------------: | :-------:  | :-------------: | :-------------: | :-------------: |
| offline Inference | 233 images |    74.73%       |    70.26%       |    72.43%       |
## 
