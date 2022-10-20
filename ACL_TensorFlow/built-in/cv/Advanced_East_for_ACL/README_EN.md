English|[中文](README.md)

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
   python3 script/preprocess.py

   ```
   ```
   python3 script/image2bin.py

   ```


### 3. Offline Inference

- configure the env

  Please follow the [guide](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719) to set the envs
- Convert h5 to pb

  ```
  python3 h5_to_pb.py

  ```
2. Convert pb to om

   [pb download link]()

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
   python3 predict.py

  ```

#### Inference accuracy results

|       model       | **data**   |    precision    |    recall       |    heamn        |
| :---------------: | :-------:  | :-------------: | :-------------: | :-------------: |
| offline Inference | 1000 images |    84.91%       |    55.54%       |    63.57%       |
## 
