

# DCN Inference for Tensorflow 

This repository provides a script and recipe to Inference the **Deep & Cross Network for Ad Click Predictions** model. Original train implement please follow this link: [DCN_for_Tensorflow](https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/built-in/recommendation/DCN_ID1986_for_TensorFlow)

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
cd Modelzoo-TensorFlow/ACL_TensorFlow/built-in/recommendation/DCN_for_ACL
```

### 2. Preprocess of the dataset

1. Download Criteo dataset by yourself, and move **Criteo/train.txt** to **scripts**.

2. Split dataset to train and test(0.8:0.2),Preprocess of the test datasets to bin files with **batchsize=4000**:
```
cd scripts
python3 data_preprocess.py Criteo/train.txt
```
and it will generate **input_x**, **labels** directories with batchsize **4000**:
```
input_x
|___batch1_X.bin
|___batch2_X.bin
...

labels
|___batch1_Y.bin
|___batch2_Y.bin
...
```

### 3. Offline Inference

**Convert pb to om.**

- configure the env

  ```
  export install_path=/usr/local/Ascend
  export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
  export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg:$PYTHONPATH
  export LD_LIBRARY_PATH=${install_path}/atc/lib64:$LD_LIBRARY_PATH
  export ASCEND_OPP_PATH=${install_path}/opp
  ```

- convert pb to om

  [pb download link](https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/modelzoo/Official/recommendation/DCN_for_ACL.zip)

  ```
  atc --model=dcn_tf.pb --framework=3 --output=dcn_tf_4000batch --output_type=FP32 --soc_version=Ascend310 --input_shape="input_1:4000,39" --input_format=ND --log=info
  ```

- Build the program

  ```
  bash build.sh
  ```
  An executable file **benchmark** will be generated under the path: **Benchmark/output/**

- Run the program:

  ```
  cd scripts
  bash benchmark_tf.sh
  ```



## Performance

### Result

Our result were obtained by running the applicable training script. To achieve the same results, follow the steps in the Quick Start Guide.

#### Inference accuracy results:

| Test Dataset | Accuracy-ROC |Accuracy-PR |
|--------------|-------------------|---------|
|  Criteo        | 80.5%             | 59.8% |

## Reference
[1] https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/built-in/recommendation/DCN_ID1986_for_TensorFlow
