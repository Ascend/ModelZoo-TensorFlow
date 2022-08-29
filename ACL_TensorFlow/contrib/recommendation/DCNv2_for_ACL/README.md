

# DCNv2 Inference for Tensorflow 

This repository provides a script and recipe to Inference of the DCNv2 model.

## Notice
**This sample only provides reference for you to learn the Ascend software stack and is not for commercial purposes.**

Before starting, please pay attention to the following adaptation conditions. If they do not match, may leading in failure.

| Conditions | Need |
| --- | --- |
| CANN Version | >=5.0.3 |
| Chip Platform| Ascend310P3 |
| 3rd Party Requirements| Please follow the 'requirements.txt' |

## Quick Start Guide

### 1. Clone the respository

```shell
git clone https://gitee.com/ascend/ModelZoo-TensorFlow.git
cd Modelzoo-TensorFlow/ACL_TensorFlow/contrib/recommendation/DCNv2_for_ACL
```

### 2. Generate random test dataset

1. Because of this is not a well trained model we test the model with random test dataset

2. Generate random test dataset:
```
cd scripts
mkdir input_bins
python3 generate_random_data.py --path=./input_bins/ --nums=32
```
There will random testdata bin fils under *input_bins/*.

### 3. Offline Inference

**Convert pb to om.**

- configure the env

  ```
  export install_path=/usr/local/Ascend
  export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
  export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg:$PYTHONPATH
  export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64:$LD_LIBRARY_PATH
  export ASCEND_OPP_PATH=${install_path}/opp
  ```

- convert pb to om

  [**pb download link**](https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/modelzoo/Research/recommendation/DCNv2_for_ACL/DCN_v2.pb)

  ```
   atc --model=DCN_v2.pb --framework=3 --soc_version=Ascend310P3 --output=DCNv2_16384_batch --log=error --op_debug_level=3 --input_shape="Input_21:16384;Input_24:16384;Input_4:16384;Input_6:16384;Input_12:16384;Input_19:16384;Input_3:16384;Input_22:16384;Input_23:16384;Input_13:16384;Input_18:16384;Input_10:16384;Input_1:16384;Input_14:16384;Input_26:16384;Input_8:16384;Input_2:16384;Input_15:16384;Input_16:16384;Input_17:16384;Input_20:16384;Input_7:16384;Input:16384,8;Input_11:16384;Input_5:16384;Input_25:16384;Input_9:16384" --out_nodes=Identity:0
  ```

- Build the program

  ```
  bash build.sh
  ```

- Run the program:

  ```
  cd scripts
  bash benchmark_tf.sh
  ```

## Performance

### Result

Our result was obtained by running the applicable inference script. To achieve the same results, follow the steps in the Quick Start Guide.

#### Inference accuracy results

|       model       | **data**  |     Mean CosineSimilarity   |
| :---------------: | :-------: | :-------------: |
| offline Inference | random data | 100.0% |
