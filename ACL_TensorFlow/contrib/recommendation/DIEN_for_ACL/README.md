

# DIEN Inference for Tensorflow 

This repository provides a script and recipe to Inference of the DIEN model.

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
cd Modelzoo-TensorFlow/ACL_TensorFlow/contrib/recommendation/DIEN_for_ACL
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

  [**pb download link**](https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/modelzoo/Research/recommendation/DIEN_for_ACL/DIEN.pb)

  ```
  export batch_size=128
  atc --model=./DIEN.pb --framework=3 --output=./DIEN_${batch_size}batch --soc_version=Ascend310P3 --input_shape="Inputs/mid_his_batch_ph:${batch_size},100;Inputs/cat_his_batch_ph:${batch_size},100;Inputs/uid_batch_ph:${batch_size};Inputs/mid_batch_ph:${batch_size};Inputs/cat_batch_ph:${batch_size};Inputs/mask:${batch_size},100;Inputs/seq_len_ph:${batch_size}" --out_nodes="final_output:0" --precision_mode="allow_fp32_to_fp16" --customize_dtypes="./customize_dtypes.txt"
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

