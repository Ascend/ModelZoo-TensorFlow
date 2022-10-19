English|[中文](README.md)

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

1. Follow the [train repo](https://github.com/mouna99/dien), download dataset and unzip it to use:
```
tar -jxvf data.tar.gz
mv data/* scripts
tar -jxvf data1.tar.gz
mv data1/* scripts
tar -jxvf data2.tar.gz
mv data2/* scripts
``` 
and you will get files below:
- cat_voc.pkl 
- mid_voc.pkl 
- uid_voc.pkl 
- local_train_splitByUser 
- local_test_splitByUser 
- reviews-info
- item-info

2. Generate test dataset:
```
cd scripts
python3 generate_data.py --batchsize=128
```
There will testdata bin fils under *input_bins/*.

### 3. Offline Inference

**Convert pb to om.**

- configure the env

  Please follow the [guide](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719) to set the envs

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

|       model       | **data**  |     MeanAccuracy   |
| :---------------: | :-------: | :-------------: |
| offline Inference | Amazon ProductGraph | 78.5% |

## Reference
[1] https://github.com/mouna99/dien
