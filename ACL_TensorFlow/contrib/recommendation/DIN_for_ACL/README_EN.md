English|[中文](README.md)

# DIN Inference for Tensorflow 

This repository provides a script and recipe to Inference of the DIN model.

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
cd Modelzoo-TensorFlow/ACL_TensorFlow/contrib/recommendation/DIN_for_ACL
```

### 2. Generate random test dataset

1. Follow the [train repo](https://github.com/zhougr1993/DeepInterestNetwork) guide download amazon data
```
step 1:
cd ../raw_data
wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz
gzip -d reviews_Electronics_5.json.gz
wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Electronics.json.gz
gzip -d meta_Electronics.json.gz
step 2:
cd utils
python 1_convert_pd.py;
python 2_remap_id.py
step 3:
cd din
build_dataset.py
```
get the dataset: dataset.pkl

2. Generate test dataset:
```
cd scripts
python3 preprocess.py 
```
There will generate testdata bin fils under *input_bins/* and dataset_conf.txt.

### 3. Offline Inference

**Convert pb to om.**

- configure the env

  Please follow the [guide](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719) to set the envs

- convert pb to om

  [**pb download link**](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/2022-09-24_tf/DIN_for_ACL/frozen_din.pb)

  ```
  export batch_size=512
  atc --model=frozen_din.pb --framework=3 --soc_version=Ascend310P3 --output=din_${batch_size}batch_dynamic_shape --log=error --op_debug_level=3 --input_shape_range="Placeholder_1:[100~512];Placeholder_2:[100~512];Placeholder_4:[100~512,-1];Placeholder_5:[100~512]"
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

|       model       |  **data**   |   test_gauc   |   test_auc   |
| :---------------: |  :-------:  | :-----------: | :-----------: |
| offline Inference | dataset.pkl |     0.6854    |     0.6836    |

## Reference
[1] https://github.com/AustinMaster/DeepInterestNetwork/tree/master/din
