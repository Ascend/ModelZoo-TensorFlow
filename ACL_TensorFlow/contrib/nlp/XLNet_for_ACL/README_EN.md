English|[中文](README.md)

# XLNet Inference for Tensorflow 

This repository provides a script and recipe to Inference the XLNet model.

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
cd Modelzoo-TensorFlow/ACL_TensorFlow/contrib/nlp/XLNet_for_ACL
```

### 2. Download and preprocess the dataset

1. Follow the guide of [**train repo**](https://github.com/zihangdai/xlnet), download the **glue_data** dataset by yourself then move them to the path: **scripts/**
```
scripts
|
|___glue_data
     |
     |___STS-B
         |
         |___LICENSE.txt
         |___dev.tsv
         |___original
         |___readme.txt
         |___test.tsv
         |___train.tsv
```
Then convert STS-B dataset to tfrecord to **scripts/proc_data/** like this:
```
proc_data
|
|___sts-b
    |__spiece.model.len-128.dev.eval.tf_record
    |__spiece.model.len-128.test.predict.tf_record
    |__spiece.model.len-128.train.eval.tf_record
```
Here we use dev-eval tfrecord for demostration.

2. Convert tfrecord to bin files
```
cd scripts
python3 data_preprocess.py
```

### 3. Offline Inference

**Convert pb to om.**

- configure the env

  Please follow the [guide](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719) to set the envs

- convert pb to om
  
  [**Pb Download Link**](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/modelzoo/Research/nlp/XLNET_tf.pb)

  batchsize=1

  ```
  atc --model=XLNet_tf.pb  --framework=3 --input_shape="input_ids_1:1,128;input_mask_1:1,128;segment_ids_1:1,128" --output=./XLNet_1batch --soc_version=Ascend310 --log=info
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

#### Inference results:

|       model       |  dataset   |     Number of samples     |   eval_pearsonr   |
|-------------------|---- |--------------|---------|
| offline Inference |  sts-b   |1500 samples  | 91.85%  |

## Reference
[1] https://github.com/zihangdai/xlnet
