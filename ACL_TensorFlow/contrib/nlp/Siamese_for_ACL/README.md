

# Siamese Inference for Tensorflow 

This repository provides a script and recipe to Inference the Siamese model. Original train implement please follow this link: [Siamese_for_Tensorflow](https://github.com/dhwajraj/deep-siamese-text-similarity)
and in this repo we trained a model for **Phrase Similarity**.

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
cd Modelzoo-TensorFlow/ACL_TensorFlow/contrib/cv/Siamese_for_ACL
```

### 2. Preprocess of the dataset

1. When the train steps finished, **validation.txt0** and **vocab** under **runs/xxxx/checkpoints/** will be generated. Copy them to **scripts/dataset** path.

2. Preprocess of the validation datasets:
```
cd scripts
python3 data_preprocess.py
```
and it will generate **input_x1**, **input_x2**, **ground_truth** directories with batchsize **128**:
```
input_x1
|___000000.bin
|___000001.bin
...

input_x2
|___000000.bin
|___000001.bin
...

ground_truth
|___000000.txt
|___000001.txt
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

  ```
  atc --model=siamese_tf.pb --framework=3 --output=siamese_tf_128batch --output_type=FP32 --soc_version=Ascend310 --input_shape="input_x1:128,15;input_x2:128,15" --log=info --precision_mode=allow_fp32_to_fp16
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

| Test Dataset | Accuracy |
|--------------|-------------------|
|  vocab        | 94.9%             |

## Reference
[1] https://github.com/dhwajraj/deep-siamese-text-similarity
