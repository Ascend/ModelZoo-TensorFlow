English|[中文](README.md)

# PWCNet Inference for Tensorflow 

This repository provides a script and recipe to Inference the PWCNet model.

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
cd Modelzoo-TensorFlow/ACL_TensorFlow/contrib/cv/PWCNet_for_ACL
```

### 2. Download and preprocess the dataset

1. Download the  MPI Sintel dataset by yourself and follow the [guide](https://github.com/philferriere/tfoptflow) to process the dataset then put it to the path: **scripts/dataset/MPI-Sintel-complete**

2. Preprocess of the test datasets and labels:
```
cd scripts
mkdir input_bins
python3 data_preprocess.py --dataset ./dataset--output ./input_bins
```
and it will generate **image** and **gt** directories under **input_bins**:
```
input_bins
|
|__image
   |______alley_1-frames_0001_0002.bin
   |______alley_1-frames_0002_0003.bin
   |______alley_1-frames_0003_0004.bin
...

|
|__gt
   |______alley_1-frames_0001_0002.bin
   |______alley_1-frames_0002_0003.bin
   |______alley_1-frames_0003_0004.bin
...

```

### 3. Offline Inference

**Convert pb to om.**

- configure the env

  Please follow the [guide](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719) to set the envs

- convert pb to om
  
  [**Pb Download Link**](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/006_train_backup/PWCNet_tf_wosaisai/offline_infer/pwcnet.pb)

  batchsize=1

  ```
  atc --model=pwcnet.pb  --framework=3 --input_shape="x_tnsr:1,2,448,1024,3" --output=./pwcnet_1batch --soc_version=Ascend310 --log=info
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

| Test Dataset | Number of pictures | EPE |
|--------------|-------------------|-------------------|
| MPI Sintel          | 1041             | 1.25             |

## Reference
[1] https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/pwcnet/PWCNet_ID0171_for_TensorFlow
