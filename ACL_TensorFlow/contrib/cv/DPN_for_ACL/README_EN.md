English|[中文](README.md)

# DPN Inference for Tensorflow 

This repository provides a script and recipe to Inference the DPN model.

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
cd Modelzoo-TensorFlow/ACL_TensorFlow/contrib/cv/DPN_for_ACL
```

### 2. Download and preprocess the dataset

1. Download the  test dataset by yourself ([Download](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/007_inference_backup/dpn/dpn_tf_hw34064571/offline_inference/dataset/dpnval.tfrecords))and put it to the path: **scripts/dataset/**

2. Preprocess of the test datasets and labels:
```
cd scripts
mkdir input_bins
python3 dpn_preprocess.py dataset/dpnval.tfrecords ./input_bins/
```
and it will generate **data** , **distance** and **label** directories:
```
data
|___000000.bin
|___000001.bin
...

distance
|___000000.bin
|___000001.bin
...

label
|__label.npy
```

### 3. Offline Inference

**Convert pb to om.**

- configure the env

  Please follow the [guide](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719) to set the envs


- convert pb to om
  
   [**Pb Download Link**](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/007_inference_backup/dpn/dpn_tf_hw34064571/offline_inference/ckpt/dpn.pb)

  batchsize=8

  ```
  atc --model=dpn.pb  --framework=3 --input_shape="inputx:8,512,512,3,inputd:10,10,1" --output=./dpn_8batch --out_nodes="upsample/Conv_2/Relu:0" --soc_version=Ascend310 --log=info
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

| Test Dataset | Number of pictures | MeanIou |
|--------------|-------------------|-------------------|
| cvcdb          | 1448             | 46%             |

## Reference
[1] https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/dpn/DPN_ID1636_for_TensorFlow
