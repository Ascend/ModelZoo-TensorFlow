English|[中文](README.md)

# DeepSort Inference for Tensorflow 

This repository provides a script and recipe to Inference the DeepSort model.

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
cd Modelzoo-TensorFlow/ACL_TensorFlow/contrib/cv/DeepSort_for_ACL
```

### 2. Download and preprocess the dataset

1. Download the MOT16 dataset then put **'test/MOT16-03'** to the path: **scripts/dataset/test/**

2. We just use one test dataset(**MOT16-03**) for demo:
```
dataset/test
|
|__MOT16-03
   |______det
   |______img1
   |______seqinfo.ini

```

### 3. Offline Inference

**Convert pb to om.**

- configure the env

  Please follow the [guide](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719) to set the envs


- convert pb to om
  
  [**Pb Download Link**](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/modelzoo/Research/cv/DeepSort_for_ACL/mars-small128.pb)

  Dynamic batchsize

  ```
  atc --model=mars-small128.pb  --framework=3 --input_shape_range="images:[-1,128,64,3]" --output=./deepsort_dynamic_batch --soc_version=Ascend310 --log=info
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

[Demo Video](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/modelzoo/Research/cv/DeepSort_for_ACL/MOT16-03.avi)

## Reference
[1] https://github.com/nwojke/deep_sort
