

# DeepSort Inference for Tensorflow 

This repository provides a script and recipe to Inference the DeepSort model.

## Notice
**This sample only provides reference for you to learn the Ascend software stack and is not for commercial purposes.**

Before starting, please pay attention to the following adaptation conditions. If they do not match, may leading in failure.

| Conditions | Need |
| --- | --- |
| CANN Version | >=5.0.3 |
| Chip Platform| Ascend310/Ascend710 |
| 3rd Party Requirements| Please follow the 'requirements.txt' |

## Quick Start Guide

### 1. Clone the respository

```shell
git clone https://gitee.com/ascend/modelzoo.git
cd modelzoo/built-in/ACL_TensorFlow/Research/cv/DeepSort_for_ACL
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

  ```
  export install_path=/usr/local/Ascend
  export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
  export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg:$PYTHONPATH
  export LD_LIBRARY_PATH=${install_path}/atc/lib64:$LD_LIBRARY_PATH
  export ASCEND_OPP_PATH=${install_path}/opp
  ```

- convert pb to om
  
  [**Pb Download Link**](https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/modelzoo/Research/cv/DeepSort_for_ACL/mars-small128.pb)

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

[Demo Video](https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/modelzoo/Research/cv/DeepSort_for_ACL/MOT16-03.avi)

## Reference
[1] https://github.com/nwojke/deep_sort
