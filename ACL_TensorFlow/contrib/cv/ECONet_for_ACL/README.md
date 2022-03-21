

# ECONet Inference for Tensorflow 

This repository provides a script and recipe to Inference the ECONet model.

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
cd modelzoo/built-in/ACL_TensorFlow/Research/cv/ECONet_for_ACL
```

### 2. Download and preprocess the dataset

Here we use a model trained by UCF101 dataset and you can also use a model trained by hmdb51 dataset.

1. Download the  UCF101 dataset by yourself and put it to the path: **scripts/dataset/ucf101**

2. Preprocess of the test datasets and labels:
```
cd scripts
mkdir input_bins
python3 data_preprocess.py --dataset ucf101 --data_path ./dataset/ucf101 --output_path ./input_bins
```
and it will generate **ucf101** and **ucf101_label.pkl** directories under **input_bins**:
```
input_bins
|
|__ucf101
|______v_ApplyEyeMakeup_g01_c01.bin
|______v_ApplyEyeMakeup_g01_c02.bin
|______v_ApplyEyeMakeup_g01_c03.bin
...

|
|__ucf101_label.pkl

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

  [**Pb Download Link**](https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/006_train_backup/econet/ECONet_tf_paper99/scripts/ucf101_best.pb)
  
  batchsize=4

  ```
  atc --model=ucf101_best.pb  --framework=3 --input_shape="clip_holder:4,224,224,3" --output=./econet_ucf101_4batch --soc_version=Ascend310 --log=info
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

| Test Dataset | Number of pictures | Top1/Top5 |
|--------------|-------------------|-------------------|
| ucf101          | 3783             | 88.4%/98.2%             |

## Reference
[1] https://gitee.com/ascend/modelzoo/tree/master/contrib/TensorFlow/Research/cv/econet/ECONet_tf_paper99
