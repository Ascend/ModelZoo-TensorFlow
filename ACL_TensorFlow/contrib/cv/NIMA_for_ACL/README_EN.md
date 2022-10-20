# NIMA Inference for Tensorflow
This respository provides a scripts and recipe to Inference of the NIMA model

## Notice
**This sample only provides reference for you to learn the Ascend software stack and is not for commercial purposes.**

Before starting, please pay attention to the following adaptation conditions. If they do not match, may leading in failure.

| Conditions | Need |
| --- | --- |
| CANN Version | >=5.0.3 |
| Chip Platform| Ascend310/Ascend310P3 |
| 3rd Party Requirements| Please follow the 'requirements.txt' |

## Quick Start Guide

### 1.Clone the respository

   ```
   git clone https://gitee.com/ascend/ModelZoo-TensorFlow.git
   cd Modelzoo-TensorFlow/ACL_TensorFlow/contrib/cv/NIMA_for_ACL
   ```

### 2. Download and preprocess the dataset


1.  Download the AVA test dataset by yourself and it should contains 5000 pictures and a AVA.txt.

2.  Move AVA test dataset to 'scripts/AVA_DATASET_TEST' like this:

    ```
    AVA_DATASET_TEST
    |
    |__image
    |   |____12315.jpg
    |   |____12316.jpg
    |   .....
    |__AVA.txt

    ```

3.  Image Preprocess
    
    ```
    cd scripts
    mkdir input_bins
    python3 data_preprocess.py AVA_DATASET_TEST ./input_bins/

    ```    
The pictures will be preprocessed to input_bins files. The lables will be preprocessed to predict_txt files. 

### 3.Offline Inference
 
1.configure the env

```
export install_path_atc=/usr/local/Ascend
export ASCEND_OPP_PATH=${install_path_atc}/opp
export PATH=/usr/local/python3.7.5/bin:${install_path_atc}/atc/ccec_compiler/bin:${install_path_atc}/atc/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin
export PYTHONPATH=${install_path_atc}/atc/python/site-packages/te:${install_path_atc}/atc/python/site-packages/topi:${install_path_atc}/atc/python/site-packages/auto_tune.egg:${install_path_atc}/atc/python/site-packages/schedule_search.egg
export LD_LIBRARY_PATH=${install_path_atc}/acllib/lib64:${install_path_atc}/atc/lib64:${install_path_atc}/toolkit/lib64:${install_path_atc}/add-ons:$LD_LIBRARY_PATH
```
   
2.convert pb to om

[**pb download link**](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/modelzoo/Research/cv/NIMA_for_ACL.zip)

```
atc --model=./nima.pb --framework=3 --output=./nima_1batch_input_fp16_output_fp32 --soc_version=Ascend310 --input_shape="input_1:1,224,224,3" --soc_version=Ascend310
```
3.Build the program
```  
bash build.sh
```
An executable file **benchmark** will be generated under the path: **Benchmark/output/**

4.Run the program
```  
cd scripts
bash benchmark_tf.sh
```

## Performance

### Result

Our results was obtained by running the applicable inference script.

#### Inference accuracy results
--------------------------
|       Dataset       |     Numbers     |   SSRC   |
|-------------------|--------------|---------|
| AVA test dataset | 5000 images  | 51.78%  |


## Reference
[1] https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/NIMA/NIMA_ID0853_for_TensorFlow
