# CenterNet Inference for Tensorflow
This respository provides a scripts and recipe to Inference of the CenterNet model

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
   cd Modelzoo-TensorFlow/ACL_TensorFlow/contrib/cv/CenterNet_for_ACL
   ```

### 2. Download and preprocess the dataset


1.  Download the VOC2007 test dataset by yourself,and then extract VOCtest_06-NOV-2007.tar.

2.  Move VOC2007 test dataset to 'scripts/VOC2007' like this:

    ```
    VOC2007
    |----Annotations
    |----ImageSets
    |----JPEGImages
    |----SegmentationClass
    |----SegmentationObject

    ```

3.  Image Preprocess
    
    ```
    cd scripts
    mkdir input_bins
    python3 preprocess.py ./input_bins/
    python3 xml2txt.py ./VOC2007/Annotations/ ./centernet_postprocess/groundtruths/

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

[**pb download link**](https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/modelzoo/Research/cv/CenterNet_for_ACL.zip)

```
atc --model=./CenterNet.pb --framework=3 --output=./Centernet_2batch_input_fp16_output_fp32 --soc_version=Ascend310 --input_shape="input_1:2,512,512,3"
```
3.Build the program
```  
bash build.sh
```
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
|       model       |     data     |   mAP   |
|-------------------|--------------|---------|
| offline Inference | 4952 images  | 74.90%  |


## Reference
[1]https://github.com/xuannianz/keras-CenterNet
