English|[中文](README.md)

# PSPNet101 inference for Tensorflow

This repository provides a script and recipe to Inference the PSPNet101 model.

## Quick Start Guide

### 1. Clone the respository

```shell
git clone https://gitee.com/ascend/ModelZoo-TensorFlow.git
cd Modelzoo-TensorFlow/ACL_TensorFlow/built-in/cv/PSPNet101_for_ACL
```

### 2. Download and preprocess the dataset

1. Download the dataset by yourself
2. Executing the Preprocessing Script
   ```
   #without flip
   python3 scripts/data_processing.py --img_num=500 --crop_width=720 --crop_height=720 --data_dir=../cityscapes --val_list=../cityscapes/list/cityscapes_val_list.txt --output_path=$dataset
   ```

   ```
   #flip
   python3 scripts/data_processing.py --img_num=500 --crop_width=720 --crop_height=720 --data_dir=../cityscapes --val_list=../cityscapes/list/cityscapes_val_list.txt --output_path=$dataset --flipped_eval --flipped_output_path=$flipped_dataset   
   ```

 
### 3. Offline Inference

**Convert pb to om.**

  [pb download link](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/modelzoo/Official/cv/PSPnet101_for_ACL.zip)

- configure the env

  Please follow the [guide](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719) to set the envs

- convert pb to om

  ```
  atc --model=model/PSPNet101.pb --framework=3 --output=model/pspnet101_1batch --soc_version=Ascend310P3 --input_shape=input_image:1,1024,2048,3 --enable_small_channel=1 --insert_op_conf=pspnet_aipp.cfg
  ```

- Build the program

  ```
  bash build.sh
  ```

- Run the program:

  ```
  without flip
  bash benchmark_tf.sh --batchSize=1 --outputType=fp32 --modelPath=../../model/pspnet101_1batch.om --dataPath=../../datasets/ --modelType=PSPnet101 --imgType=rgb
  ```

  ```
  flip
  bash benchmark_tf.sh --batchSize=1 --outputType=fp32 --modelPath=../../model/pspnet101_1batch.om --dataPath=../../datasets/ --modelType=PSPnet101 --imgType=rgb --flippedDataPath=../../flipped_datasets/ --flippedEval=1
  ```
  
## Performance

### Result

Our result were obtained by running the applicable inference script. To achieve the same results, follow the steps in the Quick Start Guide.

#### Without flip  Inference accuracy results

|       model       | **data**   |    mIoU    | 
| :---------------: | :-------:  | :--------: | 
| offline Inference | 500 images |    77%     | 


### flip  Inference accuracy results

|       model       | **data**   |    mIoU    |    
| :---------------: | :-------:  | :--------: | 
| offline Inference | 500 images |   77.24%   | 

