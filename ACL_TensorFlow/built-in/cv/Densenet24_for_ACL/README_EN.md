English|[中文](README.md)

# Densenet24 inference for Tensorflow

This repository provides a script and recipe to Inference the Densenet24 model

## Notice
**This sample only provides reference for you to learn the Ascend software stack and is not for commercial purposes.**

Before starting, please pay attention to the following adaptation conditions. If they do not match, may leading in failure.

| Conditions | Need |
| --- | --- |
| CANN Version | >=5.0.3 |
| Chip Platform| Ascend310 |
| 3rd Party Requirements| Please follow the 'requirements.txt' |

## Quick Start Guide

### 1. Clone the respository

```shell
git clone https://gitee.com/ascend/ModelZoo-TensorFlow.git
cd Modelzoo-TensorFlow/ACL_TensorFlow/built-in/cv/Densenet24_for_ACL
```

### 2. Download and preprocess the dataset

1. Download the dataset by yourself
2. Put the dataset files to **'Densenet24_for_ACL/ori_images'** like this:
```
--ori_images
  |----BRATS2017
     |----Brats17ValidationData
       |----Brats17_2013_21_1
          |----xxxxx.nii.gz
       |----Brats17_2013_25_1
       |----Brats17_2013_26_1
       |----Brats17_CBICA_ABM_1
       |----Brats17_CBICA_AUR_1
       |----Brats17_CBICA_AXN_1
       |----Brats17_CBICA_AXQ_1
       |----Brats17_TCIA_192_1
       |----Brats17_TCIA_319_1
       |----Brats17_TCIA_377_1
       |----val.txt
       |----val.txt.zl
  |----npu
     |----d24_correction-4.index
     |----d24_correction-4.meta
```
3. Executing the Preprocessing Script
   ```
   cd scripts
   python3 preprocess.py -m ../ori_images/npu/dense24_correction-4 -mn dense24 -nc True -r ../ori_images/BRATS2017/Brats17ValidationData/ -input1 ../datasets/input_flair/ -input2 ../datasets/input_t1/
   ```
 
### 3. Offline Inference

**Convert pb to om.**

- configure the env

  Please follow the [guide](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719) to set the envs


- convert pb to om

  [pb download link](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/modelzoo/Official/cv/DenseNet24_for_ACL.zip)

  ```
  cd ..
  atc --model=model/densenet24.pb --framework=3 --output=model/densenet24_1batch --soc_version=Ascend310P3 --input_shape="Placeholder:1,38,38,38,2;Placeholder_1:1,38,38,38,2"
  ```

- Build the program

  ```
  bash build.sh
  ```

- Run the program:

  ```
  cd scripts
  bash benchmark_tf.sh 
  ```
  
## Performance

### Result

Our result were obtained by running the applicable inference script. To achieve the same results, follow the steps in the Quick Start Guide.

#### Inference accuracy results

|       model       |  **data**  |   TumorCore   |    PeritumoralEdema    |    EnhancingTumor   |
| :---------------: |  :------:  | :-----------: | :--------------------: | :-----------------: |
| offline Inference |  10 images |     99.588%   |         99.812%        |        99.901%      |

