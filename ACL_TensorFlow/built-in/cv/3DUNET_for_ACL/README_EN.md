English|[中文](README.md)

# 3DUNET inference for Tensorflow

This repository provides a script and recipe to Inference the 3DUNET model

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
cd Modelzoo-TensorFlow/ACL_TensorFlow/contrib/cv/3DUNET_for_Tensorflow
```

### 2. Download and preprocess the dataset

1. Download the dataset by yourself

2. Put the dataset files to **3DUNET_for_ACL/ori_images/** like this:
```
--MICCAI_BraTS_2019_Data_Training

```

3. Executing the Preprocessing Script
   ```
   mkdir ori_images/tfrecord
   python3 scripts/preprocess_data.py --input_dir=ori_images/MICCAI_BraTS_2019_Data_Training/ --output_dir=ori_images/tfrecord
   python3 scripts/prepocess.py ./ori_images/tfrecord/ ./datasets ./labels
   ```
 
### 3. Offline Inference

**Convert pb to om.**

- configure the env

  Please follow the [guide](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719) to set the envs

[pb download link](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/2022-12-12_tf/3DUNET_TF_for_ACL/unet3d.pb)
- convert pb to om(Ascend310P3)  
  ```
  atc --model=model/unet3d.pb --framework=3 --output=model/unet3d_1batch --soc_version=Ascend310P3 --input_shape=input:1,224,224,160,4 --enable_small_channel=1
  ```

- convert pb to om(Ascend310)
  ```
  atc --model=model/unet3d.pb --framework=3 --output=model/unet3d_1batch --soc_version=Ascend310 --input_shape=input:1,224,224,160,4 --optypelist_for_implmode=ReduceMeanD --op_select_implmode=high_precision
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

|       model       | **data**   |       TumorCore     | PeritumoralEdema | EnhancingTumor | MeanDice | WholeTumor |   
| :---------------: | :-------:  | :-----------------: |  :-------------: | :------------: |:--------:|:----------:|
| offline Inference |  68 images |        72.59%       |      78.48%      |     70.46%     |  73.84%  |   90.74%   |

