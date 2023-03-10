English|[中文](README.md)

# FACENET inference for Tensorflow

This repository provides a script and recipe to Inference the FACENET model.

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
cd Modelzoo-TensorFlow/ACL_TensorFlow/built-in/cv/Facenet_for_ACL
```

### 2. Download and preprocess the dataset

1. Download the lfw dataset by yourself

2. Executing the Preprocessing Script
   ```
   python3 align/align_dataset_mtcnn.py $cur_dir/lfw $dataset --image_size 160 --margin 32 --random_order
   python3 preprocess_data.py  Path_of_Data_after_face_alignment  Outpath_of_Data_after_face_alignment  --use_fixed_image_standardization --lfw_batch_size 1 --use_flipped_images
   
   ```
 
### 3. Offline Inference

**Convert pb to om.**

  [pb(20180402) download link](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/modelzoo/Official/cv/Facenet_for_ACL.zip)  
  [pb(20180408) download link](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/2022-09-24_tf/FaceNet_for_ACL/facenet_20180408-102900.pb)

- configure the env

  Please follow the [guide](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719) to set the envs

- convert pb to om
  
  Sample of pb(20180402):

  ```
   atc --framework=3 --model=./model/facenet_tf.pb  --output=./model/facenet --soc_version=Ascend310 --insert_op_conf=./facenet_tensorflow.cfg --input_format=NHWC --input_shape=input:4,160,160,3
  ```

- Build the program

  ```
  bash build.sh
  ```

- Run the program:

  ```
  ./benchmark_tf.sh --batchSize=4 --modelPath=absolute_path_of_facenet.om --dataPath=absolute_path_of_dataset_bin --modelType=facenet --imgType=rgb
  ```
  
## Performance

### Result

Our result were obtained by running the applicable inference script. To achieve the same results, follow the steps in the Quick Start Guide.

#### Inference accuracy results

|       model    |       mode       | ***data***  |    Embeddings Accuracy    |
| :---------------:| :---------------: | :---------: | :---------: |
| pb(20180402)| offline Inference | 12000 images |   99.532%     |
| pb(20180408)| offline Inference | 12000 images |   98.917%     |

