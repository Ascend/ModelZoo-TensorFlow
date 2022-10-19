[English|中文(README_CN.md)

# FACE-RESNET50 inference for Tensorflow

This repository provides a script and recipe to Inference the face-resnet50 model.

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
cd Modelzoo-TensorFlow/ACL_TensorFlow/built-in/cv/Face_Resnet50_for_ACL
```

### 2. Download and preprocess the dataset

1. Download the lfw dataset by yourself

2. Executing the Preprocessing Script
   ```
   python3 align/align_dataset_mtcnn_facereset.py $cur_dir/lfw $dataset
   python3 preprocess.py $cur_dir/config/basemodel.py Path_of_Data_after_processing
   
   ```
 
### 3. Offline Inference

**Convert pb to om.**

- get base model
  ```
  https://github.com/seasonSH/DocFace
  
  ```
- ckpt to pb
  ```
  for example:
   python3.7 /usr/local/python3.7.5/lib/python3.7/site-packages/tensorflow_core/python/tools/freeze_graph.py --input_checkpoint=./faceres_ms/ckpt-320000 --output_graph=./model/face_resnet50_tf.pb --output_node_names="embeddings" --input_meta_graph=./faceres_ms/graph.meta --input_binary=true
  ```

- configure the env

  Please follow the [guide](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719) to set the envs

- convert pb to om

  [pb download link](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/modelzoo/Official/cv/Face_Resnet50_for_ACL.zip)

  ```
  /usr/local/Ascend/atc/bin/atc --model ./model/face_resnet50_tf.pb   --framework=3  --output=face_resnet50 --input_shape="image_batch:1,112,96,3" --enable_small_channel=1 --soc_version=Ascend310P3
  ```

- Build the program

  ```
  bash build.sh
  ```

- Run the program:

  ```
  ./benchmark_tf.sh --batchSize=1 --modelPath=../../model/face_resnet50.om --dataPath=./dataset_bin --modelType=faceresnet50 --imgType=rgb
  ```
  
## Performance

### Result

Our result were obtained by running the applicable inference script. To achieve the same results, follow the steps in the Quick Start Guide.

#### Inference accuracy results

|       model       | ***data***  |    Embeddings Accuracy    |
| :---------------: | :---------: | :---------: |
| offline Inference | 13233 images |   90.52%     |



