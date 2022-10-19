中文|[English](README_EN.md)

# FACE-RESNET50 TensorFlow离线推理

此链接提供 face-resnet50 TensorFlow模型在NPU上离线推理的脚本和方法

## 注意
**此案例仅为您学习Ascend软件栈提供参考，不用于商业目的。**

在开始之前，请注意以下适配条件。如果不匹配，可能导致运行失败。

| Conditions | Need |
| --- | --- |
|  CANN版本 | >=5.0.3 |
| 芯片平台| Ascend310/Ascend310P3 |
| 第三方依赖| 请参考 'requirements.txt' |

## 快速指南

### 1. 拷贝代码

```shell
git clone https://gitee.com/ascend/ModelZoo-TensorFlow.git
cd Modelzoo-TensorFlow/ACL_TensorFlow/built-in/cv/Face_Resnet50_for_ACL
```

### 2. 下载数据集和预处理

1. 请自行下载lfw测试数据集 

2. 执行预处理脚本
   ```
   python3 align/align_dataset_mtcnn_facereset.py $cur_dir/lfw $dataset
   python3 preprocess.py $cur_dir/config/basemodel.py Path_of_Data_after_processing
   
   ```
 
### 3. 离线推理

**离线模型转换**

- 获取基本模型
  ```
  https://github.com/seasonSH/DocFace
  
  ```
- ckpt转换为pb模型
  ```
  for example:
   python3.7 /usr/local/python3.7.5/lib/python3.7/site-packages/tensorflow_core/python/tools/freeze_graph.py --input_checkpoint=./faceres_ms/ckpt-320000 --output_graph=./model/face_resnet50_tf.pb --output_node_names="embeddings" --input_meta_graph=./faceres_ms/graph.meta --input_binary=true
  ```

- 环境变量设置

  请参考[说明](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719)，设置环境变量

- Pb模型转换为om模型

  [pb模型下载链接](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/modelzoo/Official/cv/Face_Resnet50_for_ACL.zip)

  ```
  /usr/local/Ascend/atc/bin/atc --model ./model/face_resnet50_tf.pb   --framework=3  --output=face_resnet50 --input_shape="image_batch:1,112,96,3" --enable_small_channel=1 --soc_version=Ascend310P3
  ```

- 编译程序

  ```
  bash build.sh
  ```

- 开始运行:

  ```
  ./benchmark_tf.sh --batchSize=1 --modelPath=../../model/face_resnet50.om --dataPath=./dataset_bin --modelType=faceresnet50 --imgType=rgb
  ```
  
## 性能

### 结果

本结果是通过运行上面适配的推理脚本获得的。要获得相同的结果，请按照《快速入门指南》中的步骤操作。

#### 推理精度结果

|       model       | ***data***  |    Embeddings Accuracy    |
| :---------------: | :---------: | :---------: |
| offline Inference | 13233 images |   90.52%     |



