中文|[English](README_EN.md)

# FasterRCNN TensorFlow离线推理

此链接提供FasterRCNN TensorFlow模型在NPU上离线推理的脚本和方法

## 注意
**此案例仅为您学习Ascend软件栈提供参考，不用于商业目的。**

在开始之前，请注意以下适配条件。如果不匹配，可能导致运行失败。

| Conditions | Need |
| --- | --- |
| CANN版本 | >=5.0.3 |
| 芯片平台| Ascend310/Ascend310P3 |
| 第三方依赖| 请参考 'requirements.txt' |

## 快速指南

### 1. 拷贝代码

```shell
git clone https://gitee.com/ascend/ModelZoo-TensorFlow.git
cd Modelzoo-TensorFlow/ACL/Official/cv/FasterRCNN_for_ACL
```

### 2. 下载数据集和预处理

1. 访问“datapreprocess”目录
2. 下载并生成TFRecords数据集 [COCO 2017](http://cocodataset.org/#download).

```
   bash download_and_preprocess_mscoco.sh <data_dir_path>
```
   注意：数据将被下载，预处理为tfrecords格式，并保存在<Data_dir_path>目录中（主机上）。如果您已经下载并创建了TFRecord文件（根据tensorflow的官方tpu脚本生成的TFRecord），请跳过此步骤。 
         如果您已经下载了COCO映像，请运行以下命令将其转换为TFRecord格式

         ```
         python3 object_detection/dataset_tools/create_coco_tf_record.py --include_masks=False --val_image_dir=/your/val_tfrecord_file/path --val_annotations_file=/your/val_annotations_file/path/instances_val2017.json --output_dir=/your/tfrecord_file/out/path
         ```
    
3. 将数据集转成bin文件
```
   python3 data_2_bin.py --validation_file_pattern /your/val_tfrecord_file/path/val_file_prefix* --binfilepath /your/bin_file_out_path 
```
4. 创建两个数据集文件夹，一个是用于“image_info”和“images”文件的your_data_path，另一个是“source_ids”文件的your_datasourceid_path。将bin文件移动到正确的目录；
5. 将“instances_val2017.json”复制到FasterRCNN_for_ACL/scripts 目录下；
 
### 3. Offline Inference

**Pb模型转换为om模型**
- 访问 "FasterRCNN_for_ACL" 文件夹

- 环境变量设置

 请参考[说明](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719)，设置环境变量

- Pb模型转换为om模型

  ```
  atc --model=/your/pb/path/your_fast_pb_name.pb --framework=3  --output=your_fastom_name--output_type=FP32 --soc_version=Ascend310P3 --input_shape="image:1,1024,1024,3;image_info:1,5" --keep_dtype=./keeptype.cfg  --precision_mode=force_fp16  --out_nodes="generate_detections/combined_non_max_suppression/CombinedNonMaxSuppression:3;generate_detections/denormalize_box/concat:0;generate_detections/add:0;generate_detections/combined_non_max_suppression/CombinedNonMaxSuppression:1"
  ```
注意: 替换模型参数, 输出, 环境变量

- 编译程序

  ```
  bash build.sh
  ```

- Run the program:

  ```
  cd scripts
  chmod +x benchmark_tf.sh
  ./benchmark_tf.sh --batchSize=1 --modelType=fastrcnn16  --outputType=fp32  --deviceId=2 --modelPath=/your/fastom/path/your_fastom_name.om --dataPath=/your/data/path --innum=2 --suffix1=image_info.bin --suffix2=images.bin --imgType=raw  --sourceidpath=/your/datasourceid/path
  ```
注意：替换modelPath、dataPath和sourceidpath的参数。使用绝对路径。



## 精度

### 结果

本结果是通过运行上面适配的推理脚本获得的。要获得相同的结果，请按照《快速指南》中的步骤操作。

#### 推理精度结果

|       model       | **data**    |      Bbox      |
| :---------------: | :-------:   | :------------: |
| offline Inference | 5000 images |      35.4%     |
