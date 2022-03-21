

# FasterRCNN Inference for Tensorflow 

This repository provides a script and recipe to Inference the FasterRCNN model.

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
cd modelzoo/built-in/ACL/Official/cv/FasterRCNN_for_ACL
```

### 2. Download and preprocess the dataset

1. Access  to the "datapreprocess" directory.
2. Download and build TFRecords of the dataset，[COCO 2017](http://cocodataset.org/#download).

```
   bash download_and_preprocess_mscoco.sh <data_dir_path>
```
   Note: Data will be downloaded, preprocessed to tfrecords format and saved in the <data_dir_path> directory (on the host). Or if you have downloaded and created the TFRecord file (TFRecord generated based on the official tpu script of tensorflow), skip this step. 
         Or if you have downloaded the COCO images, run the following command to convert them to TFRecord.

         ```
         python3 object_detection/dataset_tools/create_coco_tf_record.py --include_masks=False --val_image_dir=/your/val_tfrecord_file/path --val_annotations_file=/your/val_annotations_file/path/instances_val2017.json --output_dir=/your/tfrecord_file/out/path
         ```
    
2. Transfer to Bin file.
```
   python3 data_2_bin.py --validation_file_pattern /your/val_tfrecord_file/path/val_file_prefix* --binfilepath /your/bin_file_out_path 
```
4. Create 2 dataset folders, one is your_data_path for "image_info" and "images" files, and one is your_datasourceid_path for "source_ids" files. Move your bin files to the correct directory.
5. Copy the "instances_val2017.json" to the FasterRCNN_for_ACL/scripts.
 
### 3. Offline Inference

**Convert pb to om.**
- Access  to the "FasterRCNN_for_ACL" directory.

- Configure the env

  ```
  export install_path=/usr/local/Ascend
  export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
  export PYTHONPATH=${install_path}/opp/op_impl/built-in/ai_core/tbe:$PYTHONPATH
  export LD_LIBRARY_PATH=${install_path}/atc/lib64:${install_path}/acllib/lib64/:$LD_LIBRARY_PATH
  export ASCEND_OPP_PATH=${install_path}/opp
  ```
Notes: Replace the values of install_path.

- Convert pb to om

  ```
  atc --model=/your/pb/path/your_fast_pb_name.pb --framework=3  --output=your_fastom_name--output_type=FP32 --soc_version=Ascend710 --input_shape="image:1,1024,1024,3;image_info:1,5" --keep_dtype=./keeptype.cfg  --precision_mode=force_fp16  --out_nodes="generate_detections/combined_non_max_suppression/CombinedNonMaxSuppression:3;generate_detections/denormalize_box/concat:0;generate_detections/add:0;generate_detections/combined_non_max_suppression/CombinedNonMaxSuppression:1"
  ```
Notes: Replace the values of model, output, soc_version

- Build the program

  ```
  bash build.sh
  ```

- Run the program:

  ```
  cd scripts
  chmod +x benchmark_tf.sh
  ./benchmark_tf.sh --batchSize=1 --modelType=fastrcnn16  --outputType=fp32  --deviceId=2 --modelPath=/your/fastom/path/your_fastom_name.om --dataPath=/your/data/path --innum=2 --suffix1=image_info.bin --suffix2=images.bin --imgType=raw  --sourceidpath=/your/datasourceid/path
  ```
Notes: Replace the values of modelPath， dataPath， and sourceidpath. Use an absolute path.



## Accuracy 

### Result

Our result were obtained by running the applicable training script. To achieve the same results, follow the steps in the Quick Start Guide.

#### Inference accuracy results

|       model       | **data**    |      Bbox      |
| :---------------: | :-------:   | :------------: |
| offline Inference | 5000 images |      35.4%     |
