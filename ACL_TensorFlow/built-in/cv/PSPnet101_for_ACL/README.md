中文|[English](README_EN.md)

# PSPNet101 TensorFlow离线推理

此链接提供PSPNet101 TensorFlow模型在NPU上离线推理的脚本和方法

## 快速指南

### 1. 拷贝代码

```shell
git clone https://gitee.com/ascend/ModelZoo-TensorFlow.git
cd Modelzoo-TensorFlow/ACL_TensorFlow/built-in/cv/PSPNet101_for_ACL
```

### 2. 下载数据集和预处理

1. 请自行下载测试数据集
2. 执行预处理脚本
   ```
   #无翻转
   python3 scripts/data_processing.py --img_num=500 --crop_width=720 --crop_height=720 --data_dir=../cityscapes --val_list=../cityscapes/list/cityscapes_val_list.txt --output_path=$dataset
   ```

   ```
   #翻转
   python3 scripts/data_processing.py --img_num=500 --crop_width=720 --crop_height=720 --data_dir=../cityscapes --val_list=../cityscapes/list/cityscapes_val_list.txt --output_path=$dataset --flipped_eval --flipped_output_path=$flipped_dataset   
   ```

 
### 3. 离线推理

**Pb模型转换为om模型**

  [pb模型下载链接](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/modelzoo/Official/cv/PSPnet101_for_ACL.zip)

- 环境变量设置

  请参考[说明](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719)，设置环境变量

- Pb模型转换为om模型

  ```
  atc --model=model/PSPNet101.pb --framework=3 --output=model/pspnet101_1batch --soc_version=Ascend310P3 --input_shape=input_image:1,1024,2048,3 --enable_small_channel=1 --insert_op_conf=pspnet_aipp.cfg
  ```

- 编译程序

  ```
  bash build.sh
  ```

- 开始运行:

  ```
  无翻转
  bash benchmark_tf.sh --batchSize=1 --outputType=fp32 --modelPath=../../model/pspnet101_1batch.om --dataPath=../../datasets/ --modelType=PSPnet101 --imgType=rgb
  ```

  ```
  翻转
  bash benchmark_tf.sh --batchSize=1 --outputType=fp32 --modelPath=../../model/pspnet101_1batch.om --dataPath=../../datasets/ --modelType=PSPnet101 --imgType=rgb --flippedDataPath=../../flipped_datasets/ --flippedEval=1
  ```
  
## 性能

### 结果

本结果是通过运行上面适配的推理脚本获得的。要获得相同的结果，请按照《快速指南》中的步骤操作。

#### 无翻转推理精度结果

|       model       | **data**   |    mIoU    | 
| :---------------: | :-------:  | :--------: | 
| offline Inference | 500 images |    77%     | 


### 翻转推理精度结果

|       model       | **data**   |    mIoU    |    
| :---------------: | :-------:  | :--------: | 
| offline Inference | 500 images |   77.24%   | 

