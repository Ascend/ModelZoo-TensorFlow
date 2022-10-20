中文|[English](README_EN.md)
# SSD_InceptionV2 TensorFlow离线推理

此链接提供SSD_InceptionV2 TensorFlow模型在NPU上离线推理的脚本和方法。

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
cd Modelzoo-TensorFlow/ACL_TensorFlow/contrib/cv/SSD_InceptionV2_for_ACL
```

### 2. 下载数据集和预处理

1. 请自行下载coco2014测试数据集。

2. 执行预处理脚本
   ```
   cd scripts
   mkdir input_bins
   python3 scripts/ssd_dataPrepare.py --input_file_path=Path of the image --output_file_path=./input_bins --crop_width=640 --crop_height=640 --save_conf_path=./img_info
   
   ```
3. 下载gt标签
   [instances_minival2014.json](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com:443/010_Offline_Inference/Official/cv/ID1654_ssd_resnet50fpn/scripts/instances_minival2014.json?AccessKeyId=APWPYQJZOXDROK0SPPNG&Expires=1656057065&Signature=ydPmdux71bGzs38Q/xV7USQIdCg%3D)

  将json文件放到**'scripts'**
 
### 3. 离线推理

**离线模型转换**

- 环境变量设置

  请参考[说明](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719)，设置环境变量

- Pb模型转换为om模型
  
  [pb模型下载链接](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/modelzoo/Research/cv/SSD_for_ACL/ssd_inceptionv2_tf.pb)

  ```
  atc --model=model/ssd_inceptionv2_tf.pb --framework=3 --output=model/ssd_inceptionv2 --output_type=FP16 --soc_version=Ascend310P3 --input_shape="image_tensor:1,640,640,3" --out_nodes="detection_boxes:0;detection_scores:0;num_detections:0;detection_classes:0"
  ```

- 编译程序

  ```
  bash build.sh
  ```

- 开始运行:

  ```
  bash benchmark.sh
  ```
  
## 推理结果

### 结果

本结果是通过运行上面适配的推理脚本获得的。要获得相同的结果，请按照《快速指南》中的步骤操作。

#### 推理精度结果

|       model       | ***data***  |    map      |
| :---------------: | :---------: | :---------: |
| offline Inference | 4952 images |   27.8%     |

## 参考

[1] https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md

