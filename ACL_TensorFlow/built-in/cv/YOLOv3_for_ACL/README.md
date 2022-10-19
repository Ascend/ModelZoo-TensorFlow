中文|[English](README_EN.md)

# Yolov3 TensorFlow离线推理 

此链接提供Yolov3 TensorFlow模型在NPU上离线推理的脚本和方法

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
cd Modelzoo-TensorFlow/ACL_TensorFlow/built-in/cv/YOLOv3_for_ACL
```

### 2. 必要条件

opencv-python==4.2.0.34


### 3. 下载数据集和预处理

1. 数据集
  例如，与官方实施相比，我们使用 [get_coco_dataset.sh](https://github.com/pjreddie/darknet/blob/master/scripts/get_coco_dataset.sh) 准备我们的数据集。

2. 注释文件

   cd scripts

   Using script generate `coco2014_minival.txt` file. Modify the path in `coco_minival_anns.py` and `5k.txt`, then execute:

   ```
   python3 coco_minival_anns.py
   ```

   One line for one image, in the format like `image_index image_absolute_path img_width img_height box_1 box_2 ... box_n`.    
   Box_x format: 

   - `label_index x_min y_min x_max y_max`. (The origin of coordinates is at the left top corner, left top => (xmin, ymin), right bottom => (xmax, ymax).)    
   - `image_index` is the line index which starts from zero. `label_index` is in range [0, class_num - 1].

   例如:

   ```
   0 xxx/xxx/a.jpg 1920 1080 0 453 369 473 391 1 588 245 608 268
   1 xxx/xxx/b.jpg 1920 1080 1 466 403 485 422 2 793 300 809 320
   ...
   ```


### 3. 离线推理

**离线模型转换**

- 环境变量设置

  请参考[说明](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719)，设置环境变量

- Pb模型转换为om模型

  [pb模型下载链接](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/modelzoo/yolov3_tf.pb)

  For Ascend310:
  ```
  atc --model=yolov3_tf.pb --framework=3 --output=yolov3_tf_aipp --output_type=FP32 --soc_version=Ascend310 --input_shape="input:1,416,416,3" --log=info --insert_op_conf=yolov3_tf_aipp.cfg
  ```
  For Ascend310P3:
  ```
  atc --model=yolov3_tf.pb --framework=3 --output=yolov3_tf_aipp --output_type=FP32 --soc_version=Ascend310P3 --input_shape="input:1,416,416,3" --log=info --insert_op_conf=yolov3_tf_aipp.cfg
  ```

- 编译程序

  For Ascend310:
  ```
  unset ASCEND310P3_DVPP
  bash build.sh
  ```
  For Ascend310P3:
  ```
  export ASCEND310P3_DVPP=1
  bash build.sh
  ```

- 开始运行:

  ```
  cd scripts
  bash benchmark_tf.sh --batchSize=1 --modelType=yolov3 --imgType=raw --precision=fp16 --outputType=fp32 --useDvpp=1 --deviceId=0 --modelPath=yolov3_tf_aipp.om --trueValuePath=instance_val2014.json --imgInfoFile=coco2014_minival.txt --classNamePath=../../coco.names
  ```



## 推理结果

### 结果

本结果是通过运行上面适配的推理脚本获得的。要获得相同的结果，请按照《快速指南》中的步骤操作。

#### 推理精度结果

IoU=0.5
| model  | Npu_nums | **mAP** | 
| :----: | :------: | :-----: | 
| Yolov3 |    1     |  55.3%   | 

## 参考
[1] https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/built-in/cv/detection/YoloV3_ID0076_for_TensorFlow
