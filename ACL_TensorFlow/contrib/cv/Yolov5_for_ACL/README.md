中文|[English](README_EN.md)

# Yolov5 TensorFlow离线推理

此链接提供Yolov5 TensorFlow模型在NPU上离线推理的脚本和方法

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
cd Modelzoo-TensorFlow/ACL_TensorFlow/contrib/cv/Yolov5_for_ACL
```

### 2. 下载数据集和预处理

1. 参考此URL[url](https://github.com/hunglc007/tensorflow-yolov4-tflite/README.md)下载并预处理数据集
操作如下:
```
# run script in /script/get_coco_dataset_2017.sh to download COCO 2017 Dataset
# preprocess coco dataset
cd data
mkdir dataset
cd ..
cd scripts
python coco_convert.py --input ./coco/annotations/instances_val2017.json --output val2017.pkl
python coco_annotation.py --coco_path ./coco 
python img2bin.py --img-dir ./coco/images --bin-dir ./coco/input_bins
mv coco ..
```
生成coco2017测试数据集目录 *data/dataset/*.

### 3. 离线推理

**离线模型转换**

- 环境变量设置

  请参考[说明](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719)，设置环境变量

- Pb模型转换为om模型

  ```
  atc --model=yolov5_tf2_gpu.pb --framework=3 --output=yolov5_tf2_gpu --soc_version=Ascend310 --input_shape="Input:1,640,640,3" --out_nodes="Identity:0;Identity_1:0;Identity_2:0;Identity_3:0;Identity_4:0;Identity_5:0" --log=info
  ```

- 编译程序

  ```
  bash build.sh
  ```

- 开始运行:

  ```
  cd offline_inference
  bash benchmark_tf.sh
  ```
  
- 运行后期处理:

  ```
  cd ..
  python3 offline_inference/postprocess.py
  ```
  
## 推理结果

### 结果

本结果是通过运行上面适配的推理脚本获得的。要获得相同的结果，请按照《快速指南》中的步骤操作。

#### 推理精度结果

|       model       | **data**  |   AP/AR   |
| :---------------: | :-------: | :-----------: |
| offline Inference | 4952 images | 0.221/0.214 |
  

## 参考
[1] https://github.com/hunglc007/tensorflow-yolov4-tflite

[2] https://github.com/ultralytics/yolov5

[3]https://github.com/khoadinh44/YOLOv5_customized_data
