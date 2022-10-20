中文|[English](README_EN.md)
# <font face="微软雅黑">

# OpenPose TensorFlow离线推理
此链接提供OpenPose TensorFlow模型在NPU上离线推理的脚本和方法

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
cd Modelzoo-TensorFlow/ACL_TensorFlow/built-in/cv/OpenPose_for_ACL
```

### 2. 下载数据集和预处理

请自行下载COCO2014测试数据集, 详情见: [dataset](./dataset/coco/README.md)


### 3. 获取pb模型

获取OpenPose pb模型, 详情见: [models](./models/README.md)

### 4. 获取处理脚本

pafprocess、slidingwindow 下载链接: [tf_openpose](https://github.com/BoomFan/openpose-tf/tree/master/tf_pose) and put them into libs


### 5. 离线推理
**数据预处理**
```Bash
python3 preprocess.py \
    --resize 656x368 \
    --model cmu \
    --coco-year 2014 \
    --coco-dir ../dataset/coco/ \
    --output-dir ../input/

```

**Pb模型转换为om模型**
- 环境变量设置

  请参考[说明](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719)，设置环境变量



- Pb模型转换为om模型

  [pb模型下载链接](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/modelzoo/Official/cv/OpenPose_for_ACL.zip)

  ```
  atc --framework=3 \
      --model=./models/OpenPose_for_TensorFlow_BatchSize_1.pb \
      --output=./models/OpenPose_for_TensorFlow_BatchSize_1 \
      --soc_version=Ascend310 \
      --input_shape="image:1,368,656,3"
  ```

**编译程序**
编译推理应用程序, 详情见: [xacl_fmk](./xacl_fmk/README.md)

**开始运行**
```
/xacl_fmk -m ./models/OpenPose_for_TensorFlow_BatchSize_1.om \
    -o ./output/openpose \
    -i ./input \
    -b 1
```

**后处理**
```
python3 postprocess.py \
    --resize 656x368 \
    --resize-out-ratio 8.0 \
    --model cmu \
    --coco-year 2014 \
    --coco-dir ../dataset/coco/ \
    --data-idx 100 \
    --output-dir ../output/openpose 
```

**样本脚本**
我们还支持使用predict_openpose.sh运行上述所有步骤，**构建程序除外**

### 6.结果
***
OpenPose 推理 ：

| Type | IoU | Area | MaxDets | Result |
| :------- | :------- | :------- | :------- | :------- |
| Average Precision  (AP) | 0.50:0.95 | all | 20 | 0.399 |
| Average Precision  (AP) | 0.50 | all | 20 | 0.648 |
| Average Precision  (AP) | 0.75| all | 20 | 0.400 |
| Average Precision  (AP) | 0.50:0.95 | medium | 20 | 0.364 |
| Average Precision  (AP) | 0.50:0.95 | large | 20 | 0.443 |
| Average Recall     (AR) | 0.50:0.95 | all | 20 | 0.456 |
| Average Recall     (AR) | 0.50 | all | 20 | 0.683 |
| Average Recall     (AR) | 0.75 | all | 20 | 0.465 |
| Average Recall     (AR) | 0.50:0.95 | medium | 20 | 0.371 |
| Average Recall     (AR) | 0.50:0.95 | large | 20 | 0.547 |

***

## 参考

[1] https://github.com/ildoonet/tf-pose-estimation/tree/master/tf_pose/pafprocess/


# </font>
