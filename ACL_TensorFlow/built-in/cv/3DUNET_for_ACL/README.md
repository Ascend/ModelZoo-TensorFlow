中文|[English](README_EN.md)

# 3DUNET Tensorflow 离线推理

此链接提供3DUNET  TensorFlow模型在NPU上离线推理的脚本和方法

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
cd Modelzoo-TensorFlow/ACL_TensorFlow/contrib/cv/3DUNET_for_Tensorflow
```

### 2. 下载数据集和预处理

1. 请自行下载数据集

2. 将数据集文件放到 **3DUNET_for_ACL/ori_images/** 中:
```
--MICCAI_BraTS_2019_Data_Training

```

3. 运行预处理脚本
   ```
   mkdir ori_images/tfrecord
   python3 scripts/preprocess_data.py --input_dir=ori_images/MICCAI_BraTS_2019_Data_Training/ --output_dir=ori_images/tfrecord
   python3 scripts/prepocess.py ./ori_images/tfrecord/ ./datasets ./labels
   ```
 
### 3. 离线推理

**离线模型转换**

- 环境变量设置

  请参考[说明](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719)，设置环境变量


- Pb模型转换为om模型(Ascend310P3)  
  ```
  atc --model=model/unet3d.pb --framework=3 --output=model/unet3d_1batch --soc_version=Ascend310P3 --input_shape=input:1,224,224,160,4 --enable_small_channel=1
  ```

- Pb模型转换为om模型(Ascend310)
  ```
  atc --model=model/unet3d.pb --framework=3 --output=model/unet3d_1batch --soc_version=Ascend310 --input_shape=input:1,224,224,160,4 --optypelist_for_implmode=ReduceMeanD --op_select_implmode=high_precision
  ```

- 编译程序

  ```
  bash build.sh
  ```

- 开始运行:

  ```
  cd scripts
  bash benchmark_tf.sh
  ```
  
## 性能

### 结果

本结果是通过运行上面适配的推理脚本获得的。要获得相同的结果，请按照《快速指南》中的步骤操作。

#### 推理精度结果

|       model       | **data**   |       TumorCore     | PeritumoralEdema | EnhancingTumor | MeanDice | WholeTumor |   
| :---------------: | :-------:  | :-----------------: |  :-------------: | :------------: |:--------:|:----------:|
| offline Inference |  68 images |        72.59%       |      78.48%      |     70.46%     |  73.84%  |   90.74%   |

