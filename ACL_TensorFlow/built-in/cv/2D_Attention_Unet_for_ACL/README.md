中文|[English](README_EN.md)

# 2D_Attention_Unet离线推理

此链接提供2D_Attention_Unet模型在NPU上离线推理的脚本和方法

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
cd Modelzoo-TensorFlow/ACL_TensorFlow/built-in/cv/2D_Attention_Unet_for_ACL
```

### 2. 下载数据集和预处理

1. 请自行下载Massachusetts Roads数据集


2. 将数据集文件放到 **'2D_Attention_Unet_for_ACL/image_ori/'** 中：
```
--image_ori
  |----lashan
    |----test(xxx.tiff,total:49images)
    |----test_labels
    |----val
    |----val_labels
  |----Val(xxx_gt/xxx_img/xxx_pred .png)
```


3. 执行预处理脚本
   ```
   cd scripts
   python3 preprocessdata_test.py --dataset=../image_ori/lashan --crop_height=224 --crop_width=224
   ```
 
### 3. 离线推理

**离线模型转换**

- 环境变量设置

  请参考[说明](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719)，设置环境变量


- Pb模型转换为om模型


  [pb模型下载链接](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/modelzoo/Official/cv/2D_Attention_Unet_for_ACL.zip)

  ```
  cd ..
  mkdir model
  atc --model=model/2D_Attention_Unet_tf.pb --framework=3 --output=model/2DAttention_fp16_1batch --soc_version=Ascend310P3 --input_shape=inputs:1,224,224,3 --enable_small_channel=1 --insert_op_conf=2DAttention_aipp.cfg
  ```


- 编译程序
  ```
  bash build.sh
  ```

- 开始运行:

  ```
  bash benchmark_tf.sh
  ```
  
## 推理结果

### 结果

本结果是通过运行上面适配的推理脚本获得的。要获得相同的结果，请按照《快速指南》中的步骤操作。

#### 推理精度结果

|       model       | **data**   |       accuracy      |    Road      |    Others    |    precision    |    F1_score    |    Iou    |
| :---------------: | :-------:  | :-----------------: | :----------: | :----------: | :-------------: | :------------: | :-------: |
| offline Inference |  49 images |     97.19%          |    60.25%    |    99.36%    |     97.88%      |      97.44%    |    76.02% |
