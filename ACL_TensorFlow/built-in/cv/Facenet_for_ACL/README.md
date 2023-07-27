中文|[English](README_EN.md)

# FACENET Tensorflow离线推理

此链接提供FACENET TensorFlow模型在NPU上离线推理的脚本和方法

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
cd Modelzoo-TensorFlow/ACL_TensorFlow/built-in/cv/Facenet_for_ACL
```

### 2. 下载数据集和预处理

1. 请自行下载 lfw 数据集

2. 运行预处理脚本
   ```
   python3 align/align_dataset_mtcnn.py $cur_dir/lfw $dataset --image_size 160 --margin 32 --random_order
   python3 preprocess_data.py  Path_of_Data_after_face_alignment  Outpath_of_Data_after_face_alignment  --use_fixed_image_standardization --lfw_batch_size 1 --use_flipped_images
   
   ```

 
### 3. 离线推理

**离线模型转换**

  [pb模型下载链接(20180402)](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/modelzoo/Official/cv/Facenet_for_ACL.zip)  
  [pb模型下载链接(20180408)](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/2022-09-24_tf/FaceNet_for_ACL/facenet_20180408-102900.pb)

- 环境变量设置

  请参考[说明](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719)，设置环境变量

- Pb模型转换为om模型
  
  pb模型样例(20180402):

  ```
   atc --framework=3 --model=./model/facenet_tf.pb  --output=./model/facenet --soc_version=Ascend310P3 --insert_op_conf=./facenet_tensorflow.cfg --input_format=NHWC --input_shape=input:64,160,160,3
  ```

### 4.量化

1.请自行下载amct工具包  https://support.huawei.com/carrier/navi?coltype=software#col=software&detailId=PBI1-256970475&path=PBI1-21430725/PBI1-21430756/PBI1-22892969/PBI1-23710427/PBI1-251168373

```shell
cd amct/amct_tf/
pip3 install amct_tensorflow-2.16.8-py3-none-linux_x86_64.tar.gz
```

2.重新导出一份原始数据集用于量化

python3 pre_process_data_forquant.py Path_of_Data_after_face_alignment  Outpath_of_Data_after_face_alignment  --use_fixed_image_standardization --lfw_batch_size 1 --use_flipped_images

3.量化模型

python3 amct_python.py ./facenet_20180408-102900.pb ./datasets_bin/data_image_bin_original ./quant

4.pb转om

mv ./quant/facenet_quantized.pb ./

atc --framework=3 --model=./facenet_quantized.pb  --output=./facenet_quant --soc_version=Ascend310P3 --insert_op_conf=./facenet_tensorflow.cfg --input_format=NHWC --input_shape=input:64,160,160,3


- 编译程序

  ```
  bash build.sh
  ```

- 开始运行:

  ```
  ./benchmark_tf.sh --batchSize=4 --modelPath=absolute_path_of_facenet.om --dataPath=absolute_path_of_dataset_bin --modelType=facenet --imgType=rgb
  ```
  
## 性能

### 结果

本结果是通过运行上面适配的推理脚本获得的。要获得相同的结果，请按照《快速指南》中的步骤操作。

#### 推理精度结果

|       model    |       mode       | ***data***  |    Embeddings Accuracy    |
| :---------------:| :---------------: | :---------: | :---------: |
| pb(20180402)| offline Inference | 12000 images |   99.550%     |
| pb(20180408)| offline Inference | 12000 images |   99.133%     |

### 6.量化模型精度性能

1.执行推理，自行安装ais_bench工具

2.执行 python3 -m ais_bench --model ./facenet_quant.om --input datasets_bin/data_image_bin --output ./output --device 0

3. 精度验证

python3 post2.py ../datasets ../output/2023_05_11-10_55_20 ../datasets_bin/data_label_bin --lfw_batch_size 1 --distance_metric 1 --use_flipped_images --subtract_mean

