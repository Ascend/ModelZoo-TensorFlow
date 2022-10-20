中文|[English](README_EN.md)

# CTPN Tensorflow离线推理

此链接提供CTPN TensorFlow模型在NPU上离线推理的脚本和方法

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
cd Modelzoo-TensorFlow/ACL_TensorFlow/built-in/cv/CTPN_for_ACL
```

### 2. 下载数据集和预处理

1. 请自行下载数据集
   ```
   https://github.com/eragonruan/text-detection-ctpn
   ```

2. 运行预处理脚本
   ```
   python3 scripts/ctpn_dataPrepare.py --image_path=../image --output_path=$datasets --crop_width=1072 --crop_height=608 --img_conf=./img_info
   
   ```
3. 安装 BBOX
   ```
   cd scripts/utils/bbox
   bash make.sh
   ```
4. 自行下载icdar 2013测试数据集ground truth文件，并放到test目录下
   
### 3. 离线推理

**离线模型转换**

  [pb模型下载链接](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/modelzoo/Official/cv/CTPN_for_ACL.zip)

- 环境变量设置

  请参考[说明](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719)，设置环境变量


- Pb模型转换为om模型

  ```
  atc --model=model/ctpn_tf.pb --framework=3 --output=model/ctpn_model --input_shape=input_image:1,608,1072,3 --soc_version=Ascend310P3 --enable_scope_fusion_passes=ScopeDynamicRNNPass --enable_small_channel=1
  ```

- 编译程序 

  ```
  bash build.sh
  ```

- 开始运行:

  ```
  bash benchmark_tf.sh --batchSize=1 --modelPath=../../model/ctpn_model.om --dataPath=../../datasets/ --modelType=ctpn  --imgType=rgb 
  ```
  
## 性能

### 结果

本结果是通过运行上面适配的推理脚本获得的。要获得相同的结果，请按照《快速指南》中的步骤操作。

#### 推理精度结果

|       model       | **data**   |    precision    |    recall       |    heamn        |
| :---------------: | :-------:  | :-------------: | :-------------: | :-------------: |
| offline Inference | 233 images |    74.73%       |    70.26%       |    72.43%       |

