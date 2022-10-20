中文|[English](README_EN.md)

# NIMA TensorFlow离线推理
此链接提供NIMA TensorFlow模型在NPU上离线推理的脚本和方法

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

   ```
   git clone https://gitee.com/ascend/ModelZoo-TensorFlow.git
   cd Modelzoo-TensorFlow/ACL_TensorFlow/contrib/cv/NIMA_for_ACL
   ```

### 2. 下载数据集和预处理


1.  请自行下载AVA测试数据集（包含5000张图片和AVA.txt） 

2.  将AVA测试数据集移动到“scripts/AVA_dataset_test”，如下所示：

    ```
    AVA_DATASET_TEST
    |
    |__image
    |   |____12315.jpg
    |   |____12316.jpg
    |   .....
    |__AVA.txt

    ```

3.  图片预处理
    
    ```
    cd scripts
    mkdir input_bins
    python3 data_preprocess.py AVA_DATASET_TEST ./input_bins/

    ```    
图片将被预处理为input_bins文件。标签将被预处理为predict_txt文件

### 3.离线推理
 
1.环境变量设置

  请参考[说明](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719)，设置环境变量
   
2.Pb模型转换为om模型

[**pb模型下载链接**](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/modelzoo/Research/cv/NIMA_for_ACL.zip)

```
atc --model=./nima.pb --framework=3 --output=./nima_1batch_input_fp16_output_fp32 --soc_version=Ascend310 --input_shape="input_1:1,224,224,3" --soc_version=Ascend310
```
3.编译程序
```  
bash build.sh
```
将在benchmark/output目录下生成可执行文件：**benchmark**

4.开始运行:
```  
cd scripts
bash benchmark_tf.sh
```

## 性能

### 结果

本结果是通过运行上面适配的推理脚本获得的。要获得相同的结果，请按照《快速指南》中的步骤操作。

#### 推理精度结果
--------------------------
|       Dataset       |     Numbers     |   SSRC   |
|-------------------|--------------|---------|
| AVA test dataset | 5000 images  | 51.78%  |


## 参考
[1] https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/NIMA/NIMA_ID0853_for_TensorFlow
