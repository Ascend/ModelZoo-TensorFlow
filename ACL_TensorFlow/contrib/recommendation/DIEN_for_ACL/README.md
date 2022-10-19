中文|[English](README_EN.md)

# DIEN TensorFlow离线推理

此链接提供Vgg16 TensorFlow模型在NPU上离线推理的脚本和方法

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
cd Modelzoo-TensorFlow/ACL_TensorFlow/contrib/recommendation/DIEN_for_ACL
```

### 2. 生成随机测试数据集

1. 使用这个连接 [train repo](https://github.com/mouna99/dien), 下载数据集并解压缩以使用:
```
tar -jxvf data.tar.gz
mv data/* scripts
tar -jxvf data1.tar.gz
mv data1/* scripts
tar -jxvf data2.tar.gz
mv data2/* scripts
``` 
你会得到下面的文件:
- cat_voc.pkl 
- mid_voc.pkl 
- uid_voc.pkl 
- local_train_splitByUser 
- local_test_splitByUser 
- reviews-info
- item-info

2. 生成测试数据集:
```
cd scripts
python3 generate_data.py --batchsize=128
```
测试数据集会在这个目录下面 *input_bins/*.

### 3. 离线推理

**离线模型转换**

- 环境变量设置

  请参考[说明](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719)，设置环境变量

- Pb模型转换为om模型

  [**pb模型下载链接**](https://modelzoo-train-atc.obs.cn-north-4.myhuaweicloud.com/003_Atc_Models/modelzoo/Research/recommendation/DIEN_for_ACL/DIEN.pb)

  ```
  export batch_size=128
  atc --model=./DIEN.pb --framework=3 --output=./DIEN_${batch_size}batch --soc_version=Ascend310P3 --input_shape="Inputs/mid_his_batch_ph:${batch_size},100;Inputs/cat_his_batch_ph:${batch_size},100;Inputs/uid_batch_ph:${batch_size};Inputs/mid_batch_ph:${batch_size};Inputs/cat_batch_ph:${batch_size};Inputs/mask:${batch_size},100;Inputs/seq_len_ph:${batch_size}" --out_nodes="final_output:0" --precision_mode="allow_fp32_to_fp16" --customize_dtypes="./customize_dtypes.txt"
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

## 推理结果

### 结果

本结果是通过运行上面适配的推理脚本获得的。要获得相同的结果，请按照《快速指南》中的步骤操作。

#### 推理精度结果

|       model       | **data**  |     MeanAccuracy   |
| :---------------: | :-------: | :-------------: |
| offline Inference | Amazon ProductGraph | 78.5% |

## 参考
[1] https://github.com/mouna99/dien
