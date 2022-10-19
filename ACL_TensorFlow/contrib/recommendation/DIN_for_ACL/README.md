中文|[English](README_EN.md)

# DIN TensorFlow离线推理

此链接提供DIN TensorFlow模型在NPU上离线推理的脚本和方法

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
cd Modelzoo-TensorFlow/ACL_TensorFlow/contrib/recommendation/DIN_for_ACL
```

### 2. 生成随机测试数据集

1. 根据这个链接 [train repo](https://github.com/zhougr1993/DeepInterestNetwork) 指导下载数据。
```
step 1:
cd ../raw_data
wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz
gzip -d reviews_Electronics_5.json.gz
wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Electronics.json.gz
gzip -d meta_Electronics.json.gz
step 2:
cd utils
python 1_convert_pd.py;
python 2_remap_id.py
step 3:
cd din
build_dataset.py
```
获取数据集: dataset.pkl

2. 生成测试数据集:
```
cd scripts
python3 preprocess.py 
```
在 *input_bins/* 和 dataset_conf.txt下生成测试数据集

### 3. 离线推理

**离线模型转换**

- 环境变量设置

  请参考[说明](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719)，设置环境变量

- Pb模型转换为om模型


  [**pb模型下载链接**](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/2022-09-24_tf/DIN_for_ACL/frozen_din.pb)

  ```
  export batch_size=512
  atc --model=frozen_din.pb --framework=3 --soc_version=Ascend310P3 --output=din_${batch_size}batch_dynamic_shape --log=error --op_debug_level=3 --input_shape_range="Placeholder_1:[100~512];Placeholder_2:[100~512];Placeholder_4:[100~512,-1];Placeholder_5:[100~512]"
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
|       model       |  **data**   |   test_gauc   |   test_auc   |
| :---------------: |  :-------:  | :-----------: | :-----------: |
| offline Inference | dataset.pkl |     0.6854    |     0.6836    |

## 参考
[1] https://github.com/AustinMaster/DeepInterestNetwork/tree/master/din
