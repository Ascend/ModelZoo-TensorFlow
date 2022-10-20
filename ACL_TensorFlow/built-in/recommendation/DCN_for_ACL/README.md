中文|[English](README_EN.md)

# DCN Tensorflow 离线推理

此链接提供 **Deep & Cross Network for Ad Click Predictions** 模型在NPU上离线推理的脚本和方法。 原始训练模型请点击: [DCN_for_Tensorflow](https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/built-in/recommendation/DCN_ID1986_for_TensorFlow)

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
cd Modelzoo-TensorFlow/ACL_TensorFlow/built-in/recommendation/DCN_for_ACL
```

### 2. 下载数据集和预处理

1.请自行下载Criteo 数据集，并将 **Criteo/train.txt** 移至 **scripts**。

2.将数据集分成训练和测试集(0.8:0.2),并将测试数据集预处理成 **batchsize=4000** 的bin文件：
```
cd scripts
python3 data_preprocess.py Criteo/train.txt
```
将会生成 **input_x**, **labels** 目录，其 **batchsize=4000**:
```
input_x
|___batch1_X.bin
|___batch2_X.bin
...

labels
|___batch1_Y.bin
|___batch2_Y.bin
...
```

### 3. 离线推理

**离线模型转换**

- 环境变量设置

  请参考[说明](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719)，设置环境变量


- Pb模型转换为om模型

  [pb模型下载链接](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/modelzoo/Official/recommendation/DCN_for_ACL.zip)

  ```
  atc --model=dcn_tf.pb --framework=3 --output=dcn_tf_4000batch --output_type=FP32 --soc_version=Ascend310 --input_shape="input_1:4000,39" --input_format=ND --log=info
  ```

- 编译程序

  ```
  bash build.sh
  ```
 benchmark 工具的运行结果将会生成在 **Benchmark/output/**  路径下: 

- 开始运行:

  ```
  cd scripts
  bash benchmark_tf.sh
  ```



## 性能

### 结果

本结果是通过运行上面适配的推理脚本获得的。要获得相同的结果，请按照《快速指南》中的步骤操作

#### 推理精度结果:

| Test Dataset | Accuracy-ROC |Accuracy-PR |
|--------------|-------------------|---------|
|  Criteo        | 80.5%             | 59.8% |

## 参考
[1] https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/built-in/recommendation/DCN_ID1986_for_TensorFlow
