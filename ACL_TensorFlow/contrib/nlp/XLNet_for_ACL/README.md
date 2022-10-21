中文|[English](README_EN.md)

# XLNet TensorFlow离线推理

此链接提供XLNet TensorFlow模型在NPU上离线推理的脚本和方法

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
cd Modelzoo-TensorFlow/ACL_TensorFlow/contrib/nlp/XLNet_for_ACL
```

### 2. 下载数据集和预处理

1. 下载指南 [**train repo**](https://github.com/zihangdai/xlnet), 请自行下载这个数据集 **glue_data** 接着移动到这个目录下面： **scripts/**
```
scripts
|
|___glue_data
     |
     |___STS-B
         |
         |___LICENSE.txt
         |___dev.tsv
         |___original
         |___readme.txt
         |___test.tsv
         |___train.tsv
```
然后将STS-B数据集转换为tfrecord，转换为**scripts/proc_data/**，如下所示：
```
proc_data
|
|___sts-b
    |__spiece.model.len-128.dev.eval.tf_record
    |__spiece.model.len-128.test.predict.tf_record
    |__spiece.model.len-128.train.eval.tf_record
```
Here we use dev-eval tfrecord for demostration.

2. 将tfrecord转换为bin文件
```
cd scripts
python3 data_preprocess.py
```

### 3. 离线推理

**离线模型转换**

- 环境变量设置

  请参考[说明](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719)，设置环境变量

- Pb模型转换为om模型
  
  [**pb模型下载链接**](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/modelzoo/Research/nlp/XLNET_tf.pb)

  batchsize=1

  ```
  atc --model=XLNet_tf.pb  --framework=3 --input_shape="input_ids_1:1,128;input_mask_1:1,128;segment_ids_1:1,128" --output=./XLNet_1batch --soc_version=Ascend310 --log=info
  ```

- 编译程序

  ```
  bash build.sh
  ```
  An executable file **benchmark** will be generated under the path: **Benchmark/output/**

- 开始运行:

  ```
  cd scripts
  bash benchmark_tf.sh
  ```

## 推理结果

### 结果

本结果是通过运行上面适配的推理脚本获得的。要获得相同的结果，请按照《快速指南》中的步骤操作。

#### 推理精度结果

|       model       |  dataset   |     Number of samples     |   eval_pearsonr   |
|-------------------|---- |--------------|---------|
| offline Inference |  sts-b   |1500 samples  | 91.85%  |

## 参考
[1] https://github.com/zihangdai/xlnet
