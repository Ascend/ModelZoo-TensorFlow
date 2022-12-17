中文|[English](README_EN.md)

# Siamese TensorFlow离线推理

此存储库提供了推断模型的脚本和方法。原始训练推理实施请遵循以下链接：[Siamese_for_Tensorflow](https://github.com/dhwajraj/deep-siamese-text-similarity)
and in this repo we trained a model for **Phrase Similarity**.

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
cd Modelzoo-TensorFlow/ACL_TensorFlow/contrib/cv/Siamese_for_ACL
```

### 2. 下载数据集和预处理

1. 当训练完成, **validation.txt0** 和 **vocab** 在 **runs/xxxx/checkpoints/** 下面生成. 复制到这个路径 **scripts/dataset** 。

2. 验证数据集的预处理:
```
cd scripts
python3 data_preprocess.py
```
将生成 **input_x1**, **input_x2**, **ground_truth** directories with batchsize **128**:
```
input_x1
|___000000.bin
|___000001.bin
...

input_x2
|___000000.bin
|___000001.bin
...

ground_truth
|___000000.txt
|___000001.txt
...
```

### 3. 离线推理

**离线模型转换**

- 环境变量设置

  请参考[说明](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719)，设置环境变量

[pb模型下载链接](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/2022-12-12_tf/Siamese_for_ACL/siamese_tf.pb)
- Pb模型转换为om模型

  ```
  atc --model=siamese_tf.pb --framework=3 --output=siamese_tf_128batch --output_type=FP32 --soc_version=Ascend310 --input_shape="input_x1:128,15;input_x2:128,15" --log=info --precision_mode=allow_fp32_to_fp16
  ```

- 编译程序

  ```
  bash build.sh
  ```
  将在路径下生成可执行文件 **benchmark** : **Benchmark/output/**

- 开始运行:

  ```
  cd scripts
  bash benchmark_tf.sh
  ```



## 推理结果

### 结果

本结果是通过运行上面适配的推理脚本获得的。要获得相同的结果，请按照《快速指南》中的步骤操作。

#### 推理精度结果

| Test Dataset | Accuracy |
|--------------|-------------------|
|  vocab        | 94.9%             |

## 参考
[1] https://github.com/dhwajraj/deep-siamese-text-similarity
