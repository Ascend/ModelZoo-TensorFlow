中文|[English](README_EN.md)

# CRNN Tensorflow离线推理

此链接提供CRNN TensorFlow模型在NPU上离线推理的脚本和方法。原始训练模型请点击此链接: [CRNN_for_Tensorflow](https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/built-in/cv/detection/CRNN_for_TensorFlow)

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
cd ModelZoo-TensorFlow/tree/master/ACL_TensorFlow/built-in/cv/CRNN_for_ACL
```

### 2. 下载数据集和预处理

1.请自行下载IIIT5K/ICDAR03/SVT测试数据集，并将其放在 **scripts/data/** 中

2.测试数据集和标签的预处理
```
cd scripts
python3 tools/preprocess.py
```
将会生成 **img_bin** 和 **labels** 目录:
```
img_bin
|___batch_data_000.bin
|___bathc_data_001.bin
...

labels
|___batch_label_000.txt
|___batch_label_001.txt
...
```

### 3. 离线推理
**ckpt冻结ob**

请使用训练脚本中的frozen_graph.py: [https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/built-in/cv/detection/CRNN_for_TensorFlow/tools/frozen_graph.py)
```
python3 frozen_graph.py --ckpt_path= ckpt_path/shadownet_xxx.ckpt-600000
```

**离线模型转换**

  [pb模型下载链接](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/2022-09-24_tf/CRNN_for_ACL/shadownet_tf_64batch.pb)

- 环境变量设置

  请参考[说明](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719)，设置环境变量


- Pb模型转换为om模型

  ```
  atc --model=shadownet_tf_64batch.pb --framework=3 --output=shadownet_tf_64batch --output_type=FP32 --soc_version=Ascend310 --input_shape="test_images:64,32,100,3" --log=info
  ```

- 编译程序

  ```
  bash build.sh
  ```
  **benchmark** 工具的运行结果将会生成在 **Benchmark/output/** 路径下

- 开始运行:

  ```
  cd scripts
  bash benchmark_tf.sh
  ```



## 性能

### 结果

本结果是通过运行上面适配的推理脚本获得的。要获得相同的结果，请按照《快速指南》中的步骤操作。

#### 推理精度结果:

| Test Dataset | Per_Char_Accuracy | Full_Seq_Accuracy |
|--------------|-------------------|-------------------|
| SVT          | 88.9%             | 77.2%             |
| ICDAR2013    | 93.5%             | 87.3%             |
| IIIT5K       | 91.4%             | 79.6%             |

## 参考
[1] https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/built-in/cv/detection/CRNN_for_TensorFlow
