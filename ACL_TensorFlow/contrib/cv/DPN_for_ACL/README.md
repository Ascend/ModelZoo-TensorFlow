中文|[English](README_EN.md)

# DPN Tensorflow离线推理

此链接提供DPN TensorFlow模型在NPU上离线推理的脚本和方法

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
cd Modelzoo-TensorFlow/ACL_TensorFlow/contrib/cv/DPN_for_ACL
```

### 2. 下载数据集和预处理

1. 请自行下载ImageNet2012测试数据集 ([下载](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/007_inference_backup/dpn/dpn_tf_hw34064571/offline_inference/dataset/dpnval.tfrecords))并将其放入: **scripts/dataset/** 中：

2. 测试数据集和标签的预处理:
```
cd scripts
mkdir input_bins
python3 dpn_preprocess.py dataset/dpnval.tfrecords ./input_bins/
```
将会生成 **data** , **distance** 和 **label** 目录:
```
data
|___000000.bin
|___000001.bin
...

distance
|___000000.bin
|___000001.bin
...

label
|__label.npy
```

### 3. 离线推理

**离线模型转换**

- 环境变量设置

  请参考[说明](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719)，设置环境变量

- Pb模型转换为om模型
  
   [**pb模型下载链接**](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/007_inference_backup/dpn/dpn_tf_hw34064571/offline_inference/ckpt/dpn.pb)

  batchsize=8

  ```
  atc --model=dpn.pb  --framework=3 --input_shape="inputx:8,512,512,3,inputd:10,10,1" --output=./dpn_8batch --out_nodes="upsample/Conv_2/Relu:0" --soc_version=Ascend310 --log=info
  ```

- 编译程序

  ```
  bash build.sh
  ```
  **benchmark** 工具的运行结果将会生成在 **Benchmark/output/** 路径下:

- 开始运行:

  ```
  cd scripts
  bash benchmark_tf.sh
  ```



## 性能

### 结果

本结果是通过运行上面适配的推理脚本获得的。要获得相同的结果，请按照《快速指南》中的步骤操作。

#### 推理精度结果:

| Test Dataset | Number of pictures | MeanIou |
|--------------|-------------------|-------------------|
| cvcdb          | 1448             | 46%             |

## 参考
[1] https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/dpn/DPN_ID1636_for_TensorFlow
