中文|[English](README_EN.md)

# PWCNet TensorFlow离线推理

此链接提供PWCNet TensorFlow模型在NPU上离线推理的脚本和方法

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
cd Modelzoo-TensorFlow/ACL_TensorFlow/contrib/cv/PWCNet_for_ACL
```

### 2. 下载数据集和预处理
1. 请自行下载MPI Sintel测试数据集遵循[指南](https://github.com/philferriere/tfoptflow) 把预处理好数据集放到**scripts/dataset/MPI-Sintel-complete**目录下

2. 测试数据集和标签的预处理：
```
cd scripts
mkdir input_bins
python3 data_preprocess.py --dataset ./dataset--output ./input_bins
```
将在**input_bins**目录下生成 **image** 和**gt**目录:
```
input_bins
|
|__image
   |______alley_1-frames_0001_0002.bin
   |______alley_1-frames_0002_0003.bin
   |______alley_1-frames_0003_0004.bin
...

|
|__gt
   |______alley_1-frames_0001_0002.bin
   |______alley_1-frames_0002_0003.bin
   |______alley_1-frames_0003_0004.bin
...

```

### 3. 离线推理

**Pb模型转换为om模型**

- 环境变量设置

  请参考[说明](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719)，设置环境变量

- Pb模型转换为om模型
  
  [**pb模型下载链接**](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/006_train_backup/PWCNet_tf_wosaisai/offline_infer/pwcnet.pb)

  batchsize=1

  ```
  atc --model=pwcnet.pb  --framework=3 --input_shape="x_tnsr:1,2,448,1024,3" --output=./pwcnet_1batch --soc_version=Ascend310 --log=info
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



## 性能

### 结果

本结果是通过运行上面适配的推理脚本获得的。要获得相同的结果，请按照《快速指南》中的步骤操作。

#### 推理精度结果

| Test Dataset | Number of pictures | EPE |
|--------------|-------------------|-------------------|
| MPI Sintel          | 1041             | 1.25             |

## 参考
[1] https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/cv/pwcnet/PWCNet_ID0171_for_TensorFlow
