中文|[English](README_EN.md)

# ECONet Tensorflow离线推理

此链接提供ECONet TensorFlow模型在NPU上离线推理的脚本和方法

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
cd Modelzoo-TensorFlow/ACL_TensorFlow/contrib/cv/ECONet_for_ACL
```

### 2. 下载数据集和预处理

在这里，我们使用UCF101数据集训练模型，您也可以使用hmdb51数据集训练模型。

1. 请自行下载UCF101数据集，并将其放入路径: **scripts/dataset/ucf101**

2. 测试数据集和标签预处理:
```
cd scripts
mkdir input_bins
python3 data_preprocess.py --dataset ucf101 --data_path ./dataset/ucf101 --output_path ./input_bins
```
在 **input_bins** 路径下，它将会生成 **ucf101** 和 **ucf101_label.pkl** 目录文件:
```
input_bins
|
|__ucf101
|______v_ApplyEyeMakeup_g01_c01.bin
|______v_ApplyEyeMakeup_g01_c02.bin
|______v_ApplyEyeMakeup_g01_c03.bin
...

|
|__ucf101_label.pkl

```

### 3. 离线推理

**离线模型转换**

- 环境变量配置

  请参考[说明](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719)，设置环境变量

- Pb模型转换为om模型

  [**pb模型下载链接**](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/006_train_backup/econet/ECONet_tf_paper99/scripts/ucf101_best.pb)
  
  batchsize=4

  ```
  atc --model=ucf101_best.pb  --framework=3 --input_shape="clip_holder:4,224,224,3" --output=./econet_ucf101_4batch --soc_version=Ascend310 --log=info
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

| Test Dataset | Number of pictures | Top1/Top5 |
|--------------|-------------------|-------------------|
| ucf101          | 3783             | 88.4%/98.2%             |

## 参考
[1] https://gitee.com/ascend/ModelZoo-TensorFlow/tree/master/contrib/TensorFlow/Research/cv/econet/ECONet_tf_paper99
