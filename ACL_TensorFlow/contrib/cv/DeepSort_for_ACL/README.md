中文|[English](README_EN.md)

# DeepSort Tensorflow离线推理

此链接提供DeepSort TensorFlow模型在NPU上离线推理的脚本和方法

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
cd Modelzoo-TensorFlow/ACL_TensorFlow/contrib/cv/DeepSort_for_ACL
```

### 2. 下载数据集和预处理

1. 下载MOT16数据集，并将 **test/MOT16-03** 放入 **scripts/dataset/test/** 路径中：

2. 我们仅使用一个测试数据集(**MOT16-03**)作为样例
```
dataset/test
|
|__MOT16-03
   |______det
   |______img1
   |______seqinfo.ini

```

### 3. 离线推理

**离线模型转换**

- 环境变量设置

  请参考[说明](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719)，设置环境变量


- Pb模型转换为om模型
  
  [**pb模型下载链接**](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/modelzoo/Research/cv/DeepSort_for_ACL/mars-small128.pb)

  动态batchsize

  ```
  atc --model=mars-small128.pb  --framework=3 --input_shape_range="images:[-1,128,64,3]" --output=./deepsort_dynamic_batch --soc_version=Ascend310 --log=info
  ```

-编译程序

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

#### 推理结果:

[样例视频](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/modelzoo/Research/cv/DeepSort_for_ACL/MOT16-03.avi)

## 参考
[1] https://github.com/nwojke/deep_sort
