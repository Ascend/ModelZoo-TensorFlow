中文|[English](README_EN.md)

# Advanced_East_for_ACL
此存储库提供了推断Advanced_East模型的脚本和方法。

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
cd Modelzoo-TensorFlow/ACL_TensorFlow/built-in/cv/Advanced_East_for_ACL
```

### 2. 下载并预处理数据集ICPR MTWI 2018

1. 自行下载数据集
   ```
    tianchi ICPR MTWI 2018
   
   ```

2. 执行预处理脚本
  
   ```
   python3 script/preprocess.py

   ```
   ```
   python3 script/image2bin.py

   ```


### 3. 离线推理

**离线模型转换**

- 环境变量设置

  请参考[说明](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719)，设置环境变量
- Pb模型转换为h5模型

  ```
  python3 h5_to_pb.py

  ```
2. Pb模型转换为om模型

   [pb模型下载链接]()

    ```
   atc --model=model.pb --input_shape="input_img:1,736,736,3" --framework=3 --output=Advanced_East --soc_version=Ascend310 --input_format=NHWC 
  
    ```


- 编译程序

  ```
  bash build.sh
  ```

- 开始运行:

  ```
  bash benchmark_tf.sh --batchSize=1 --modelPath=../../model/Advanced_East.om --dataPath=../../datasets/ --modelType=Advanced_East  --imgType=rgb 

  ```
## 推理结果

### 结果

本结果是通过运行上面适配的推理脚本获得的。要获得相同的结果，请按照《快速指南》中的步骤操作。
  ```
   python3 predict.py

  ```

#### 推理精度结果

|       model       | **data**   |    precision    |    recall       |    heamn        |
| :---------------: | :-------:  | :-------------: | :-------------: | :-------------: |
| offline Inference | 1000 images |    84.91%       |    55.54%       |    63.57%       |
## 
