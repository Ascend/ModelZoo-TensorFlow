中文|[英文](readme_en.md)

# TextToTextTransferTransformer Tensorflow 2.x 在线推理
> 此链接提供 TextToTextTransferTransformer TensorFlow 2.x pb模型在NPU上在线推理的脚本和方法

# 注意
> 此案例仅为您学习Ascend软件栈提供学习参考，不用于商业目的。

在开始之前，请注意以下适配条件。如果不匹配，可能导致运行失败

|依赖|要求|
|---|---|
|CANN 版本|>=6.0.0|
|芯片平台|Ascend310/Ascend310P3|
|第三主依赖|请参考 requirements.txt|

## 1. 拷贝代码

```shell
git clone https://gitee.com/ascend/ModelZoo-TensorFlow.git
cd ModelZoo-TensorFlow/TensorFlow2/built-in/nlp/TextToTextTransferTransformer_ID4150_for_TensorFlow2.X
```

## 2. 下载pb模型
1. [下载pb模型](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/model/2022-12-12_tf/t5_tf2_online_inference/t5.pb)
2. 将pb模型放置在任意位置，建议路径为 ./model/t5.pb
3. 目录结构如下
```
TextToTextTransferTransformer_ID4150_for_TensorFlow2.x
|---model
|---|---t5.pb
|---main.py
|---questions.txt
|---readme.md
|---readme_en.md
|---requirements.txt
```

## 3. 在线推理

> 请准备好输入数据，即文本问题，可以是txt文件，也可以是字符串，以下两种方法皆可

```python3.7
python3 main.py -f questions.txt
python3 main.py 'question1' 'question2'
```

## 性能结果
本结果是通过运行上边适配的推理脚本获得的。

1. gpu 结果
|设备|结果|
|---|---|
|nq question: where is google's headquarters|in Columbus, Ohio|
|nq question: what is the most populous country in the world|China|
|nq question: name a member of the beatles|Harrison|
|nq question: how many teeth do humans have|20 primary|

1. npu 结果
|设备|结果|
|---|---|
|nq question: where is google's headquarters|in Columbus, Ohio|
|nq question: what is the most populous country in the world|China|
|nq question: name a member of the beatles|Harrison|
|nq question: how many teeth do humans have|20 primary|


## 参考
[谷歌t5网络 github地址](https://github.com/google-research/text-to-text-transfer-transformer)

