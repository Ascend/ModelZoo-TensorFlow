中文|[English](README_EN.md)

# CHINESE-OCR Tensorflow离线推理

此链接提供CHINESE-OCR TensorFlow模型在NPU上离线推理的脚本和方法

## 注意
**此案例仅为您学习Ascend软件栈提供参考，不用于商业目的。**

在开始之前，请注意以下适配条件。如果不匹配，可能导致运行失败。

| Conditions | Need |
| --- | --- |
| CANN版本 | >=5.0.3 |
| 芯片平台| Ascend310/Ascend310P3 |

## 快速指南

### 1. 拷贝代码

```shell
git clone https://gitee.com/ascend/ModelZoo-TensorFlow.git
cd Modelzoo-TensorFlow/ACL_TensorFlow/contrib/nlp/CHINESE-OCR_ID2090_for_ACL
```

### 2. 下载pb模型及模型转换

1. 请下载pb模型 ([pb模型和om模型下载](https://pan.baidu.com/s/1gNDUcZa5VrRf0-JzCzeJKQ?pwd=3vpg 
)，提取码：3vpg)并将其放入: **model/** 中：

2. 离线模型转换:

- 环境变量设置

  请参考[说明](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719)，设置环境变量

- pb模型转换为om模型

  ```
  atc --model=./VGGnet_fast_rcnn.pb --framework=3 --output=./VGGnet_fast_rcnn --soc_version=Ascend310 --input_shape="Placeholder:1,900,900,3" --out_nodes="rpn_bbox_pred/Reshape_1:0;Reshape_2:0"
  ```


### 3. 离线推理
- 准备推理数据和模型
  
  将所需要推理的图片放入 **data/** 中，om模型放入 **model/** 中

- 开始运行:

  ```
  cd src
  python3 ocr.py
  ```




## 结果

### 查看推理结果

在 **out/** 中查看推理图片的结果。本结果是通过运行上面适配的推理脚本获得的。要获得相同的结果，请按照《快速指南》中的步骤操作。

## 参考
[1] https://gitee.com/yingzhidaofei/ModelZoo-TensorFlow/tree/master/TensorFlow/contrib/nlp/CHINESE-OCR_ID2090_for_Tensorflow
