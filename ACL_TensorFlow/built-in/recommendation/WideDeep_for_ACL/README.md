中文|[English](README_EN.md)

# widedeep TensorFlow离线推理

此链接提供widedeep TensorFlow模型在NPU上离线推理的脚本和方法

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
git clone https://gitee.com/ascend/modelzoo.git
cd Modelzoo-TensorFlow/ACL/Research/recommendation/WideDeep_for_ACL
```

### 2. 下载数据集和预处理
[Weights files,access code:**ascend**](https://e-share.obs-website.cn-north-1.myhuaweicloud.com?token=ps4oG2NRMXJdKgQrPTQLN2xbRMW0y1ENptC+xgiaJBjyVS8X1vzxx3oBtbozGX5utCjbEdzYpBgLXj6IBanV++xpRfqeZY/WOqR7eGcKMbLsbPD/QvsqFYTIgPbYIFgtJnhmpp2t3V4akctZr6rbOc5bjzGaZiq/dU6r+GwXaemxgLzTBoUMrTNJdKsvD+QZBppUjiF23f6xOHwqUquOXr7HPlFtx+K0ImJOXbDdZlYoSJAb1wGZ6RgNgmWNX691n4hWhjGQ4qkqckVqOV+UZrRaca66qT7i+GMsd5TNb/iAQ6b8R9wpGIkqgS/y17gawgeqGlL3Hy0aEToOCYMUESrnw30waqxA7E5/ahP6GCO3brhvmkefNqA/8yweYnNB78Ii6Mc4cgM7fX7kWfJbsp7HqfTF39ywkQe2/ecCqJG8aQDG7yolKrYQOLUiP8+oRUYUSI3dERHDcXPDf4atAbZN9Y/1XXAhmhc+E3xy4HbIa+uZy0Oik3Jhkvl1i5zB2Gb83QdyIQuCKUclTJXo/OQo46BJl6HcWELY+UfqZhTXDy5FZWYrnjfzyzOdq+GY0n0/fofy2/8LrvUldHgmp2jojS6jroyEQvT03vkzOzixRDZrWtFUYQmTev7+YBprV+GfmWm60TU0Olznc65Gqrjqs1xtTHwglKqYn1/22z03wtreRdhgfLcrf3pN8RdybScmRWMp28Ro9gR+0lITa5ct22Lh+B4WhDRSc6SYckwsa58Kzg/D4ctRAcrHjsaQAz73nfVj1DX7qaIWVRPizi6uaSR5CZbAmjClX+FFlibpnLmvFlBJ4NSmKAOoxKx+VCw8szniylSmp2OFRQ70dQ==)

and put them to **'acl/bin'**

[测试数据集](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com:443/010_Offline_Inference/Official/cv/WideDeep_ID0028_for_ACL/acl/data/adult.test?AccessKeyId=APWPYQJZOXDROK0SPPNG&Expires=1656054988&Signature=UXJ9XFtUdgDeM2PDVesztNBzQXs%3D)

将测试数据集放入 **'acl/data/'**


### 3. 离线推理

**离线模型转换**

- 环境变量设置

  请参考[说明](https://gitee.com/ascend/ModelZoo-TensorFlow/wikis/02.%E7%A6%BB%E7%BA%BF%E6%8E%A8%E7%90%86%E6%A1%88%E4%BE%8B/Ascend%E5%B9%B3%E5%8F%B0%E6%8E%A8%E7%90%86%E7%8E%AF%E5%A2%83%E5%8F%98%E9%87%8F%E8%AE%BE%E7%BD%AE?sort_id=6458719)，设置环境变量

- Pb模型转换为om模型

  [pb模型下载链接](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/modelzoo/Official/recommendation/widedeep_tf.pb)

  ```
  atc --model=acl/model/widedeep_tf.pb --framework=3 --output=acl/model/widedeep_framework_tensorflow_aipp_0_batch_1_input_fp32_output_fp32 --soc_version=Ascend310 --input_shape="Input:1,51"
  ```

- 编译程序

  ```
  cd benchmark-master
  bash run.sh
  ```

- 开始运行:

  ```
  cd ../
  bash benchmark_tf.sh --batchSize=1 --modelPath=acl/model/widedeep_framework_tensorflow_aipp_0_batch_1_input_fp32_output_fp32.om --dataPath=acl/data/adult.test
  ```


## 推理结果

### 结果

本结果是通过运行上面适配的推理脚本获得的。要获得相同的结果，请按照《快速指南》中的步骤操作。

#### 推理精度结果

|       model       | **data**  |    Top1    |
| :---------------: | :-------: | :-------------: |
| offline Inference | 16281     | 85.44 % |

